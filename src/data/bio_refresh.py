"""Keep member biographies current from the group's own conversation.

`data/group_members.json` was generated once, on 2026-07-20, and froze. The group
has talked to the bot constantly since, and none of it reaches the profiles —
which is why the bot still calls somebody the group's "ladies man" years after he
stopped talking about it, and knows nothing at all about the quieter members.

Three things had to be true for a refresh to run on a schedule, and each is why a
piece of this module looks the way it does.

**It writes the field the prompt actually reads.** ``key_facts`` is what
``build_member_prompt_suffix`` injects into every system prompt, and until now
nothing wrote it — it was hand-curated. ``biography_summary`` IS generated, but
only append-only, and only used as a fallback. So the pipeline that existed kept
updating the field the model does not read. This one distils ``key_facts``.

**It reads the live corpus.** ``generate_knowledge_base.py`` reads the static
July export. The conversation lands in ``data/live_messages/``.

**It does not need the GPU that prod is using.** The box already serves a capable
12B through llama-server, and ``src/chat/summary.py`` proves the pattern: take
``gpu_section`` with a short timeout and give up on ``GpuBusyError``. No second
teacher model, so no outage.

Two rules are load-bearing and neither is an optimisation:

* **Shared scope only.** ``group_members.json`` is injected into EVERY chat's
  system prompt, so a fact learned in a DM would become readable by the whole
  group. That is exactly what ``src/chat/scope.py`` exists to prevent, and
  profiles are the one store that would otherwise ignore it.
* **It proposes, it never writes.** A cycle produces
  ``data/bio_proposals.json``; a human accepts through ``scripts/review_bios.py``.
  These are statements about real people, made by a model, with nobody watching.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.generate_knowledge_base import (
    build_extraction_prompt,
    chunk_messages,
    format_chunk_for_prompt,
    get_mentioned_members,
    strip_markdown_fences,
)
from src.data.identity_resolver import SenderResolver
from src.data.ingest import strip_failed_descriptions
from src.data.term_blocklist import compile_blocklist, filter_list

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATE = "data/bio_refresh_state.json"
DEFAULT_PROPOSALS = "data/bio_proposals.json"

# What the extractor is allowed to record. The existing EXTRACTION_SYSTEM_PROMPT
# says "do not invent facts" and stops there, which is not enough for this
# corpus: half of a lively evening is people baiting the bot and insulting each
# other. Extracted verbatim, "paneleiro oficial" becomes a permanent recorded
# fact about a real person — the exact failure this whole effort exists to undo.
DURABLE_FACTS_SYSTEM = """You are a meticulous archivist building factual profiles of members of a Portuguese friend group called "Kaya", from their group chat.

Record ONLY durable facts — things that would still be true next month:
  work and studies, where they live or travel, family and relationships, pets,
  sports and hobbies they actually do, health events, significant purchases,
  things that happened to them.

Day-to-day chatter is NOT a fact, however clearly it is stated. Do not record
who is free on a given day, who is coming to a dinner, who is running late,
what someone is doing this weekend, or anything that stops being true once the
week ends. "X is unavailable until Sunday" and "X may not be in Lisbon on the
23rd" are the kind of thing to leave out entirely.

NEVER record:
  - insults, nicknames, teasing, or who the group mocks;
  - opinions, verdicts or rankings about a person ("the funniest", "the dumbest",
    "the group's X"), even when everyone agrees;
  - what a person argued or believes about politics, religion, sex or any other
    sensitive subject, however plainly they said it. That someone takes part in
    a debate is not a fact worth keeping about them;
  - anything said TO or ABOUT the chatbot, or about the chat itself;
  - anything a person said about someone else's character;
  - jokes, sarcasm and exaggeration read as fact.

A fact must be supported by the messages. If a chunk contains nothing durable
about a member, omit that member entirely. Prefer omitting to guessing.

Return ONLY a valid JSON object, no markdown and no explanation:
{
  "members": {
    "MemberName": {
      "facts": ["one short factual sentence", "another"],
      "occupation": "job or studies, or null",
      "living_place": "city/country, or null",
      "interests": ["only things they actually do"]
    }
  }
}

Write the facts in English, one clause each, naming the member explicitly
("Rafa trains kickboxing"), so they read correctly on their own.
Only include members named in the list you are given."""

DISTIL_SYSTEM = """You maintain the fact list for one member of a Portuguese friend group.

You are given what is currently known about them and any new facts learned since.
Return at most {limit} facts as a JSON array of short sentences, and nothing else.

Rules, in order of importance:
  - COPY an existing fact VERBATIM when it is still true. Do not reword it, do
    not reformat it, do not drop or add parentheses. Rewrite one only when it is
    actually wrong or has been superseded by something newer.
  - Only DROP a fact when the new material contradicts it or clearly makes it
    out of date. Old and rarely mentioned is not the same as untrue: things that
    happened to a person stay true.
  - Add the new facts that are worth keeping, merging any that duplicate an
    existing one instead of listing both.
  - Every fact must name the person by the exact name in "Person:" below. Never
    use their chat display name or any other form of it.
  - Never invent anything that is not in the material you were given.
  - Never include insults, nicknames, or opinions about the person's character.
  - Facts listed as PINNED must be kept verbatim.

Most of the list should normally come back unchanged. Return only the JSON array."""


class BioRefreshState:
    """How far into the group's history the last successful cycle got.

    Mirrors ``IngestState`` (``src/data/ingest.py``) deliberately, including the
    atomic write: the same "catch up on what was missed, never re-read" job, and
    the same crash behaviour. Advanced only after a cycle produces proposals, so
    a failure re-reads rather than skips.
    """

    def __init__(self, path: str = DEFAULT_STATE):
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = BASE_DIR / path
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @property
    def watermark(self) -> int:
        return int(self._data.get("last_ts", 0))

    def advance(self, ts: int, members_touched: int) -> None:
        self._data["last_ts"] = int(ts)
        self._data["last_run"] = datetime.now(timezone.utc).isoformat()
        self._data["total_cycles"] = int(self._data.get("total_cycles", 0)) + 1
        self._data["total_members_touched"] = (
            int(self._data.get("total_members_touched", 0)) + int(members_touched))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("could not save bio refresh state to %s: %s", self.path, exc)


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("bio_refresh", {}) or {}


def blocklist(config: Dict[str, Any]):
    """Terms that may never appear in a proposed fact.

    Reuses ``data.blocked_terms``, already applied to key_facts by the batch
    pipeline, so there is one list to maintain rather than two.
    """
    return compile_blocklist((config.get("data", {}) or {}).get("blocked_terms") or [])


def _bot_tokens(config: Dict[str, Any]) -> List[str]:
    """``@<id>`` mention tokens that mean a message is aimed at the bot."""
    wcfg = config.get("whatsapp", {}) or {}
    ids = [wcfg.get("bot_jid", ""), wcfg.get("bot_lid", "")]
    ids += list(wcfg.get("bot_jids", []) or [])
    return [f"@{str(i).split('@', 1)[0]}" for i in ids if i]


def is_about_the_bot(text: str, bot_tokens: List[str]) -> bool:
    """Whether this message is talking to, or about, the bot.

    A large share of the group's recent traffic is people testing it — "arguing
    with a machine is wild", "este bot tem de se acalmar", instructions, roast
    requests. None of that is biography, and feeding it back would fill the
    profiles with the group's opinions of the chatbot.
    """
    lowered = (text or "").lower()
    if any(token.lower() in lowered for token in bot_tokens):
        return True
    return any(cue in lowered for cue in (
        "kaya bot", "o bot ", "este bot", "esse bot", "the bot", "chatbot",
        "prompt", "modelo de linguagem",
    ))


def read_new_messages(
    config: Dict[str, Any],
    state: BioRefreshState,
    resolver: SenderResolver,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    """Group messages worth extracting from. ``(messages, unresolved, newest_ts)``.

    Only the shared scope is ever read, and the scope is checked per row rather
    than inferred from the filename — a profile is injected into every chat, so
    anything learned here is effectively public to the group.

    Senders are resolved to canonical members and anything unresolved is
    REPORTED, never invented. An unmapped display name is a mapping waiting to be
    added; treating it as a new person is how a phantom member gets a profile.
    """
    from src.data.message_log import MessageLog

    wcfg = config.get("whatsapp", {}) or {}
    log = MessageLog(base_dir=wcfg.get("message_log_dir", "data/live_messages"))
    bot_tokens = _bot_tokens(config)

    kept: List[Dict[str, Any]] = []
    unresolved: Dict[str, int] = {}
    newest = state.watermark

    for row in log.read("shared", after_ts=state.watermark):
        if row.get("scope") != "shared":
            continue
        newest = max(newest, int(row.get("timestamp") or 0))

        text = strip_failed_descriptions((row.get("text") or "").strip())
        if not text or is_about_the_bot(text, bot_tokens):
            continue

        sender = resolver.resolve(row.get("sender") or "")
        if not resolver.is_member(sender):
            unresolved[row.get("sender") or "?"] = unresolved.get(
                row.get("sender") or "?", 0) + 1
            continue

        kept.append({
            "id": row.get("id", ""),
            "sender": sender,
            "text": text,
            # format_chunk_for_prompt slices this as an ISO string; the live log
            # stores a unix int.
            "timestamp": datetime.fromtimestamp(
                int(row.get("timestamp") or 0), tz=timezone.utc).isoformat(),
        })

    return kept, unresolved, newest


def member_alias_map(members: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """``{name: [aliases]}`` in the shape ``get_mentioned_members`` expects."""
    return {
        m["name"]: [m["name"].lower()] + [str(a).lower() for a in (m.get("aliases") or [])]
        for m in members if m.get("name")
    }


def _generate(backend: Any, config: Dict[str, Any], system: str, user: str,
              max_new_tokens: int) -> str:
    """One call to the already-loaded serving model, under the GPU lock.

    The lock is taken per call and released between them, so a cycle never sits
    on it for the whole run: ``whatsapp_server._process`` DROPS an inbound
    message when the lock is contended, and a refresh must never cost somebody
    their reply.
    """
    from src.chat.gpu_lock import gpu_section

    timeout = float(_config(config).get("lock_timeout_seconds", 20))
    with gpu_section(config, timeout=timeout):
        return backend.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_new_tokens=max_new_tokens,
            sampling={"temperature": 0.2, "top_p": 0.9, "top_k": 0,
                      "repetition_penalty": 1.05},
        )


def _comparable(fact: str) -> str:
    """A fact stripped to its content, for spotting restatements.

    Parentheticals go because the alias list is the usual thing a rewrite drops:
    "Rafa (Rafael, Chamusca) is married to Mel" and "Rafa is married to Mel" are
    the same fact, and showing them as one removal and one addition makes the
    review a wall of noise nobody will read.
    """
    import re as _re

    text = _re.sub(r"\([^)]*\)", " ", (fact or "").lower())
    return " ".join(_re.sub(r"[^\w\s]", " ", text).split())


def settle_wording(proposed: List[str], current: List[str],
                   threshold: float = 0.82) -> List[str]:
    """Keep the existing wording wherever a proposed fact merely restates one.

    The model was asked to copy still-true facts verbatim and mostly does, but it
    cannot resist tidying. Every removal in the first two dry runs was a
    parenthetical being stripped, not a fact changing. This makes "unchanged"
    actually mean unchanged, so a real removal stands out.
    """
    from difflib import SequenceMatcher

    settled = []
    for fact in proposed:
        target = _comparable(fact)
        match = next(
            (existing for existing in current
             if SequenceMatcher(None, target, _comparable(existing)).ratio() >= threshold),
            None,
        )
        settled.append(match if match else fact)
    # A restatement can collapse onto a fact already in the list.
    return list(dict.fromkeys(settled))


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(strip_markdown_fences((raw or "").strip()))
    except (ValueError, TypeError):
        return None


def extract_facts(backend: Any, config: Dict[str, Any], messages: List[Dict[str, Any]],
                  members: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """New durable facts per member, each carrying the messages behind it."""
    cfg = _config(config)
    aliases = member_alias_map(members)
    profiles = {m["name"]: m for m in members}
    chunk_words = int(cfg.get("chunk_size_words", 1200))
    max_chunks = int(cfg.get("max_chunks_per_cycle", 4))

    found: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunk_messages(messages, chunk_words)[:max_chunks]:
        mentioned = get_mentioned_members(chunk, aliases)
        if not mentioned:
            continue
        prompt = build_extraction_prompt(
            format_chunk_for_prompt(chunk), profiles, mentioned,
            ["name", "occupation", "living_place", "interests"])
        parsed = _parse_json(
            _generate(backend, config, DURABLE_FACTS_SYSTEM, prompt,
                      int(cfg.get("extract_max_new_tokens", 900))))
        if not isinstance(parsed, dict):
            logger.warning("extraction returned unparseable output; skipping chunk")
            continue

        evidence = [m["id"] for m in chunk if m.get("id")]
        for name, payload in (parsed.get("members") or {}).items():
            if name not in profiles or not isinstance(payload, dict):
                continue
            for fact in (payload.get("facts") or []):
                if isinstance(fact, str) and fact.strip():
                    found.setdefault(name, []).append(
                        {"fact": fact.strip(), "evidence": evidence})
    return found


def distil_key_facts(backend: Any, config: Dict[str, Any], member: Dict[str, Any],
                     new_facts: List[str], limit: int) -> Optional[List[str]]:
    """A ranked, capped ``key_facts`` list. None when it cannot be produced.

    The input is bounded on purpose. ``biography_summary`` is append-only and
    already 6-9 KB per member; handing the model all of it every cycle would be
    slow, expensive, and would bury this week's facts under two years of them.
    """
    pinned = [f for f in (member.get("pinned_facts") or []) if str(f).strip()]
    current = list(member.get("key_facts") or [])
    bio_tail = (member.get("biography_summary") or "")[-1500:]

    if not new_facts and current:
        return None  # nothing new to say about this member

    blocks = [f"Person: {member['name']}"]
    if pinned:
        blocks.append("PINNED (keep verbatim):\n" + "\n".join(f"- {f}" for f in pinned))
    if current:
        blocks.append("Currently known:\n" + "\n".join(f"- {f}" for f in current))
    if new_facts:
        blocks.append("Newly learned:\n" + "\n".join(f"- {f}" for f in new_facts))
    if bio_tail:
        blocks.append(f"Background (older, for context only):\n{bio_tail}")

    parsed = _parse_json(_generate(
        backend, config, DISTIL_SYSTEM.format(limit=limit), "\n\n".join(blocks),
        int(_config(config).get("distil_max_new_tokens", 500))))
    if not isinstance(parsed, list):
        logger.warning("distillation for %s returned unparseable output", member["name"])
        return None

    facts = [str(f).strip() for f in parsed if str(f).strip()]
    # A fact that never says who it is about is useless: the member list is
    # shuffled per prompt, so "Wedding planned for next September" attaches to
    # nobody. Dropped rather than repaired — guessing the subject is how a fact
    # ends up on the wrong person.
    facts = [f for f in facts if member["name"].lower() in f.lower()]
    # The blocklist is the deterministic half of "never record this". The prompt
    # asks and mostly gets its way; this does not ask. `filter_list` is
    # documented for key_facts specifically, and `data.blocked_terms` is where
    # the group's own no-go words already live.
    facts = filter_list(facts, blocklist(config))
    facts = settle_wording(facts, current)
    # A pinned fact the model dropped is put back: "pinned" has to mean pinned.
    for fact in pinned:
        if fact not in facts:
            facts.insert(0, fact)
    return facts[:limit] or None


def run_cycle(config: Dict[str, Any], backend: Any,
              members_file: Optional[Path] = None,
              state_path: Optional[str] = None,
              proposals_path: Optional[str] = None) -> Dict[str, Any]:
    """One refresh. Writes proposals and returns a summary. Never raises.

    Nothing here touches ``group_members.json``. The output is a proposal a
    person accepts or rejects — these are statements about real people, produced
    by a model, and the review is the point at which a human sees them.
    """
    from src.chat.gpu_lock import GpuBusyError

    started = time.time()
    cfg = _config(config)
    members_path = Path(members_file or (config.get("data", {}) or {}).get(
        "group_members_file", "data/group_members.json"))
    if not members_path.is_absolute():
        members_path = BASE_DIR / members_path

    state = BioRefreshState(state_path or cfg.get("state_file", DEFAULT_STATE))
    resolver = SenderResolver(members_path,
                              (config.get("data", {}) or {}).get("sender_aliases") or {})
    members = json.loads(members_path.read_text(encoding="utf-8"))["members"]
    by_name = {m["name"]: m for m in members}

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "messages_read": 0, "unresolved_senders": {}, "proposals": {}, "skipped": "",
    }

    try:
        messages, unresolved, newest = read_new_messages(config, state, resolver)
        report["messages_read"] = len(messages)
        report["unresolved_senders"] = unresolved
        if not messages:
            report["skipped"] = "nothing new"
            return report

        new_facts = extract_facts(backend, config, messages, members)
        # How many facts are STORED, which is not how many are shown. The prompt
        # cap (rag.max_facts_per_member) is applied at injection time by
        # build_member_prompt_suffix, so distilling down to it would make every
        # accepted proposal delete history: the first dry run had a member go
        # from six facts to four, discarding an assault he had reported to the
        # police. Store generously, inject narrowly.
        limit = int(cfg.get("max_key_facts", 8))

        for name, entries in new_facts.items():
            member = by_name[name]
            facts = [e["fact"] for e in entries]
            proposed = distil_key_facts(backend, config, member, facts, limit)
            if not proposed:
                continue
            current = list(member.get("key_facts") or [])
            if proposed == current:
                continue
            report["proposals"][name] = {
                "current": current,
                "proposed": proposed,
                "added": [f for f in proposed if f not in current],
                "removed": [f for f in current if f not in proposed],
                "new_facts": entries,
            }
    except GpuBusyError:
        # The watermark is untouched, so the next cycle re-reads this window.
        # Being late costs nothing; taking the lock from a live reply does.
        report["skipped"] = "GPU busy"
        logger.info("bio refresh deferred — GPU busy")
        return report
    except Exception as exc:  # noqa: BLE001 — a refresh must never take the bot down
        report["skipped"] = f"error: {exc}"
        logger.warning("bio refresh failed: %s", exc)
        return report

    _write_proposals(config, report, proposals_path)
    state.advance(newest, len(report["proposals"]))
    report["seconds"] = round(time.time() - started, 1)
    logger.info("bio refresh: %d msg, %d proposal(s), %.0fs",
                report["messages_read"], len(report["proposals"]), time.time() - started)
    return report


def proposals_path(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    path = Path(override or _config(config).get("proposals_file", DEFAULT_PROPOSALS))
    return path if path.is_absolute() else BASE_DIR / path


def _write_proposals(config: Dict[str, Any], report: Dict[str, Any],
                     override: Optional[str] = None) -> None:
    """Merge this cycle's proposals into the pending set.

    Merged rather than replaced: review happens when somebody gets round to it,
    and a cycle two hours later must not silently discard what the last one
    found. ``current`` always reflects the newest read of the profile.
    """
    path = proposals_path(config, override)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}

    pending = existing.get("proposals", {}) if isinstance(existing, dict) else {}
    for name, proposal in report["proposals"].items():
        previous = pending.get(name) or {}
        proposal["new_facts"] = (previous.get("new_facts") or []) + proposal["new_facts"]
        pending[name] = proposal

    payload = {
        "generated_at": report["generated_at"],
        "unresolved_senders": report["unresolved_senders"],
        "proposals": pending,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("could not write proposals to %s: %s", path, exc)
