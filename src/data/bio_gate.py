"""Which proposed profile facts reach the bot on their own, and which wait for Gustavo.

Until 2026-11-19 the refresh only PROPOSED (``data/bio_proposals.json``) and a
person had to run ``scripts/review_bios.py``. Nobody was ever asked, so 455 facts
piled up for three months, real ones (a member's past job, from his own words) next
to noise ("X works."). Asked who someone was, the bot filled the gap from retrieved
history and gave him another member's job.

A fact now goes one of three ways (Gustavo's choice, 2026-11-19):

* **dropped**: it doesn't name its member, it's trivial, or its quote is about
  somebody else (Bruno writing "quem trabalhava na Acme era a Ana" is a fact about
  Ana, never about Bruno).
* **asked**: it's sensitive (politics, health, relationships, money, religion, the
  police) or the model wasn't sure. These go to Gustavo once a week on WhatsApp
  (``digest_text`` / ``apply_answer``), at most ten, and expire after four weeks.
* **waiting**: everything else needs a second, different quote; one that never gets
  it is dropped after four weeks, so his list carries no noise.
* **applied**: it was seen in at least two different quotes, and the local model
  confirms the quotes say it about that person. It is added to ``key_facts`` with a
  snapshot, and logged with its quotes, so it can be undone.

Every state lives in ``data/bio_queue.json``. Nothing here raises into the bot.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

BIO_TAG = "👤"
DEFAULT_QUEUE = "data/bio_queue.json"
SAME_FACT = 0.72             # SequenceMatcher ratio for "the same fact, reworded"
CANDIDATE_DAYS = 28          # a fact seen once waits this long for a second quote, then is dropped
ASK_EXPIRE_DAYS = 28         # an asked fact nobody answered is dropped after this
DIGEST_SIZE = 10
DIGEST_EVERY_DAYS = 7

# Sensitive in either language. Applying these needs his yes: they're statements about
# real people that the whole group reads back.
_SENSITIVE = re.compile(
    r"(?i)\b(pol[ií]tic\w*|partid\w*|chega|ventura|vot(?:e|es|ed|ing|a|ou|ar)\w*|left-wing|right-wing|"
    r"esquerda|direita|conservat\w*|liberal\w*|socialis\w*|communis\w*|religi\w*|church|igreja|"
    r"health|sa[uú]de|doen[cç]\w*|ill(?:ness)?|sick|hospital\w*|surgery|cirurgi\w*|cancer|cancro|"
    r"depress\w*|anxiety|ansiedade|therap\w*|terapi\w*|medicat\w*|drug\w*|droga\w*|alcohol\w*|"
    r"[aá]lcool|sobriety|sobriedade|sex|sexo|sexual\w*|sexy|girlfriend|boyfriend|namorad\w*|wife|husband|"
    r"cigar\w*|smok\w*|tobacco|tabaco|fum(?:a|ar|o|ador)\w*|vape\w*|weed|marijuana|charro\w*|"
    r"methadone|metadona|anest\w*|"
    r"mulher|marido|dating|relationship\w*|rela[cç][aã]o|divorc\w*|pregnan\w*|gr[aá]vid\w*|"
    r"debt|d[ií]vida\w*|salary|sal[aá]rio|broke|police|pol[ií]cia|assault\w*|arrest\w*|preso)\b")
# Moments, not facts: the extraction prompt asks for durable facts and mostly gets them.
_TRIVIAL = re.compile(
    r"(?i)\b(photo|picture|foto|imagem|sticker|meme|emoji|laugh\w*|joked|tired|cansad\w*|"
    r"busy|ocupad\w*|running late|atrasad\w*|this week|today|tonight|hoje|weekend|"
    r"fim de semana)\b"
    # opinions, plans and where someone happens to be are not facts about him
    r"|\b(believes?|thinks?|feels?|opinion|prefers?|dislikes?|does not like|doesn.t like|wants? to|plans? to|planning|considering|would|"
    r"might|may|will be|is (?:in|at) [A-ZÀ-Ý][\w ]*\.?$)")
_WORD = re.compile(r"\w+", re.UNICODE)
# The refresh thread, the weekly sender and his answer all write data/bio_queue.json.
QUEUE_LOCK = threading.Lock()

GROUNDING_SYSTEM = (
    "You check facts about members of a Portuguese friend group against their own "
    "messages. Answer with one word: YES or NO.")


# ── small helpers ────────────────────────────────────────────────────────────

def _searchable(text: str) -> str:
    return re.sub(r"[^\w]", "", (text or "").lower())


def _comparable(fact: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", (fact or "").lower())
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


_STOP = {"a", "an", "the", "to", "at", "in", "on", "of", "and", "is", "was", "has", "had", "used",
         "for", "with", "his", "her", "he", "she", "as", "o", "a", "de", "do", "da", "em", "no", "na"}


def _content(fact: str) -> set:
    words = (w for w in _comparable(fact).split() if w not in _STOP and len(w) > 1)
    return {w[:-2] if w.endswith("ed") else w[:-1] if w.endswith("s") else w for w in words}


def same_fact(a: str, b: str) -> bool:
    """The same fact, reworded: close wording, or one's content words inside the other's
    ("Ana used to work at Acme" and "Ana worked at Acme, in analytics and sales")."""
    if SequenceMatcher(None, _comparable(a), _comparable(b)).ratio() >= SAME_FACT:
        return True
    small, large = sorted((_content(a), _content(b)), key=len)
    return len(small) >= 3 and len(small & large) >= 0.8 * len(small)


def _names(member: Dict[str, Any]) -> List[str]:
    out = [str(member.get("name") or "")] + [str(a) for a in (member.get("aliases") or [])]
    return [n.lower() for n in out if len(n.strip()) >= 3]


def _mentions(text: str, names: Iterable[str]) -> bool:
    low = f" {(text or '').lower()} "
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", low) for n in names)


def _item_id(member: str, fact: str) -> str:
    return hashlib.sha1(f"{member}|{_comparable(fact)}".encode()).hexdigest()[:10]


def _now() -> float:
    return time.time()


# ── the decision ─────────────────────────────────────────────────────────────

def speaker_of(entry: Dict[str, Any], lookup: Dict[str, Dict[str, str]]) -> str:
    """Who wrote the message the quote was copied from ("" when it can't be found).

    ``evidence`` lists every message id in the chunk the fact came from, not the one
    line, so the line is found by the quote itself."""
    needle = _searchable(entry.get("quote", ""))
    if len(needle) < 8:
        return ""
    for mid in entry.get("evidence") or []:
        row = lookup.get(mid)
        if row and needle in _searchable(row.get("text", "")):
            return row.get("sender", "")
    return ""


def decide(entry: Dict[str, Any], member: Dict[str, Any], others: List[Dict[str, Any]],
           speaker: str) -> Tuple[str, str]:
    """``(verdict, reason)``, verdict being ``drop``, ``ask`` or ``candidate``."""
    fact, quote = str(entry.get("fact") or ""), str(entry.get("quote") or "")
    name = str(member.get("name") or "")
    if quote.lstrip().startswith("["):
        # "[Imagem: um homem…]": a description says what is seen, never who it is.
        return "drop", "the quote is a media description"
    if not _mentions(fact, _names(member)):
        return "drop", "the fact doesn't say who it is about"
    rest = [w for w in _WORD.findall(fact) if w.lower() not in _names(member)]
    if len(rest) < 2 or _TRIVIAL.search(fact):
        return "drop", "trivial"
    other_names = [n for o in others for n in _names(o)]
    if _mentions(quote, _names(member)):
        pass                                   # the quote names this member
    elif speaker == name and not _mentions(quote, other_names):
        pass                                   # he said it about himself
    else:
        return "drop", "the quote is about somebody else"
    if _SENSITIVE.search(fact) or _SENSITIVE.search(quote):
        return "ask", "sensitive"
    return "candidate", ""


def grounded(generate: Callable[[str, str], str], member: str, fact: str,
             quotes: List[Dict[str, str]]) -> bool:
    """The local model's yes: do these lines state this fact about this person?"""
    lines = "\n".join(f"- «{q['quote']}» (written by {q.get('speaker') or 'someone'})"
                      for q in quotes[:4])
    user = (f"Messages:\n{lines}\n\nDo these messages clearly state that: \"{fact}\"? "
            f"Answer NO if it is about whoever wrote a line or anybody else instead of {member}, "
            f"if a line is a joke, a question or sarcasm, if it is only a plan, a wish or "
            f"something happening that day, or if the fact says more than the lines do. "
            f"Answer YES or NO.")
    answer = (generate(GROUNDING_SYSTEM, user) or "").strip().upper()
    return answer.startswith(("YES", "SIM"))


# ── the queue file ───────────────────────────────────────────────────────────

class BioQueue:
    """``data/bio_queue.json``: candidates, the ask list, what was applied, the digest."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        for key in ("candidates", "ask", "applied"):
            self.data.setdefault(key, [])
        self.data.setdefault("dropped", 0)
        self.data.setdefault("asked", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _find(self, key: str, member: str, fact: str) -> Optional[Dict[str, Any]]:
        return next((i for i in self.data[key]
                     if i["member"] == member and same_fact(i["fact"], fact)), None)

    def known(self, member: str, fact: str) -> bool:
        return any(self._find(k, member, fact) for k in ("candidates", "ask", "applied"))

    def add(self, key: str, member: str, entry: Dict[str, Any], speaker: str,
            reason: str = "") -> Dict[str, Any]:
        """Add the fact (or one more quote for it) to ``key``; returns the item."""
        quote = {"quote": str(entry.get("quote") or ""), "speaker": speaker}
        item = self._find(key, member, entry["fact"])
        if item is None:
            item = {"id": _item_id(member, entry["fact"]), "member": member,
                    "fact": str(entry["fact"]).strip(), "quotes": [], "reason": reason,
                    "first_seen": _now()}
            self.data[key].append(item)
        if _searchable(quote["quote"]) not in {_searchable(q["quote"]) for q in item["quotes"]}:
            item["quotes"].append(quote)
        return item

    def move(self, item: Dict[str, Any], src: str, dst: str, reason: str = "") -> None:
        self.data[src] = [i for i in self.data[src] if i is not item]
        if reason:
            item["reason"] = reason
        item["moved_at"] = _now()
        self.data[dst].append(item)


# ── applying ─────────────────────────────────────────────────────────────────

def snapshot(members_path: Path) -> Path:
    """Copy the profiles before writing (the same folder review_bios.py uses)."""
    target = members_path.parent / "profile_snapshots"
    target.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = target / f"group_members.{stamp}.json"
    serial = 1
    while path.exists():
        path = target / f"group_members.{stamp}-{serial}.json"
        serial += 1
    shutil.copy2(members_path, path)
    return path


def write_members(members_path: Path, payload: Dict[str, Any]) -> None:
    tmp = members_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(members_path)


def apply_items(members_path: Path, items: List[Dict[str, Any]], limit: int,
                how: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Add each item's fact to its member's ``key_facts``. ``(applied, full)``.

    A fact that restates one already there counts as applied (nothing to write). A
    member already at ``limit`` facts is not trimmed to make room: that deletes
    history (bio_refresh's own lesson), so the item goes back to be asked."""
    doc = json.loads(members_path.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in doc.get("members", [])}
    applied, full, changed = [], [], False
    for item in items:
        member = by_name.get(item["member"])
        if member is None:
            continue
        facts = list(member.get("key_facts") or [])
        if not any(same_fact(f, item["fact"]) for f in facts):
            if len(facts) >= limit:
                full.append(item)
                continue
            facts.append(item["fact"])
            member["key_facts"] = facts
            changed = True
        item["applied"] = how
        applied.append(item)
    if changed:
        snap = snapshot(members_path)
        write_members(members_path, doc)
        for item in applied:
            item["snapshot"] = str(snap)
    return applied, full


# ── one pass over newly extracted facts ──────────────────────────────────────

def gate(new_facts: Dict[str, List[Dict[str, Any]]], members: List[Dict[str, Any]],
         lookup: Dict[str, Dict[str, str]], queue: BioQueue,
         generate: Optional[Callable[[str, str], str]], members_path: Path,
         limit: int) -> Dict[str, int]:
    """Sort this batch, promote what is ready, apply what passes. Returns counts.

    ``generate`` is None when the GPU can't be had: candidates then just wait."""
    by_name = {m["name"]: m for m in members}
    counts = {"dropped": 0, "asked": 0, "candidates": 0, "applied": 0}
    for name, entries in new_facts.items():
        member = by_name.get(name)
        if member is None:
            continue
        others = [m for m in members if m is not member]
        for entry in entries:
            if not str(entry.get("fact") or "").strip():
                continue
            speaker = speaker_of(entry, lookup)
            verdict, reason = decide(entry, member, others, speaker)
            if verdict == "drop":
                counts["dropped"] += 1
                queue.data["dropped"] += 1
                continue
            if queue._find("applied", name, entry["fact"]):
                continue
            if verdict == "ask" or queue._find("ask", name, entry["fact"]):
                queue.add("ask", name, entry, speaker, reason or "sensitive")
                counts["asked"] += 1
                continue
            queue.add("candidates", name, entry, speaker)
            counts["candidates"] += 1

    now = _now()
    ready, aged = [], []
    for item in list(queue.data["candidates"]):
        if len(item["quotes"]) >= 2:
            ready.append(item)
        elif now - item["first_seen"] > CANDIDATE_DAYS * 86400:
            aged.append(item)
    for item in aged:
        queue.data["candidates"].remove(item)
        queue.data["dropped"] += 1
    if ready and generate is not None:
        passed = []
        for item in ready:
            try:
                ok = grounded(generate, item["member"], item["fact"], item["quotes"])
            except Exception as exc:  # noqa: BLE001 - GPU busy or lent: the rest wait for next cycle
                logger.info("bio gate: grounding deferred (%s)", exc)
                break
            if ok:
                passed.append(item)
            else:
                queue.move(item, "candidates", "ask", "the model wasn't sure")
        applied, full = apply_items(members_path, passed, limit, "auto")
        for item in applied:
            queue.move(item, "candidates", "applied")
        for item in full:
            queue.move(item, "candidates", "ask", "his profile is full")
        counts["applied"] = len(applied)
    queue.data["ask"] = [i for i in queue.data["ask"]
                         if now - i.get("moved_at", i["first_seen"]) <= ASK_EXPIRE_DAYS * 86400
                         or i["id"] in (queue.data["asked"].get("ids") or [])]
    return counts


# ── the weekly list and his answer ───────────────────────────────────────────

def _added_since(queue: BioQueue, since: float) -> List[Dict[str, Any]]:
    return [i for i in queue.data["applied"]
            if i.get("applied") == "auto" and float(i.get("moved_at") or 0) > since]


def digest_due(queue: BioQueue, wrong: List[Dict[str, Any]]) -> bool:
    last = float(queue.data["asked"].get("sent_at") or 0)
    waiting = queue.data["ask"] or wrong or _added_since(queue, last)
    return bool(waiting) and _now() - last >= DIGEST_EVERY_DAYS * 86400


def digest_text(queue: BioQueue, wrong: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """The weekly DM, and the item ids its numbers stand for."""
    items = sorted(queue.data["ask"], key=lambda i: i["first_seen"])[:DIGEST_SIZE]
    lines = [f"{BIO_TAG} Kaya's weekly check"]
    if items:
        lines += ["", "Facts about the group waiting for your yes (they reach Kaya's profiles "
                  "only if you say so). Reply to this message with the numbers, \"all\" or "
                  "\"none\":"]
        for n, item in enumerate(items, 1):
            q = item["quotes"][0] if item["quotes"] else {"quote": "", "speaker": ""}
            lines.append(f"{n}. {item['fact']} ({item.get('reason') or 'unsure'}) "
                         f"— «{q['quote'][:120]}»{' — ' + q['speaker'] if q.get('speaker') else ''}")
        if len(queue.data["ask"]) > len(items):
            lines.append(f"(+{len(queue.data['ask']) - len(items)} more next week)")
    added = _added_since(queue, float(queue.data["asked"].get("sent_at") or 0))
    if added:
        lines += ["", "Added to profiles on their own (grounded in two quotes; "
                  "\"review_bios.py --undo <name>\" takes the last one back):"]
        for item in added[:10]:
            lines.append(f"- {item['fact']}")
        if len(added) > 10:
            lines.append(f"(+{len(added) - 10} more)")
    if wrong:
        lines += ["", "Answers the group called wrong this week:"]
        for w in wrong[:5]:
            lines.append(f"- «{(w.get('user_message') or '')[:80]}» → «{(w.get('assistant_response') or '')[:140]}» "
                         f"({w.get('speaker') or 'someone'}: «{(w.get('comment') or '')[:60]}»)")
    return "\n".join(lines), [i["id"] for i in items]


def parse_answer(text: str, count: int) -> Optional[List[int]]:
    """1-based numbers he accepted; [] for none; None when it isn't an answer."""
    low = (text or "").strip().lower()
    if low in ("all", "todos", "todas", "tudo", "sim", "yes"):
        return list(range(1, count + 1))
    if low in ("none", "nenhum", "nenhuma", "nada", "no", "não", "nao"):
        return []
    nums = [int(n) for n in re.findall(r"\d+", low)]
    if not nums or re.sub(r"[\d\s,;e&and]", "", low):
        return None
    return sorted({n for n in nums if 1 <= n <= count})


def apply_answer(queue: BioQueue, members_path: Path, text: str, limit: int) -> str:
    """Apply what he accepted from the last list; the reply to send him."""
    ids = list(queue.data["asked"].get("ids") or [])
    if not ids:
        return f"{BIO_TAG} There's no open list to answer."
    picked = parse_answer(text, len(ids))
    if picked is None:
        return f"{BIO_TAG} Reply with the numbers (e.g. 1 3), \"all\" or \"none\"."
    items = {i["id"]: i for i in queue.data["ask"]}
    chosen = [items[ids[n - 1]] for n in picked if ids[n - 1] in items]
    applied, full = apply_items(members_path, chosen, limit, "his yes")
    for item in applied:
        queue.move(item, "ask", "applied")
    # Everything else on that list was seen and not chosen: it's dropped.
    dropped = [items[i] for i in ids if i in items and items[i] not in chosen]
    queue.data["ask"] = [i for i in queue.data["ask"] if i not in dropped]
    queue.data["dropped"] += len(dropped)
    queue.data["asked"] = {"sent_at": queue.data["asked"].get("sent_at"), "ids": []}
    queue.save()
    parts = [f"{BIO_TAG} Applied {len(applied)} fact(s)"]
    if dropped:
        parts.append(f"dropped {len(dropped)}")
    if full:
        parts.append(f"{len(full)} not applied: those profiles are full ({', '.join(sorted({i['member'] for i in full}))})")
    return ", ".join(parts) + "."


def undo(queue: BioQueue, members_path: Path, member: str) -> Optional[str]:
    """Restore the snapshot taken before the last applied fact about ``member``."""
    items = [i for i in queue.data["applied"] if i["member"] == member and i.get("snapshot")]
    if not items:
        return None
    last = max(items, key=lambda i: i.get("moved_at", 0))
    snap = Path(last["snapshot"])
    if not snap.exists():
        return None
    shutil.copy2(snap, members_path)
    queue.data["applied"] = [i for i in queue.data["applied"]
                             if i.get("snapshot") != last["snapshot"]]
    queue.save()
    return str(snap)


def status(queue: BioQueue) -> Dict[str, Any]:
    """For labwatch: how much waits for him, and for how long."""
    ask = queue.data["ask"]
    oldest = min((i.get("moved_at", i["first_seen"]) for i in ask), default=None)
    return {"waiting": len(ask), "candidates": len(queue.data["candidates"]),
            "applied": len(queue.data["applied"]), "dropped": queue.data["dropped"],
            "oldest_waiting_days": round((_now() - oldest) / 86400, 1) if oldest else None,
            "digest_sent_at": queue.data["asked"].get("sent_at")}


def recent_called_wrong(feedback_path: Path, days: int = DIGEST_EVERY_DAYS) -> List[Dict[str, Any]]:
    """The answers the group called wrong in the last ``days`` (feedback log, newest first)."""
    from datetime import datetime, timezone

    cutoff = _now() - days * 86400
    out: List[Dict[str, Any]] = []
    try:
        lines = Path(feedback_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") != "called_wrong":
            continue
        try:
            when = datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc).timestamp()
        except (KeyError, ValueError):
            continue
        if when >= cutoff:
            out.append(row)
    return out[::-1]
