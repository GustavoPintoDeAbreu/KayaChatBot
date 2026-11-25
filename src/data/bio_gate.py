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

Since 2026-11-25 every fact written here is dated (``fact_meta``), and the weekly list
also asks whether up to three old job/home/relationship facts are still true.
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

from src.data import fact_meta

logger = logging.getLogger(__name__)

BIO_TAG = "👤"
DEFAULT_QUEUE = "data/bio_queue.json"
SAME_FACT = 0.72             # SequenceMatcher ratio for "the same fact, reworded"
CANDIDATE_DAYS = 28          # a fact seen once waits this long for a second quote, then is dropped
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


_STOP = {"a", "an", "the", "to", "at", "in", "on", "of", "and", "is", "was", "has", "had", "used", "named", "called",
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


def apply_items(members_path: Path, items: List[Dict[str, Any]], limit: Optional[int],
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
        existing = next((f for f in facts if same_fact(f, item["fact"])), None)
        if existing is None:
            if limit is not None and len(facts) >= limit:
                full.append(item)
                continue
            facts.append(item["fact"])
            member["key_facts"] = facts
            fact_meta.stamp(member, item["fact"], "gate" if how == "auto" else how)
            changed = True
        elif how != "auto":
            # His yes to a fact already there confirms it: it is no longer old.
            fact_meta.stamp(member, existing, how)
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
    return counts


# ── the weekly list and his answer ───────────────────────────────────────────
#
# The list numbers the facts waiting for his yes first, then the facts added on
# their own since his last answer (so he can drop or reword those too). Nothing he
# hasn't answered disappears: a skipped week comes back next week, flagged, until he
# answers (Gustavo, 2026-11-19: "if I skip a week, tell me what I forgot").
# He answers by quoting the list, in text or in a voice note, in plain words:
# "everything's right", or "number two is wrong, drop it; number three: he works
# at X". ``interpret_answer`` turns that into actions with the local model; the
# bare forms ("1 3", "all", "none") need no model.

def _since_answer(queue: BioQueue) -> float:
    asked = queue.data["asked"]
    return float(asked.get("answered_at") or asked.get("first_sent_at") or 0)


def _added_since(queue: BioQueue, since: float) -> List[Dict[str, Any]]:
    return [i for i in queue.data["applied"]
            if i.get("applied") == "auto" and float(i.get("moved_at") or 0) > since]


def digest_due(queue: BioQueue, wrong: List[Dict[str, Any]],
               stale: Optional[List[Dict[str, Any]]] = None) -> bool:
    last = float(queue.data["asked"].get("sent_at") or 0)
    waiting = queue.data["ask"] or wrong or _added_since(queue, _since_answer(queue)) or stale
    return bool(waiting) and _now() - last >= DIGEST_EVERY_DAYS * 86400


def digest_text(queue: BioQueue, wrong: List[Dict[str, Any]],
                stale: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[str], List[str]]:
    """The weekly DM, the waiting ids and the added ids its numbers stand for (in order).

    ``stale`` (``fact_meta.stale_facts``) are numbered last, under "Still true?";
    ``mark_sent(..., recheck=stale)`` remembers them."""
    asked = queue.data["asked"]
    unanswered = set(asked.get("ids") or []) if not asked.get("answered_at") or \
        float(asked.get("answered_at") or 0) < float(asked.get("sent_at") or 0) else set()
    waiting = sorted(queue.data["ask"], key=lambda i: (i["id"] not in unanswered, i["first_seen"]))
    items = waiting[:DIGEST_SIZE]
    added = _added_since(queue, _since_answer(queue))[:DIGEST_SIZE]
    lines = [f"{BIO_TAG} Kaya's weekly check"]
    if unanswered:
        weeks = max(int(i.get("asked_times") or 1) for i in queue.data["ask"] if i["id"] in unanswered) \
            if any(i["id"] in unanswered for i in queue.data["ask"]) else 1
        lines.append(f"(You didn't answer last week's list{'s' if weeks > 1 else ''}; "
                     f"those are first, marked ↩.)")
    n = 0
    if items:
        lines += ["", "Waiting for your yes (they reach Kaya's profiles only if you say so):"]
        for item in items:
            n += 1
            q = item["quotes"][0] if item["quotes"] else {"quote": "", "speaker": ""}
            mark = "↩ " if item["id"] in unanswered else ""
            lines.append(f"{n}. {mark}{item['fact']} ({item.get('reason') or 'unsure'}) "
                         f"— «{q['quote'][:120]}»{' — ' + q['speaker'] if q.get('speaker') else ''}")
        if len(waiting) > len(items):
            lines.append(f"(+{len(waiting) - len(items)} more next week)")
    if added:
        lines += ["", "Added on their own since your last answer (say if one is wrong):"]
        for item in added:
            n += 1
            lines.append(f"{n}. {item['fact']}")
    if stale:
        lines += ["", "Still true? (old facts about jobs, homes or relationships; say keep, drop "
                  "or the new version):"]
        for item in stale:
            n += 1
            when = (time.strftime("%b %Y", time.localtime(item["learned_at"]))
                    if item.get("learned_at") else "no date")
            lines.append(f"{n}. {item['fact']} ({when})")
    if wrong:
        lines += ["", "Answers the group called wrong:"]
        for w in wrong[:5]:
            lines.append(f"- «{(w.get('user_message') or '')[:80]}» → «{(w.get('assistant_response') or '')[:140]}» "
                         f"({w.get('speaker') or 'someone'}: «{(w.get('comment') or '')[:60]}»)")
    if n:
        lines += ["", "Reply to this message, by text or voice, in your own words: \"all good\", "
                  "\"1 and 3 yes, drop the rest\", \"number 2: he works at X\"…"]
    return "\n".join(lines), [i["id"] for i in items], [i["id"] for i in added]


def mark_sent(queue: BioQueue, ids: List[str], added_ids: List[str],
              recheck: Optional[List[Dict[str, Any]]] = None) -> None:
    """Remember what the list's numbers mean and how often each fact was asked."""
    previous = queue.data["asked"]
    for item in queue.data["ask"]:
        if item["id"] in ids:
            item["asked_times"] = int(item.get("asked_times") or 0) + 1
    queue.data["asked"] = {"sent_at": _now(), "ids": ids, "added_ids": added_ids,
                           "recheck": [{"id": i["id"], "member": i["member"], "fact": i["fact"]}
                                       for i in (recheck or [])],
                           "answered_at": previous.get("answered_at"),
                           "first_sent_at": previous.get("first_sent_at") or _now()}


def parse_answer(text: str, count: int) -> Optional[List[int]]:
    """1-based numbers he accepted; [] for none; None when it isn't a bare answer."""
    low = (text or "").strip().lower().rstrip(".!")
    if low in ("all", "todos", "todas", "tudo", "sim", "yes", "ok", "all good", "tudo certo",
               "everything's right", "everything is right", "está tudo certo", "ta tudo certo"):
        return list(range(1, count + 1))
    if low in ("none", "nenhum", "nenhuma", "nada", "no", "não", "nao"):
        return []
    nums = [int(n) for n in re.findall(r"\d+", low)]
    if not nums or re.sub(r"[\d\s,;e&and]", "", low):
        return None
    return sorted({n for n in nums if 1 <= n <= count})


INTERPRET_SYSTEM = (
    "You turn Gustavo's answer to a numbered list of facts about his friends into actions. "
    "His answer may be a transcribed voice note, in Portuguese or English, informal. "
    "Reply with JSON only: {\"items\": [{\"n\": <number>, \"action\": \"keep\"|\"drop\"|\"edit\", "
    "\"fact\": \"<the corrected fact, only for edit>\"}], \"rest\": \"keep\"|\"drop\"|\"leave\"}. "
    "keep = the fact is right; drop = it is wrong or should not be kept; edit = he gave a "
    "correction: write the whole corrected fact in English, in the same style, starting with "
    "the person's name. \"rest\" is what happens to the numbers he did not mention: keep if he "
    "said everything else is right, drop if he said to drop the rest, otherwise leave. "
    "Never invent a correction he did not say.")


def interpret_answer(text: str, numbered: List[str], generate: Callable[[str, str], str]
                     ) -> Optional[Dict[str, Any]]:
    """His words as ``{"items": [...], "rest": ...}``; None if the model's answer is unusable."""
    listing = "\n".join(f"{n}. {fact}" for n, fact in enumerate(numbered, 1))
    raw = generate(INTERPRET_SYSTEM, f"The list:\n{listing}\n\nHis answer:\n{text}")
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items", []), list):
        return None
    clean = []
    for entry in parsed.get("items") or []:
        try:
            n = int(entry.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        action = str(entry.get("action") or "").lower()
        fact = str(entry.get("fact") or "").strip()
        if 1 <= n <= len(numbered) and action in ("keep", "drop", "edit") and (action != "edit" or fact):
            clean.append({"n": n, "action": action, "fact": fact})
    rest = str(parsed.get("rest") or "leave").lower()
    return {"items": clean, "rest": rest if rest in ("keep", "drop", "leave") else "leave"}


def _replace_fact(members_path: Path, member: str, old: str, new: Optional[str],
                  source: str = "his edit") -> bool:
    """Drop (``new`` None) or reword one fact in a member's ``key_facts``, with a snapshot.
    A reworded fact is dated now; ``new == old`` only re-dates it (his "still true")."""
    doc = json.loads(members_path.read_text(encoding="utf-8"))
    for person in doc.get("members", []):
        if person.get("name") != member:
            continue
        facts = list(person.get("key_facts") or [])
        hit = next((f for f in facts if f == old or same_fact(f, old)), None)
        if hit is None:
            return False
        facts = [f for f in facts if f != hit] if new is None else [new if f == hit else f for f in facts]
        person["key_facts"] = facts
        fact_meta.forget(person, hit)
        if new is not None:
            fact_meta.stamp(person, new, source)
        snapshot(members_path)
        write_members(members_path, doc)
        return True
    return False


def apply_answer(queue: BioQueue, members_path: Path, text: str, limit: int,
                 generate: Optional[Callable[[str, str], str]] = None) -> str:
    """Apply his answer to the last list; the reply to send him."""
    asked = queue.data["asked"]
    ids, added_ids = list(asked.get("ids") or []), list(asked.get("added_ids") or [])
    recheck = list(asked.get("recheck") or [])
    if not ids and not added_ids and not recheck:
        return f"{BIO_TAG} There's no open list to answer."
    waiting = {i["id"]: i for i in queue.data["ask"]}
    added = {i["id"]: i for i in queue.data["applied"]}
    numbered = [waiting[i] if i in waiting else None for i in ids] + \
               [added[i] if i in added else None for i in added_ids] + recheck
    first_recheck = len(ids) + len(added_ids) + 1

    picked = parse_answer(text, len(numbered)) if (ids or recheck) else None
    if picked is not None:
        plan = {"items": [{"n": n, "action": "keep", "fact": ""} for n in picked],
                "rest": "drop" if ids else "leave"}
        rest_added = "keep"
        # A bare "1 3" says nothing about the old facts he did not name: they are
        # asked again, never dropped.
        rest_recheck = "leave"
    else:
        if generate is None:
            return f"{BIO_TAG} I couldn't read that; reply with the numbers (e.g. 1 3), \"all\" or \"none\"."
        plan = interpret_answer(text, [i["fact"] if i else "(gone)" for i in numbered], generate)
        if plan is None:
            return f"{BIO_TAG} I couldn't make that out; nothing was changed. Try again, or numbers (e.g. 1 3)."
        rest_added = "keep" if plan["rest"] == "leave" else plan["rest"]
        rest_recheck = plan["rest"]

    said = {e["n"]: e for e in plan["items"]}
    to_apply, dropped, edited, removed, reworded, left = [], [], [], [], [], []
    confirmed = []
    for n, item in enumerate(numbered, 1):
        if item is None:
            continue
        if n >= first_recheck:
            entry = said.get(n) or {"action": rest_recheck, "fact": ""}
            if entry["action"] == "keep" and _replace_fact(
                    members_path, item["member"], item["fact"], item["fact"], "his yes"):
                confirmed.append(item)
            elif entry["action"] == "drop" and _replace_fact(
                    members_path, item["member"], item["fact"], None):
                removed.append(item)
            elif entry["action"] == "edit" and _replace_fact(
                    members_path, item["member"], item["fact"], entry["fact"]):
                reworded.append(item)
            continue
        is_waiting = n <= len(ids)
        entry = said.get(n) or {"action": plan["rest"] if is_waiting else rest_added, "fact": ""}
        action = entry["action"]
        if is_waiting:
            if action == "keep":
                to_apply.append(item)
            elif action == "edit":
                item["fact"] = entry["fact"]
                to_apply.append(item)
                edited.append(item)
            elif action == "drop":
                dropped.append(item)
            else:
                left.append(item)
        else:
            if action == "drop" and _replace_fact(members_path, item["member"], item["fact"], None):
                queue.data["applied"] = [i for i in queue.data["applied"] if i is not item]
                removed.append(item)
            elif action == "edit" and _replace_fact(members_path, item["member"], item["fact"], entry["fact"]):
                item["fact"], item["applied"] = entry["fact"], "his edit"
                reworded.append(item)
    # The cap keeps auto-added facts from piling up; it never overrides his yes
    # (2026-11-19: five facts he approved were left out because his profile was full).
    applied, full = apply_items(members_path, to_apply, None, "his yes")
    edited = [i for i in edited if i in applied]
    for item in applied:
        queue.move(item, "ask", "applied")
    queue.data["ask"] = [i for i in queue.data["ask"] if i not in dropped]
    queue.data["dropped"] += len(dropped) + len(removed)
    queue.data["asked"] = {**asked, "ids": [i["id"] for i in left], "added_ids": [],
                           "recheck": [], "answered_at": _now()}
    queue.save()

    parts = [f"{BIO_TAG} Done:"]
    if applied:
        parts.append(f"applied {len(applied)}" + (f" ({len(edited)} with your wording)" if edited else ""))
    if dropped:
        parts.append(f"dropped {len(dropped)}")
    if removed:
        parts.append(f"took {len(removed)} back off the profiles")
    if reworded:
        parts.append(f"reworded {len(reworded)}")
    if confirmed:
        parts.append(f"{len(confirmed)} old fact{'s' if len(confirmed) > 1 else ''} still true")
    if left:
        parts.append(f"{len(left)} left for next week")
    if full:
        parts.append(f"{len(full)} not applied, those profiles are full "
                     f"({', '.join(sorted({i['member'] for i in full}))})")
    if len(parts) == 1:
        parts.append("nothing changed")
    return parts[0] + " " + ", ".join(parts[1:]) + "."


def undo(queue: BioQueue, members_path: Path, member: str) -> Optional[str]:
    """Restore the snapshot taken before the last change about ``member``: a fact the
    gate or his yes applied, or a hand edit from ``review_bios.py`` (``manual``)."""
    items = [i for key in ("applied", "manual") for i in queue.data.get(key, [])
             if i["member"] == member and i.get("snapshot")]
    if not items:
        return None
    last = max(items, key=lambda i: i.get("moved_at", 0))
    snap = Path(last["snapshot"])
    if not snap.exists():
        return None
    shutil.copy2(snap, members_path)
    for key in ("applied", "manual"):
        if key in queue.data:
            queue.data[key] = [i for i in queue.data[key] if i.get("snapshot") != last["snapshot"]]
    queue.save()
    return str(snap)


def status(queue: BioQueue) -> Dict[str, Any]:
    """For labwatch: how much waits for him, and for how long."""
    ask = queue.data["ask"]
    oldest = min((i.get("moved_at", i["first_seen"]) for i in ask), default=None)
    return {"waiting": len(ask), "candidates": len(queue.data["candidates"]),
            "applied": len(queue.data["applied"]), "dropped": queue.data["dropped"],
            "oldest_waiting_days": round((_now() - oldest) / 86400, 1) if oldest else None,
            "digest_sent_at": queue.data["asked"].get("sent_at"),
            "recheck_open": len(queue.data["asked"].get("recheck") or [])}


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
