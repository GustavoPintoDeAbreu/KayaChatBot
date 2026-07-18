"""Provenance-backed per-member fact store + conflict/staleness audit.

The current member knowledge is two flat, provenance-free layers: auto-generated
scalars/lists (last-writer-wins / append-only, no contradiction handling) and a
hand-curated ``key_facts`` list. Neither records *why* a fact is believed, so
attribution errors and stale facts are invisible until a wrong answer surfaces
them (e.g. a dog attributed to the wrong member; a job the member already quit).

This module adds the missing structure without pretending to auto-resolve
genuinely ambiguous group-chat truth:

  * ``MemberEvidenceIndex`` attributes every message to a canonical member (via
    the same ``SenderResolver`` extraction uses) and lets you pull the messages
    that mention a member together with a fact's salient terms.
  * ``audit_fact`` scores a claimed fact against that evidence — how many
    distinct members are associated with the fact's key term (attribution
    ambiguity), how recent the evidence is (staleness), and returns sample
    source-message ids as provenance.
  * ``audit_member_key_facts`` runs this over a member's curated ``key_facts`` so
    low-evidence / cross-attributed / stale claims can be reviewed with receipts.

It is deterministic (no teacher) and read-only over the logs, so it is safe to
run any time to keep the curated facts honest.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.data.identity_resolver import SenderResolver

# Reuse the retriever's content-token rules so the audit keys on the same signal
# (proper nouns, nicknames) the RAG lexical channel does.
from src.chat.retriever import _lexical_tokens


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


@dataclass
class FactEvidence:
    """Evidence supporting (or contradicting) a claimed fact about a member."""

    term: str
    support_count: int                       # messages by/mentioning the member + term
    associated_members: Dict[str, int]       # member -> message count for the term (all authors)
    last_seen: Optional[str]                  # ISO date of most recent supporting message
    sample_msg_ids: List[str] = field(default_factory=list)

    @property
    def attribution_ambiguous(self) -> bool:
        """True when another member is more strongly tied to the term than ours."""
        if not self.associated_members:
            return False
        top = max(self.associated_members.values())
        mine = self.support_count
        return mine < top  # someone else owns the term more than the claimed member


class MemberEvidenceIndex:
    """Attributes messages to canonical members and indexes them for lookup."""

    def __init__(self, members_file: Path, sender_aliases: Optional[Dict[str, str]] = None):
        self.resolver = SenderResolver(members_file, sender_aliases or {})
        # member -> list of (msg_id, iso_ts, normalised_text)
        self._by_member: Dict[str, List] = {}
        self._alias_to_member = self._build_alias_map(members_file)

    @staticmethod
    def _build_alias_map(members_file: Path) -> Dict[str, str]:
        import json
        data = json.loads(Path(members_file).read_text(encoding="utf-8"))
        members = data if isinstance(data, list) else data.get("members", [])
        amap: Dict[str, str] = {}
        for m in members:
            amap[m["name"].lower()] = m["name"]
            for alias in m.get("aliases", []):
                amap[alias.lower()] = m["name"]
        return amap

    def index(self, messages: List[Dict]) -> "MemberEvidenceIndex":
        """Attribute each message to its author (canonical) and any mentioned member."""
        for i, msg in enumerate(messages):
            text = msg.get("text", "")
            norm = _strip_accents(text)
            msg_id = msg.get("id") or f"msg_{i}"
            ts = msg.get("timestamp", "")
            targets = set()
            author = self.resolver.resolve(msg.get("sender", ""))
            if author and author in set(self._alias_to_member.values()):
                targets.add(author)
            # Members explicitly named in the text (whole-word alias match).
            for alias, name in self._alias_to_member.items():
                if re.search(rf"\b{re.escape(alias)}\b", norm):
                    targets.add(name)
            for name in targets:
                self._by_member.setdefault(name, []).append((msg_id, ts, norm))
        return self

    def audit_fact(self, member: str, terms: List[str]) -> Optional[FactEvidence]:
        """Score how well ``terms`` are supported for ``member`` vs other members."""
        terms = [t for t in terms if t]
        if not terms:
            return None
        # Support = messages tied to this member that contain any of the terms.
        member_msgs = self._by_member.get(member, [])
        support = [(mid, ts) for (mid, ts, norm) in member_msgs
                   if any(_strip_accents(t) in norm for t in terms)]
        # Cross-member association: which members are tied to these terms at all.
        assoc: Dict[str, int] = {}
        for name, rows in self._by_member.items():
            cnt = sum(1 for (_mid, _ts, norm) in rows
                      if any(_strip_accents(t) in norm for t in terms))
            if cnt:
                assoc[name] = cnt
        last_seen = max((ts for _mid, ts in support), default=None)
        return FactEvidence(
            term=", ".join(terms),
            support_count=len(support),
            associated_members=assoc,
            last_seen=last_seen,
            sample_msg_ids=[mid for mid, _ts in support[:5]],
        )


# Words that never discriminate a fact — dropped when picking a fact's key terms.
_FACT_STOPWORDS = {
    "owns", "dog", "named", "also", "works", "with", "high", "recently", "moved",
    "role", "enjoys", "music", "likes", "known", "group", "started", "focused",
    "health", "and", "the", "his", "her", "who", "that", "they", "them", "from",
    "member", "group's", "supports", "plays", "often", "offers", "been", "active",
    "again", "lives", "trains", "nights",
}


def salient_terms(fact_text: str, member_aliases: List[str], limit: int = 4) -> List[str]:
    """Pick the discriminating tokens of a fact (proper nouns / rare terms).

    Drops the member's own name/aliases and generic vocabulary, keeps the tokens
    that actually identify the claim (a pet name, a company, a place).
    """
    aliases = {_strip_accents(a) for a in member_aliases}
    seen = []
    for tok in _lexical_tokens(fact_text):
        if tok in aliases or tok in _FACT_STOPWORDS or len(tok) < 4:
            continue
        if tok not in seen:
            seen.append(tok)
    # Prefer capitalised proper nouns from the original text when present.
    proper = [w for w in re.findall(r"\b([A-Z][a-zA-Zà-ú]{3,})\b", fact_text)
              if _strip_accents(w) not in aliases]
    ordered = [p.lower() for p in proper if p.lower() in seen]
    ordered += [t for t in seen if t not in ordered]
    return ordered[:limit]


def audit_member_key_facts(member: Dict, index: MemberEvidenceIndex) -> List[Dict]:
    """Audit a member's curated key_facts against message evidence."""
    aliases = member.get("aliases", []) + [member["name"]]
    out = []
    for fact in member.get("key_facts", []):
        terms = salient_terms(fact, aliases)
        ev = index.audit_fact(member["name"], terms)
        if ev is None:
            out.append({"fact": fact, "terms": [], "verdict": "no-salient-terms"})
            continue
        if ev.support_count == 0:
            verdict = "UNSUPPORTED"
        elif ev.attribution_ambiguous:
            verdict = "CROSS-ATTRIBUTED"
        else:
            verdict = "ok"
        out.append({
            "fact": fact,
            "terms": terms,
            "verdict": verdict,
            "support_count": ev.support_count,
            "associated_members": ev.associated_members,
            "last_seen": ev.last_seen,
            "sample_msg_ids": ev.sample_msg_ids,
        })
    return out
