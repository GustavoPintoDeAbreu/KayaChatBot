"""Behavior-targeted question bank for on-prem synthetic data generation.

Produces the QUESTIONS half of the fine-tune; the answers are generated locally
by a teacher model (see generate_local_synthetic.py). The templates deliberately
target the behaviors the group flagged and the old quote-only data missed:

  * per-member detail   — "Quem é o X?", "Qual é a profissão do X?"
  * group-wide superlatives — "Quem é o mais convencido do grupo?"
  * opinions/assessments — "O que achas de X?"
  * general group dynamics — "Como é que o grupo se conheceu?"

Pure functions: no model/network, fully unit-testable.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# Group-wide superlative traits (PT + EN). Each becomes "Quem é o mais {t}…".
_TRAITS_PT = [
    "convencido", "engraçado", "inteligente", "trabalhador", "preguiçoso",
    "dramático", "calmo", "teimoso", "generoso", "competitivo", "sarcástico",
    "stressado", "novo", "velho", "alto", "baixo", "barulhento", "reservado",
]
_TRAITS_EN = [
    "arrogant", "funny", "smart", "hard-working", "lazy", "dramatic",
    "chill", "stubborn", "generous", "competitive", "sarcastic",
]

# Per-member question templates ({n} = a member name or alias).
_MEMBER_PT = [
    "Quem é o {n}?",
    "O que sabes sobre o {n}?",
    "O que é que o {n} gosta de fazer?",
    "Qual é a profissão do {n}?",
    "Conta-me sobre o {n}.",
    "O que achas do {n}?",
    "Como descreverias o {n}?",
]
_MEMBER_EN = [
    "Who is {n}?",
    "What do you know about {n}?",
    "What does {n} like?",
    "What's {n}'s job?",
    "Tell me about {n}.",
]

_GROUPWIDE_PT = [
    "Quem é o mais {t} do grupo?",
    "Na tua opinião, quem é o mais {t} do grupo?",
]
_GROUPWIDE_EN = ["Who is the most {t} in the group?"]

_OPINION_PT = [
    "O que achas de {topic}?",
    "Qual é a tua opinião sobre {topic}?",
]
_OPINION_TOPICS = [
    "política", "futebol", "a forma como o grupo conversa",
    "os planos de viagem do grupo", "tecnologia", "música",
    "a dinâmica do grupo", "eventos mundiais",
]

_GENERAL = [
    "Quem são os membros do grupo?",
    "Como é que o grupo se conheceu?",
    "Há quanto tempo é que se conhecem?",
    "Descreve a dinâmica do grupo.",
    "Qual é a vibe do grupo?",
    "Who are the members of the group?",
    "How did the group meet?",
    "Describe the group's dynamic.",
]


def _member_display_names(members_data: Dict, aliases_per_member: int = 2) -> List[str]:
    """Return member names plus a couple of distinctive aliases each."""
    names: List[str] = []
    for member in members_data.get("members", []):
        name = member.get("name")
        if not name:
            continue
        names.append(name)
        extra = [a for a in member.get("aliases", []) if a.lower() != name.lower()]
        names.extend(extra[:aliases_per_member])
    return names


def build_questions(
    members_data: Dict,
    seed: int = 3407,
    per_category: Optional[int] = None,
) -> List[str]:
    """Build the deduplicated, shuffled question bank.

    ``per_category`` caps how many questions are kept from each category (after
    shuffling) — useful to balance the mix or shrink for a quick run. None keeps
    every expansion.
    """
    rng = random.Random(seed)
    names = _member_display_names(members_data)

    categories: List[List[str]] = []

    # Per-member
    member_qs = [tpl.format(n=n) for n in names for tpl in _MEMBER_PT]
    member_qs += [tpl.format(n=n) for n in names for tpl in _MEMBER_EN]
    categories.append(member_qs)

    # Group-wide superlatives
    groupwide = [tpl.format(t=t) for t in _TRAITS_PT for tpl in _GROUPWIDE_PT]
    groupwide += [tpl.format(t=t) for t in _TRAITS_EN for tpl in _GROUPWIDE_EN]
    categories.append(groupwide)

    # Opinions
    opinion = [tpl.format(topic=topic) for topic in _OPINION_TOPICS for tpl in _OPINION_PT]
    categories.append(opinion)

    # General
    categories.append(list(_GENERAL))

    questions: List[str] = []
    for cat in categories:
        rng.shuffle(cat)
        questions.extend(cat[:per_category] if per_category else cat)

    # Deduplicate (preserve order) and final shuffle for training variety.
    seen = set()
    deduped = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    rng.shuffle(deduped)
    return deduped


def load_members(path: str) -> Dict:
    """Load group_members.json (single source of member names/aliases)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
