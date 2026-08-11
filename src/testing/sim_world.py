"""The invented people the simulator plays, and the chats they talk in.

Deliberately synthetic. `src/testing/agent_sim.py` builds its personas from the
real `group_members.json` and sends those facts to xAI; this does not, because
the privacy invariant says group data stays on the box and a test harness is a
poor reason to make an exception. Nothing here corresponds to a real member, a
real number or a real chat.

What that costs: the personas will not sound exactly like the group. What it buys
is that a simulator run can be driven by any cloud model without leaking anything,
and that the transcript is safe to paste into a bug report.

Memory and recall are still exercised — the bot answers out of the real vector
store (copied into `data_sim/`, never leaving the box), and the scenarios plant
their own facts mid-conversation and probe for them later.

Both `scripts/seed_sim_data.py` and the harness import from here, so the
whitelist, contacts and scope files the sim instance reads always agree with the
JIDs the harness actually posts from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# A group id and numbers that cannot collide with anything real: the 34 prefix is
# not a Portuguese mobile range, and the group id is outside WhatsApp's format.
SIM_GROUP = "999000000000000001@g.us"
BOT_JID = "351900000000@c.us"


@dataclass(frozen=True)
class Persona:
    """One invented person the LLM plays."""

    name: str
    phone: str
    trait: str          # how they talk — drives the persona system prompt
    interest: str       # what they bring up, so improv has somewhere to go

    @property
    def jid(self) -> str:
        return f"{self.phone}@c.us"


PERSONAS: List[Persona] = [
    Persona("Tó Zé", "349000000001",
            "short messages, lots of laughing, never uses punctuation",
            "futebol e cerveja"),
    Persona("Manel", "349000000002",
            "asks a lot of questions, curious, sometimes types in English",
            "viagens e restaurantes"),
    Persona("Chico", "349000000003",
            "sarcastic, teases the others, sends memes",
            "música e concertos"),
    Persona("Bruno", "349000000004",
            "practical, organises plans, writes longer messages",
            "jantares e fins-de-semana"),
    Persona("Nuno", "349000000005",
            "arrives late to conversations and asks what he missed",
            "trabalho e carros"),
]


def personas(count: int) -> List[Persona]:
    return PERSONAS[: max(1, min(count, len(PERSONAS)))]


def contacts_map() -> Dict[str, str]:
    """phone -> display name, the shape data/whatsapp_contacts.json expects."""
    return {p.phone: p.name for p in PERSONAS}


def whitelist() -> List[str]:
    """Every sim number, so DMs are not dropped by the anti-spam gate.

    Without this the sim inherits the real whitelist and every synthetic DM is
    silently ignored — the webhook answers `handled: false` and nothing explains
    why.
    """
    return [p.phone for p in PERSONAS]


def shared_chats() -> List[str]:
    """The sim group is shared memory; DMs are private.

    This is what makes the scope-leak scenario meaningful: a secret told in a DM
    must not surface in a chat whose scope is `shared`.
    """
    return [SIM_GROUP]
