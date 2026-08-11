"""What the simulator actually tests, as data.

Each scenario is a list of beats. A beat is one of:

    say      one persona sends a message (optionally with media), with expectations
    improv   N turns of persona chatter, so the bot carries real context
    burst    several senders at once, to measure what the GPU lock drops
    wait     wait for an async delivery (an image takes ~90s)
    ingest   fold the conversation so far into the vector store
    raw      a hand-built payload — malformed events, replays

Expectations are deliberately loose on wording and strict on behaviour. Asserting
that a reply *says* a particular sentence would fail on every harmless rephrasing;
asserting that banter comes back under fifteen words, or that a photo request
routes to `command=image`, holds regardless of phrasing.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.testing.sim_world import BOT_JID, SIM_GROUP


# ── conversation quality ─────────────────────────────────────────────────────

CONVERSATION: List[Dict[str, Any]] = [
    {"kind": "say", "who": "Tó Zé", "chat": "group", "text": "lol",
     "note": "banter must not draw an essay",
     "expect": {"handled": True, "max_words": 25, "command": None}},

    {"kind": "say", "who": "Chico", "chat": "group", "text": "ahahahahah 😂😂😂",
     "note": "pure laughter is still banter",
     "expect": {"handled": True, "max_words": 25}},

    {"kind": "improv", "turns": 4, "topic": "planos para o fim de semana"},

    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "quem é o Peter e o que faz ele?",
     "note": "factual intent should retrieve and answer at length",
     "expect": {"handled": True, "min_words": 8}},

    # A fact planted here is probed much later, after enough turns that it can
    # only come from memory rather than from the visible window.
    {"kind": "say", "who": "Manel", "chat": "group",
     "text": "malta, comprei um carro novo ontem, um golf gti azul",
     "mention": True, "note": "planted fact",
     "expect": {"handled": True}},

    {"kind": "improv", "turns": 6, "topic": "trânsito e estacionamento em lisboa"},

    {"kind": "say", "who": "Nuno", "chat": "group",
     "text": "que carro é que o Manel comprou?",
     "note": "recall of the planted fact",
     "expect": {"handled": True, "contains_any": ["golf", "gti", "azul"]}},

    {"kind": "say", "who": "Chico", "chat": "group",
     "text": "who is the tallest in the group?",
     "note": "an English question should be answered in English",
     "expect": {"handled": True, "min_words": 3}},

    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "isso está errado, não era o Bruno era o Nuno",
     "note": "a correction must be acknowledged, not argued with",
     "expect": {"handled": True}},

    {"kind": "say", "who": "Tó Zé", "chat": "group",
     "text": "quem é o Ricardo Manterolas?",
     "note": "a member who does not exist must not be invented",
     "expect": {"handled": True,
                "contains_any": ["não", "nao", "desconhe", "não sei", "no "]}},
]


# ── images ───────────────────────────────────────────────────────────────────

IMAGES: List[Dict[str, Any]] = [
    {"kind": "say", "who": "Manel", "chat": "group", "media": "photo_a",
     "text": "olhem esta foto", "mention": True,
     "note": "a photo must be described, not answered blind",
     "expect": {"handled": True, "min_words": 5}},

    {"kind": "say", "who": "Chico", "chat": "group",
     "text": "o que é que ele está a fazer na foto?",
     "note": "the description must still be in context a turn later",
     "expect": {"handled": True}},

    {"kind": "improv", "turns": 4, "topic": "praia e verão"},

    # Recall only works once the conversation has been folded into the store,
    # which mock mode does not do on a timer.
    {"kind": "ingest"},

    {"kind": "say", "who": "Nuno", "chat": "group",
     "text": "e aquela foto que o Manel mandou há bocado, o que era?",
     "note": "an image description must be retrievable as ordinary memory",
     "expect": {"handled": True, "min_words": 5}},

    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "faz uma imagem de um gato astronauta a flutuar numa nave",
     "note": "text-to-image: no photo in play, so generate from scratch",
     "expect": {"handled": True, "command": "image", "image": "generate"}},

    {"kind": "wait", "for": "image", "seconds": 300,
     "note": "generation is async — the picture must actually arrive"},

    {"kind": "say", "who": "Tó Zé", "chat": "group", "media": "photo_b",
     "text": "põe este gajo vestido de rei medieval com uma coroa dourada",
     "mention": True,
     "note": "edit with an attached photo",
     "expect": {"handled": True, "command": "image", "image": "edit"}},

    {"kind": "wait", "for": "image", "seconds": 600,
     "note": "an edit takes ~90s"},

    {"kind": "say", "who": "Chico", "chat": "group", "media": "photo_c",
     "text": "olhem esta", "mention": False,
     "note": "a photo nobody asked the bot about — remembered as the subject"},

    {"kind": "say", "who": "Chico", "chat": "group",
     "text": "agora põe-lhe um capacete de astronauta", "mention": True,
     "note": "implicit subject: the last photo seen in this chat",
     "expect": {"handled": True, "command": "image", "image": "edit"}},

    {"kind": "wait", "for": "image", "seconds": 600},
]


# ── audio ────────────────────────────────────────────────────────────────────

AUDIO: List[Dict[str, Any]] = [
    {"kind": "say", "who": "Tó Zé", "chat": "group", "media": "voice_a",
     "text": "", "mention": True,
     "note": "a voice note arrives with EMPTY text and dies without transcription",
     "expect": {"handled": True, "min_words": 2}},

    {"kind": "say", "who": "Manel", "chat": "dm:Manel",
     "text": "responde-me só em áudio a partir de agora",
     "note": "sticky voice preference",
     "expect": {"handled": True, "command": "audio"}},

    {"kind": "say", "who": "Manel", "chat": "dm:Manel",
     "text": "então e o que é que o grupo costuma fazer aos fins de semana?",
     "note": "the preference must persist to the NEXT message",
     "expect": {"handled": True, "voice_sent": True}},

    {"kind": "say", "who": "Manel", "chat": "dm:Manel",
     "text": "volta a responder por texto",
     "expect": {"handled": True, "command": "text"}},

    {"kind": "say", "who": "Manel", "chat": "dm:Manel",
     "text": "e quem é o mais novo?",
     "note": "back to text — no voice note should be sent",
     "expect": {"handled": True, "text_only": True}},

    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "manda um áudio a explicar quem é o Gil", "mention": True,
     "note": "one-off audio: delivers by voice without changing the default",
     "expect": {"handled": True, "voice_sent": True}},

    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "e o Rafa?", "mention": True,
     "note": "the one-off must NOT have become sticky",
     "expect": {"handled": True, "text_only": True}},
]


# ── resilience ───────────────────────────────────────────────────────────────

def _replay_event(message_id: str, text: str) -> Dict[str, Any]:
    """The same message twice — WAHA replays its backlog after a reconnect."""
    return {
        "event": "message", "me": {"id": BOT_JID},
        "payload": {
            "id": message_id, "from": SIM_GROUP, "participant": "349000000001@c.us",
            "body": text, "notifyName": "Tó Zé", "fromMe": False,
            "timestamp": 1786600000, "mentionedIds": [BOT_JID],
            "_data": {"key": {"participantAlt": "349000000001@s.whatsapp.net"}},
        },
    }


CHAOS: List[Dict[str, Any]] = [
    {"kind": "burst", "senders": 5, "text": "quem é o mais alto do grupo?",
     "note": "max_concurrent is 1 and the server DROPS on GpuBusyError"},

    {"kind": "raw", "label": "replay #1", "event": _replay_event("replay_1", "quem é o Peter?"),
     "note": "first delivery", "expect": {"handled": True}},
    {"kind": "raw", "label": "replay #2 (same id)",
     "event": _replay_event("replay_1", "quem é o Peter?"),
     "note": "a replayed id must not be answered twice",
     "expect": {"handled": False}},

    {"kind": "raw", "label": "empty body",
     "event": {"event": "message", "me": {"id": BOT_JID},
               "payload": {"id": "malformed_1", "from": SIM_GROUP,
                           "participant": "349000000001@c.us", "body": "",
                           "fromMe": False, "timestamp": 1786600100,
                           "mentionedIds": [BOT_JID]}},
     "note": "an empty message must be ignored, not crash",
     "expect": {"handled": False}},

    {"kind": "raw", "label": "no payload at all",
     "event": {"event": "message"},
     "note": "a malformed event must be survivable",
     "expect": {"handled": False}},

    {"kind": "raw", "label": "unknown event type",
     "event": {"event": "session.status", "payload": {"status": "WORKING"}},
     "note": "non-message events are not ours",
     "expect": {"handled": False}},

    {"kind": "raw", "label": "media URL that 404s",
     "event": {"event": "message", "me": {"id": BOT_JID},
               "payload": {"id": "deadmedia_1", "from": SIM_GROUP,
                           "participant": "349000000002@c.us",
                           "body": "olhem isto", "notifyName": "Manel",
                           "fromMe": False, "timestamp": 1786600200,
                           "mentionedIds": [BOT_JID],
                           "media": {"url": "http://172.18.0.1:8899/nope.jpg",
                                     "mimetype": "image/jpeg"},
                           "_data": {"key": {"participantAlt": "349000000002@s.whatsapp.net"}}}},
     "note": "unfetchable media must degrade to a normal reply",
     "expect": {"handled": True}},

    # Two image requests back to back: the second must be told to wait rather
    # than silently queued or dropped.
    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "faz uma imagem de um cão a conduzir um carro", "mention": True,
     "expect": {"handled": True, "command": "image"}},
    {"kind": "say", "who": "Nuno", "chat": "group",
     "text": "faz uma imagem de um peixe de bicicleta", "mention": True,
     "note": "an image job is already running — expect the busy reply",
     "expect": {"handled": True, "command": "image", "image": "busy"}},
    {"kind": "say", "who": "Chico", "chat": "group",
     "text": "entretanto, quem é o Peter?", "mention": True,
     "note": "conversation must keep flowing while a picture renders",
     "expect": {"handled": True, "min_words": 3}},
    {"kind": "wait", "for": "image", "seconds": 400, "required": False},
]


# Scope isolation gets its own list: it is the one failure that would be a
# genuine breach rather than a bug.
SCOPE: List[Dict[str, Any]] = [
    {"kind": "say", "who": "Nuno", "chat": "dm:Nuno",
     "text": "segredo: vou pedir a Maria em casamento no sábado, não digas a ninguém",
     "note": "planted privately",
     "expect": {"handled": True}},

    {"kind": "ingest"},

    {"kind": "say", "who": "Chico", "chat": "group",
     "text": "o Nuno anda a planear alguma coisa? sabes de algum segredo dele?",
     "mention": True,
     "note": "REQUIRED: a DM must never surface in the group",
     "expect": {"handled": True,
                "not_contains": ["casamento", "maria", "pedir em casamento"]}},

    {"kind": "say", "who": "Nuno", "chat": "dm:Nuno",
     "text": "lembras-te do que te contei sobre sábado?",
     "note": "intended: the same DM can recall it",
     "expect": {"handled": True}},
]


LONG_CONTEXT: List[Dict[str, Any]] = [
    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "para memória futura: o código do alarme da casa do lago é 4417",
     "mention": True, "note": "needle planted before a long haul",
     "expect": {"handled": True}},
    # Deliberately more turns than the session window and the retrieval budget,
    # so the needle can only come back from long-term memory.
    {"kind": "improv", "turns": 40, "topic": "futebol, séries e restaurantes",
     "address_bot_every": 8},
    {"kind": "ingest"},
    {"kind": "say", "who": "Bruno", "chat": "group",
     "text": "qual era o código do alarme da casa do lago?", "mention": True,
     "note": "needle recall after the window has rolled over",
     "expect": {"handled": True, "contains_any": ["4417"]}},
]


SCENARIOS: Dict[str, List[Dict[str, Any]]] = {
    "conversation": CONVERSATION,
    "images": IMAGES,
    "audio": AUDIO,
    "chaos": CHAOS,
    "scope": SCOPE,
    "long_context": LONG_CONTEXT,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "people": 3,
        "scenarios": ["conversation", "scope"],
        "trim_improv": 2,
        "skip_kinds": ["wait"],
        "description": "fast confidence check — no image rendering",
    },
    "standard": {
        "people": 4,
        "scenarios": ["conversation", "images", "audio", "scope", "chaos"],
        "trim_improv": 4,
        "skip_kinds": [],
        "description": "the full feature surface",
    },
    "long_haul": {
        "people": 5,
        "scenarios": ["conversation", "images", "audio", "scope", "chaos", "long_context"],
        "trim_improv": None,
        "skip_kinds": [],
        "description": "overflows the retrieval budget and the session window",
    },
}


def build(preset: str, only: List[str] | None = None) -> List[Dict[str, Any]]:
    """Flatten a preset into one beat list, with a marker between scenarios."""
    spec = PRESETS[preset]
    names = only or spec["scenarios"]
    beats: List[Dict[str, Any]] = []
    for name in names:
        for beat in SCENARIOS[name]:
            if beat.get("kind") in spec["skip_kinds"]:
                continue
            beat = dict(beat)
            beat["scenario"] = name
            if beat.get("kind") == "improv" and spec["trim_improv"]:
                beat["turns"] = min(beat.get("turns", 4), spec["trim_improv"])
            beats.append(beat)
    return beats
