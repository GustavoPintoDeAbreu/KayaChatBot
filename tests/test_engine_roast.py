"""Roasts must spread out, and "think hard" must actually cost a think.

Measured over the first full group session: one member took 29.4% of every
mention the bot made, against an 8% even split, and 37.5% of turns named
somebody nobody had asked about. The group read that as the bot having it in for
him. It was the prompt shape — an unaimed roast routed `factual`, got all 13
profiles, and reached for whoever had the most material.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import router
from src.chat.engine import KayaEngine

MEMBERS = ["Gil", "Rafa", "Pedro", "Gustavo", "Murgeiro"]


class ScriptedBackend:
    def __init__(self, label, answers=("uma resposta qualquer bem diferente",)):
        self.label = label
        self.answers = list(answers)
        self.answer_calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        if max_new_tokens is not None and max_new_tokens <= 16:
            return self.label
        self.answer_calls.append({"messages": messages, "max_new_tokens": max_new_tokens,
                                  "sampling": sampling})
        return self.answers.pop(0) if self.answers else "resposta"


class StubRetriever:
    """Name matching only — enough for target selection and telemetry."""

    def extract_query_persons(self, query):
        return [m.lower() for m in MEMBERS if m.lower() in (query or "").lower()]

    def named_members(self, text):
        return [m for m in MEMBERS if m.lower() in (text or "").lower()]

    def retrieve_all(self, *a, **kw):
        return "contexto do grupo"


def make_engine(backend, **chat_over):
    config = {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "fallback_mode": "factual"},
            "modes": {
                "roast": {"retrieval": True, "max_new_tokens": 200,
                          "mode_hint": "Isto é um roast."},
                "factual": {"retrieval": True, "max_new_tokens": 256},
            },
            "concurrency": {"acquire_timeout": 5},
            **chat_over,
        },
        "rag": {"enabled": True},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256,
                      "temperature": 0.8, "no_repeat_last_replies": 4},
        "web_search": {"enabled": False},
    }
    return KayaEngine(model=None, tokenizer=None, retriever=StubRetriever(),
                      config=config, backend=backend)


def user_turn(backend, index=0):
    return backend.answer_calls[index]["messages"][-1]["content"]


HISTORY_ABOUT_GIL = [
    "Pedro: quem é o mais burro?",
    "Kaya Bot: O Gil, sem qualquer dúvida, é o campeão nessa categoria.",
]


# ── target rotation ──────────────────────────────────────────────────────────
def test_an_unaimed_roast_is_steered_off_the_last_target():
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("quem é o mais engraçado?", "Gil",
                                 HISTORY_ABOUT_GIL, "sys")

    assert "Não escolhas outra vez Gil" in user_turn(backend)


def test_an_aimed_roast_is_left_alone():
    """"roast the Gil" must roast the Gil, however recently he was hit."""
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("diz mal do Gil", "Pedro", HISTORY_ABOUT_GIL, "sys")

    assert "Não escolhas outra vez" not in user_turn(backend)


def test_every_recent_target_is_excluded():
    backend = ScriptedBackend("ROAST")
    history = ["Kaya Bot: O Gil é o pior.", "Kaya Bot: E o Rafa levou uma tareia."]

    make_engine(backend).respond("quem é o mais burro?", "Pedro", history, "sys")

    hint = user_turn(backend)
    assert "Gil" in hint and "Rafa" in hint


def test_a_fresh_conversation_needs_no_steering():
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("quem é o mais burro?", "Pedro", [], "sys")

    assert "Não escolhas outra vez" not in user_turn(backend)


def test_only_roasts_are_steered():
    """A factual question about a member is not a roast at him."""
    backend = ScriptedBackend("FACTUAL")

    make_engine(backend).respond("o que faz o Gil?", "Pedro", HISTORY_ABOUT_GIL, "sys")

    assert "Não escolhas outra vez" not in user_turn(backend)


def test_the_mode_hint_reaches_the_prompt():
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("quem é o mais burro?", "Pedro", [], "sys")

    assert "Isto é um roast." in user_turn(backend)


def test_the_route_is_reported_as_roast():
    backend = ScriptedBackend("ROAST")

    reply = make_engine(backend).respond("quem é o mais burro?", "Pedro", [], "sys")

    assert reply.route.mode == router.ROAST
    assert reply.telemetry["route_mode"] == router.ROAST


# ── reasoning ────────────────────────────────────────────────────────────────
def test_an_explicit_request_buys_a_planning_pass():
    """"try one last time. Really hard and detailed" got one throwaway line."""
    backend = ScriptedBackend("ROAST", answers=["- o Gil corre\n- e perde sempre",
                                                "A resposta final bem pensada."])

    reply = make_engine(backend).respond(
        "justifica bem porque é que o Gil é o alvo", "Pedro", [], "sys")

    assert len(backend.answer_calls) == 2
    assert reply.text == "A resposta final bem pensada."


def test_the_notes_are_fed_back_and_never_sent():
    backend = ScriptedBackend("ROAST", answers=["- nota interna", "A resposta."])

    reply = make_engine(backend).respond(
        "pensa bem, quem é o mais burro?", "Pedro", [], "sys")

    assert "nota interna" in user_turn(backend, 1)
    assert "nota interna" not in reply.text


def test_the_plan_is_written_at_a_lower_temperature():
    backend = ScriptedBackend("ROAST", answers=["- nota", "A resposta."])

    make_engine(backend).respond("pensa bem nisso", "Pedro", [], "sys")

    plan, answer = backend.answer_calls
    assert plan["sampling"]["temperature"] < answer["sampling"]["temperature"]


def test_an_ordinary_question_costs_one_generation():
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("quem é o mais burro?", "Pedro", [], "sys")

    assert len(backend.answer_calls) == 1


def test_reasoning_can_be_switched_off():
    backend = ScriptedBackend("ROAST")

    engine = make_engine(backend, reasoning={"enabled": False})
    engine.respond("pensa bem, quem é o mais burro?", "Pedro", [], "sys")

    assert len(backend.answer_calls) == 1


# ── who is writing (bug: "rafa is peaking to you and you think its peter") ───
def test_the_turn_says_who_is_writing():
    """Rafa asked "why he roasting ME in my iq guess" and the bot answered about
    another member entirely — it kept working through the member list instead of
    resolving "me" to the person who had just written. Every history line in a
    group looks like "Nome: texto", so a final line in the same shape is a weak
    signal; this says it outright."""
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("why he roasting me in my iq guess", "Rafa", [], "sys")

    turn = user_turn(backend)
    assert "Quem está a escrever agora é o Rafa" in turn
    assert '"eu", "me", "mim" e "meu"' in turn


def test_the_generic_caller_gets_no_speaker_line():
    """The web UI and benchmarks pass "User", where naming a speaker is noise."""
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("quem é o mais burro?", "User", [], "sys")

    assert "Quem está a escrever agora" not in user_turn(backend)


# ── the maintainer clause ────────────────────────────────────────────────────
# The bot cannot change its own code, so a technical complaint is answered by
# naming whoever maintains it. That person is in the group, which the clause
# never allowed for: Gustavo asked why image generation was so bad and was told
# "O Gustavo tem de tratar disso", then had to reply "Yah eu sou o Gustavo".
MAINT = {
    "maintainer": "Gustavo",
    "maintainer_clause": "diz que o {maintainer} tem de tratar disso",
    "maintainer_self_clause": "quem está a escrever é o {maintainer}, não lho digas",
}


def system_prompt(backend, index=0):
    return backend.answer_calls[index]["messages"][0]["content"]


def test_the_clause_names_the_maintainer_to_everybody_else():
    backend = ScriptedBackend("BANTER", answers=["ya"])
    engine = make_engine(backend, modes={
        "banter": {"retrieval": False, "system_prompt": "És o bot. {maintainer_clause}"},
    }, **MAINT)

    engine.respond("isto está lento", "Gil", [], "sys")

    assert "diz que o Gustavo tem de tratar disso" in system_prompt(backend)


def test_the_maintainer_is_not_told_to_go_tell_himself():
    backend = ScriptedBackend("BANTER", answers=["ya"])
    engine = make_engine(backend, modes={
        "banter": {"retrieval": False, "system_prompt": "És o bot. {maintainer_clause}"},
    }, **MAINT)

    engine.respond("Stfu horrível a gerar imagens", "Gustavo", [], "sys")

    prompt = system_prompt(backend)
    assert "tem de tratar disso" not in prompt
    assert "quem está a escrever é o Gustavo" in prompt


def test_the_match_is_case_insensitive():
    backend = ScriptedBackend("BANTER", answers=["ya"])
    engine = make_engine(backend, modes={
        "banter": {"retrieval": False, "system_prompt": "{maintainer_clause}"},
    }, **MAINT)

    engine.respond("bug", "gustavo", [], "sys")

    assert "quem está a escrever é o Gustavo" in system_prompt(backend)


def test_the_placeholder_never_survives_into_the_prompt():
    """A leaked "{maintainer_clause}" would be read aloud by the model."""
    backend = ScriptedBackend("BANTER", answers=["ya"])
    engine = make_engine(backend, modes={
        "banter": {"retrieval": False, "system_prompt": "És o bot. {maintainer_clause}"},
    }, **MAINT)

    for speaker in ("Gil", "Gustavo"):
        engine.respond("isto está lento", speaker, [], "sys")
    assert all("{maintainer_clause}" not in system_prompt(backend, i)
               for i in range(len(backend.answer_calls)))


def test_no_maintainer_configured_leaves_the_prompt_alone():
    backend = ScriptedBackend("BANTER", answers=["ya"])
    engine = make_engine(backend, modes={
        "banter": {"retrieval": False, "system_prompt": "És o bot."},
    })

    engine.respond("isto está lento", "Gil", [], "sys")

    assert system_prompt(backend).startswith("És o bot.")


# ── variety ──────────────────────────────────────────────────────────────────
# Peter asked to be roasted four times over three days and got Rotterdam,
# editing other people's videos and Five Guys every time. Two causes: the
# WhatsApp prompt is built once at import, and key_facts[:max_facts] handed over
# the same first four facts in the same order on every turn.
def test_an_open_ended_turn_redraws_the_member_facts():
    backend = ScriptedBackend("ROAST")
    engine = make_engine(backend)
    calls = []

    def factory(sample_facts=False):
        calls.append(sample_facts)
        return "prompt com factos novos"

    engine.system_prompt_factory = factory
    engine.respond("roast me", "Peter", [], "sys")

    assert calls == [True]
    assert backend.answer_calls[0]["messages"][0]["content"] == "prompt com factos novos"


def test_a_factual_turn_keeps_the_prompt_it_was_given():
    """"o que faz o Gil?" must not depend on whether his job survived a draw."""
    backend = ScriptedBackend("FACTUAL")
    engine = make_engine(backend)
    calls = []
    engine.system_prompt_factory = lambda sample_facts=False: calls.append(sample_facts) or "redrawn"

    engine.respond("o que faz o Gil?", "Pedro", [], "o prompt fixo")

    assert calls == []
    assert backend.answer_calls[0]["messages"][0]["content"] == "o prompt fixo"


def test_a_failing_factory_falls_back_to_the_fixed_prompt():
    """A variety nicety must never cost a reply."""
    backend = ScriptedBackend("ROAST")
    engine = make_engine(backend)

    def boom(sample_facts=False):
        raise RuntimeError("members file gone")

    engine.system_prompt_factory = boom
    engine.respond("roast me", "Peter", [], "o prompt fixo")

    assert backend.answer_calls[0]["messages"][0]["content"] == "o prompt fixo"


def _with_history(monkeypatch, rows):
    from src.chat import metrics

    monkeypatch.setattr(metrics, "load_interactions", lambda path=None, limit=0: rows)


def _said_about(name, reply):
    return {"route_mode": "roast", "route_command": "",
            "assistant_response": reply, "reply_members": [name]}


# Gil rather than Peter only because MEMBERS above is the stub retriever's roster;
# the case being reproduced is Peter's four identical roasts.
GIL_ROAST = ("O Gil vive entre Roterdão e Queijas só para editar vídeos de "
             "outros e comer Five Guys sem parar nunca.")


def test_roast_me_is_told_what_it_already_said_about_the_asker(monkeypatch):
    """The subject of "roast me" is the speaker — the exact case that repeated."""
    _with_history(monkeypatch, [_said_about("Gil", GIL_ROAST)])
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("roast me", "Gil", [], "sys")

    turn = user_turn(backend)
    assert "Não repitas o material" in turn and "Roterdão" in turn


def test_a_named_target_is_looked_up_too(monkeypatch):
    _with_history(monkeypatch, [_said_about("Gil", GIL_ROAST)])
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("diz mal do Gil", "Pedro", [], "sys")

    assert "Roterdão" in user_turn(backend)


def test_a_factual_turn_gets_no_variety_hint(monkeypatch):
    _with_history(monkeypatch, [_said_about("Gil", GIL_ROAST)])
    backend = ScriptedBackend("FACTUAL")

    make_engine(backend).respond("o que faz o Gil?", "Pedro", [], "sys")

    assert "Não repitas o material" not in user_turn(backend)


def test_nothing_said_about_them_yet_adds_nothing(monkeypatch):
    _with_history(monkeypatch, [])
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("roast me", "Gil", [], "sys")

    assert "Não repitas o material" not in user_turn(backend)


def test_a_non_member_speaker_is_not_looked_up(monkeypatch):
    """"Alguém" is the fallback for an unresolved sender, not a person."""
    _with_history(monkeypatch, [_said_about("Gil", GIL_ROAST)])
    backend = ScriptedBackend("ROAST")

    make_engine(backend).respond("roast me", "Alguém", [], "sys")

    assert "Não repitas o material" not in user_turn(backend)
