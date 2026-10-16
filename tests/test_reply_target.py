"""Kaya answers the message it was sent, and knows "tu" is itself.

Almoço de Natal, 16 Oct: Peter replied "That's my g" under the bot's roast, and the
bot answered "Aceito o diagnóstico", which fitted Gustavo's line just above, because
the history block told it to "responde só à última". Earlier, "quem fez merda @Kaya
tu ou o fred?" was rewritten as "o Peter ou o Frederico?" and Peter got roasted.
"""
from src.chat import router
from tests.test_engine_roast import ScriptedBackend, make_engine, user_turn

HISTORY = [
    "Kaya Bot: O Peter está com a cabeça tão na lua que nem distingue uma tasquinha de um link.",
    "Frederico: [a responder a Kaya Bot: O Peter está com a cabeça tão na lua] Vai dormir bro",
    "Gustavo: [a responder a Frederico: Vai dormir bro] Yeah he's drunk, eu já falo com ele",
]
ASKED = "[a responder a Kaya Bot: O Peter está com a cabeça tão na lua] That's my g"


def _turn(message=ASKED, speaker="Peter", recent=HISTORY, **kw):
    backend = ScriptedBackend("BANTER")
    make_engine(backend).respond(message, speaker, list(recent), "sys", **kw)
    return user_turn(backend)


def test_history_no_longer_points_at_its_own_last_line():
    turn = _turn()
    assert "responde só à última" not in turn
    assert "Não respondas a nenhuma destas linhas" in turn


def test_the_answered_message_is_labelled_and_comes_last():
    turn = _turn()
    label = "Mensagem a que respondes agora:\nPeter: " + ASKED
    assert label in turn
    assert turn.index(HISTORY[-1]) < turn.index(label)
    assert turn.rindex("Peter: ") == turn.index(label) + len("Mensagem a que respondes agora:\n")


def test_a_reply_to_the_bot_is_answered_as_the_speakers_reaction():
    turn = _turn()
    assert "A mensagem responde a uma mensagem tua" in turn
    assert "Responde ao que o Peter diz sobre ela, não às linhas da conversa acima." in turn


def test_without_history_the_turn_is_unchanged():
    turn = _turn(message="olá", recent=[])
    assert "\n\nPeter: olá" in turn and "Mensagem a que respondes agora" not in turn


def test_tu_is_the_bot():
    turn = _turn(message="quem fez merda tu ou o fred?", speaker="Carnall")
    assert '"tu", "te", "ti", "contigo" e "você" referem-se a ti, o Kaya Bot.' in turn


def test_router_rewrite_keeps_tu_as_the_bot():
    assert '"quem fez merda tu ou o fred?" -> Q: quem fez asneira, a Kaya ou o Frederico?' in router._ROUTER_SYSTEM
    assert 'write\n"a Kaya" for them, never a member\'s name' in router._ROUTER_SYSTEM


def test_late_note_reaches_the_prompt_before_the_message():
    note = "Esta mensagem é das 17:23 (há 6 horas)."
    turn = _turn(message="quem fez merda tu ou o fred?", speaker="Carnall", late_note=note)
    assert note in turn and turn.index(note) < turn.index("Mensagem a que respondes agora")
    assert note not in _turn(message="quem fez merda tu ou o fred?", speaker="Carnall")
