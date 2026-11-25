"""Which requests are self-roasts, and which names a negation rules out (src/chat/roast.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.roast import is_self_aimed, self_roast_hint, strip_negated


@pytest.mark.parametrize("text", [
    "It’s been a while, roast me alive my g, and don’t get it wrong: I’m Pedro Ninguém or "
    "Peter, I’m not the engineer abroad, I’m the one who lives here",
    "roast me", "Kaya, roast me bro", "faz-me um roast", "me faz um roast", "insulta-me",
    "goza comigo à vontade", "destrói-me", "do a roast of me",
])
def test_self_roasts(text):
    assert is_self_aimed(text)


@pytest.mark.parametrize("text", [
    "roast the Gil", "roast the guy who sent me this", "quem é o mais burro?",
    "goza com o Rafa", "manda-me o link", "diz-me o que o Gil faz", "",
])
def test_not_self_roasts(text):
    assert not is_self_aimed(text)


def test_a_negated_name_is_removed():
    assert "Rafa" not in strip_negated("roast the Gil, not Rafa")
    assert "Rafa" not in strip_negated("goza com o Gil e não o Rafa por favor")
    assert "Gil" in strip_negated("goza com o Gil e não o Rafa por favor")


def test_the_hint_names_the_speaker_and_asks_for_second_person():
    hint = self_roast_hint("Peter")
    assert "Peter" in hint and "segunda pessoa" in hint and "nunca são alvos" in hint
