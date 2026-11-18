"""A tag word at the end of a sentence (", bro."), and the hint that stops it repeating (2026-11-18)."""
from src.chat import variety


def test_tags_at_the_end_of_sentences_are_found():
    assert variety.closing_tags("More like a recipe for an early grave, bro.") == ["bro"]
    assert variety.closing_tags("O lucro não vale um enterro precoce, mano.") == ["mano"]
    assert variety.closing_tags("That's a recipe for disaster, man. Mixing that is bad.") == ["man"]
    assert variety.closing_tags("Tudo depende do teu nível de interpretação, bro") == ["bro"]
    assert variety.closing_tags("Confio em ti, puto.") == ["puto"]


def test_ordinary_endings_are_not_tags():
    assert variety.closing_tags("Pois é, estamos todos no limite aqui.") == []
    assert variety.closing_tags("The man is here.") == []
    assert variety.closing_tags("Bro, that's wild.") == []


def rows(*replies, command=None):
    return [{"assistant_response": r, "route_mode": "banter", "route_command": command} for r in replies]


def test_the_hint_names_tags_used_lately_newest_first():
    log = rows("O lucro não vale um enterro precoce, mano.", "Ok.", "Tudo depende, bro.")
    hint = variety.closer_hint_for(log, window=8)
    assert "«, bro»" in hint and "«, mano»" in hint and hint.index("bro") < hint.index("mano")


def test_no_hint_when_no_tag_was_used_or_outside_the_window():
    assert variety.closer_hint_for(rows("Pois é.", "Claro que sim."), window=8) == ""
    old = rows("Isso é grave, bro.") + rows(*["Normal reply."] * 8)
    assert variety.closer_hint_for(old, window=8) == ""
    assert variety.closer_hint_for(rows("Feito, mano.", command="text"), window=8) == ""
