"""An opinion may vary; a fact may not.

Peter asked to be roasted four times over three days and got the same four beats
every time — Rotterdam and Queijas, editing other people's videos, Five Guys,
posting concert videos like a music critic. Romano got "analista político por ler
tweets" five times. Nothing was wrong with the model: the WhatsApp system prompt
is built once at import, so the member profiles were byte-identical for the whole
uptime, and `key_facts[:max_facts]` handed over the same first four facts in the
same order on every turn.

Three things are tested here: which turns are allowed to vary, that the facts are
drawn rather than truncated, and that the bot is told what it already said about
the person in front of it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import variety
from src.chat.response_utils import build_member_prompt_suffix


def row(mode, reply, members, command=""):
    return {"route_mode": mode, "route_command": command,
            "assistant_response": reply, "reply_members": list(members)}


ROAST_PETER = ("O Peter vive entre Roterdão e Queijas só para editar vídeos de "
               "outros, e come Five Guys como quem preenche um vazio.")
ROAST_PETER_2 = ("O Peter partilha vídeos de concertos como se fosse o maior "
                 "crítico musical da história, mas só edita o que os outros filmam.")


class TestWhatMayVary:
    """A count must come out the same every time it is asked."""

    @pytest.mark.parametrize("mode", ["banter", "mixed", "roast"])
    def test_open_ended_modes_may_vary(self, mode):
        assert variety.is_open_ended(mode) is True

    @pytest.mark.parametrize("mode", ["factual", "general"])
    def test_factual_answers_may_not(self, mode):
        assert variety.is_open_ended(mode) is False

    def test_a_command_never_varies(self):
        """CMD_COUNT borrows the factual mode config; it must not borrow this."""
        assert variety.is_open_ended("factual", "count") is False
        assert variety.is_open_ended("banter", "image") is False


class TestRecentLines:
    def test_it_finds_what_was_said_about_the_person(self):
        rows = [row("roast", ROAST_PETER, ["Peter"])]
        assert "Roterdão" in variety.recent_lines_about(["Peter"], rows)["Peter"][0]

    def test_somebody_never_talked_about_has_nothing_to_avoid(self):
        rows = [row("roast", ROAST_PETER, ["Peter"])]
        assert variety.recent_lines_about(["Ricky"], rows) == {}

    def test_the_newest_lines_come_first(self):
        rows = [row("roast", ROAST_PETER, ["Peter"]),
                row("roast", ROAST_PETER_2, ["Peter"])]
        lines = variety.recent_lines_about(["Peter"], rows)["Peter"]
        assert "concertos" in lines[0]

    def test_it_stops_at_the_limit(self):
        rows = [row("roast", f"O Peter é assim numero {n} e mais umas palavras aqui", ["Peter"])
                for n in range(10)]
        assert len(variety.recent_lines_about(["Peter"], rows, limit=3)["Peter"]) == 3

    def test_identical_replies_are_not_listed_twice(self):
        rows = [row("roast", ROAST_PETER, ["Peter"])] * 4
        assert len(variety.recent_lines_about(["Peter"], rows)["Peter"]) == 1

    def test_several_people_at_once(self):
        rows = [row("roast", ROAST_PETER, ["Peter"]),
                row("roast", "O Romano só manda stickers o dia todo sem parar nunca", ["Romano"])]
        found = variety.recent_lines_about(["Peter", "Romano"], rows)
        assert set(found) == {"Peter", "Romano"}


class TestWhatDoesNotCount:
    """The hint has to be material, not noise."""

    def test_a_factual_answer_is_not_material(self):
        """A tally names every member and is not a joke anyone can repeat."""
        rows = [row("factual", "Rafa: 198, Gustavo: 37, Gil: 9, e por aí fora sempre",
                    ["Rafa", "Gustavo", "Gil"])]
        assert variety.recent_lines_about(["Rafa"], rows) == {}

    def test_a_command_turn_is_not_material(self):
        rows = [row("factual", "Aqui tens a lista actualizada com os números todos",
                    ["Rafa"], command="count")]
        assert variety.recent_lines_about(["Rafa"], rows) == {}

    def test_a_greeting_carries_no_angle(self):
        rows = [row("banter", "Yo Peter, what's up man?", ["Peter"])]
        assert variety.recent_lines_about(["Peter"], rows) == {}

    def test_a_reply_naming_half_the_group_is_about_nobody(self):
        everyone = ["Rafa", "Gil", "Peter", "Gustavo", "Romano", "Pedro", "Mateus"]
        rows = [row("roast", "Top 3 intelectuais: Gustavo, Bernardo e Manuel, e o resto "
                             "que se governe como puder", everyone)]
        assert variety.recent_lines_about(["Peter"], rows) == {}

    def test_a_multiline_reply_is_flattened(self):
        rows = [row("roast", "O Peter vive em Roterdão\ne come Five Guys\ntodos os dias",
                    ["Peter"])]
        line = variety.recent_lines_about(["Peter"], rows)["Peter"][0]
        assert "\n" not in line


class TestHint:
    def test_it_names_the_person_and_quotes_the_material(self):
        rows = [row("roast", ROAST_PETER, ["Peter"])]
        hint = variety.hint_for(["Peter"], rows)
        assert "Peter" in hint and "Roterdão" in hint
        assert "Não repitas" in hint

    def test_nothing_to_avoid_means_no_hint(self):
        assert variety.hint_for(["Ricky"], []) == ""
        assert variety.hint_for([], []) == ""

    def test_a_broken_log_never_costs_a_reply(self):
        """A hint is never worth a dropped message."""
        assert variety.hint_for(["Peter"], [{"reply_members": None}]) == ""


class TestFactDraw:
    """`key_facts[:max_facts]` takes the same first N every time — which is how a
    roast became a recital."""

    MEMBERS = {"members": [{
        "name": "Peter", "aliases": [],
        "key_facts": ["mora em Roterdão", "edita vídeos", "come Five Guys",
                      "partilha concertos", "tem um cão chamado Kobe"],
    }]}

    def _peter(self, **kw):
        text = build_member_prompt_suffix(self.MEMBERS, **kw)
        return next(l for l in text.splitlines() if l.startswith("- Peter"))

    def test_truncation_is_identical_every_time(self):
        seen = {self._peter(max_facts=3) for _ in range(20)}
        assert len(seen) == 1

    def test_sampling_varies(self):
        seen = {self._peter(max_facts=3, sample_facts=True) for _ in range(40)}
        assert len(seen) > 1, "sampling produced one fixed draw"

    def test_sampling_still_respects_the_cap(self):
        line = self._peter(max_facts=2, sample_facts=True)
        assert line.count(".") <= 3  # two facts plus the trailing stop

    def test_the_fact_truncation_dropped_never_surfaces_without_sampling(self):
        """Kobe is the fifth fact, so a cap of 4 means the model has never seen it."""
        assert "Kobe" not in self._peter(max_facts=4)
        assert any("Kobe" in self._peter(max_facts=4, sample_facts=True)
                   for _ in range(60))

    def test_no_cap_keeps_everything(self):
        line = self._peter(max_facts=0, sample_facts=True)
        for fact in ("Roterdão", "Five Guys", "Kobe"):
            assert fact in line

    def test_fewer_facts_than_the_cap_is_not_an_error(self):
        thin = {"members": [{"name": "Murgeiro", "aliases": [], "key_facts": ["aparece sempre"]}]}
        text = build_member_prompt_suffix(thin, max_facts=4, sample_facts=True)
        assert "aparece sempre" in text
