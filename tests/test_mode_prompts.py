"""The per-mode prompts are trimmed personas, and the trimming lost rules.

`banter`, `mixed` and `general` each carry their own system prompt so a "😂" is
not answered with an essay. They were written by cutting the detailed persona
down, and three rules went out with the cut — which is exactly what the group
spent an evening running into:

* nothing said "do it, don't announce it", so asked to rap, the bot replied
  "prepara o psicológico", "ainda nem comecei", "prepara-te para apanhar" and
  "estava só a deixar-te ganhar tempo" over four turns until Gil wrote "Chega de
  'prepara-te'. Faz o que disseste que ias fazer";
* nothing forbade a trailing emoji, so 28% of banter replies ended in one, 36 of
  them 😂; and
* nothing forbade signing off with the addressee's name, so 22% of all banter
  replies — and 11 of the last 14 — ended ", <Nome>. 😂".

These tests are about the SHAPE of the config, not about model behaviour, so they
run without a GPU and fail the moment a rule is dropped from a mode again.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

CONFIG = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mode_prompts(config):
    """Every mode that overrides the detailed persona with one of its own."""
    modes = config["chat"]["modes"]
    return {name: cfg["system_prompt"]
            for name, cfg in modes.items() if cfg.get("system_prompt")}


def test_the_modes_with_their_own_prompt_are_the_expected_ones(mode_prompts):
    """`factual` and `roast` leave system_prompt null on purpose — roasting needs
    the member profiles the trimmed prompt drops."""
    assert set(mode_prompts) == {"banter", "mixed", "general"}


class TestConversationalModes:
    """banter and mixed talk to people, so they need the conversational rules."""

    MODES = ("banter", "mixed")

    @pytest.mark.parametrize("mode", MODES)
    def test_it_must_act_rather_than_announce(self, mode, mode_prompts):
        prompt = mode_prompts[mode]
        assert "anuncies" in prompt or "anunciares" in prompt

    @pytest.mark.parametrize("mode", MODES)
    def test_it_must_not_end_on_an_emoji(self, mode, mode_prompts):
        assert "termines com emojis" in mode_prompts[mode]

    @pytest.mark.parametrize("mode", MODES)
    def test_it_must_not_sign_off_with_the_name(self, mode, mode_prompts):
        assert "nome de quem te falou" in mode_prompts[mode]


class TestRulesEveryModeKeeps:
    """Rules whose absence produced a filed bug once already."""

    def test_no_em_dashes_anywhere(self, mode_prompts):
        for name, prompt in mode_prompts.items():
            assert "travessões" in prompt, f"{name} lost the em-dash rule"

    def test_european_portuguese_everywhere(self, mode_prompts):
        for name, prompt in mode_prompts.items():
            assert "português europeu" in prompt, f"{name} lost the language rule"

    def test_the_group_facing_modes_know_they_can_speak(self, mode_prompts):
        """"nunca digas que não consegues gerar áudio" — `general` is exempt, it
        is not about the group and never gets asked."""
        for name in ("banter", "mixed"):
            assert "mensagens de voz" in mode_prompts[name]

    def test_the_maintainer_clause_is_templated_not_hardcoded(self, mode_prompts, config):
        """A hardcoded name told Gustavo that Gustavo had to fix it."""
        for name in ("banter", "mixed"):
            assert "{maintainer_clause}" in mode_prompts[name]
        assert config["chat"]["maintainer"]


class TestGeneralStaysOutOfTheGroup:
    """`general` answers the world; pulling in members is what it exists to stop."""

    def test_it_is_told_not_to_mention_members(self, mode_prompts):
        assert "Não menciones membros do grupo" in mode_prompts["general"]

    def test_it_does_not_carry_the_member_rules(self, mode_prompts):
        assert "Imagem enviada por" not in mode_prompts["general"]


class TestPlaceholdersResolve:
    """A leaked "{...}" would be read aloud by the model."""

    def test_every_mode_prompt_resolves(self, config, mode_prompts):
        from src.chat.engine import build_mode_system_prompt

        for name, prompt in mode_prompts.items():
            built = build_mode_system_prompt(config, prompt)
            assert "{" not in built, f"{name} leaked a placeholder: {built[:200]}"

    def test_the_detailed_prompt_resolves(self, config):
        from src.chat.engine import fill_prompt_defaults

        assert "{" not in fill_prompt_defaults(config, config["data"]["system_prompt"])
