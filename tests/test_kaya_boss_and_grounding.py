"""Who "o teu chefe" is, and no facts borrowed from what someone said (2026-11-19).

A member asked "achas que o teu chefe tem capacidade?" and the router rewrote it
as "o chefe do Gustavo"; asked "quem é este gajo?" about a member, the bot gave him
another member's job, from his own message about that job. Fictional members here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import router
from src.chat.engine import render_maintainer_clause
from src.chat.response_utils import build_member_prompt_suffix
from src.config_loader import load_config

CONFIG = load_config(str(Path(__file__).parent.parent / "config.yaml"))


def test_the_router_knows_the_bots_possessives_and_its_boss():
    prompt = router.router_system(CONFIG)
    assert '"teu", "tua", "teus", "tuas" and "your" are the bot\'s' in prompt
    assert '"achas que o teu chefe tem capacidade?" -> Q: o Gustavo, o criador da Kaya, tem capacidade?' in prompt
    assert 'never "o chefe do Gustavo"' in prompt
    assert "{maintainer}" not in prompt


def test_without_a_maintainer_the_router_still_reads_cleanly():
    prompt = router.router_system({})
    assert "{maintainer}" not in prompt and "the person who made the bot" in prompt


def test_the_classifier_sends_the_filled_prompt():
    class Backend:
        def __init__(self):
            self.messages = None

        def generate(self, messages, **kw):
            self.messages = messages
            return "MIXED\nQ: o Gustavo, o criador da Kaya, tem capacidade?"

    backend = Backend()
    route = router.classify(backend, CONFIG, "achas que o teu chefe tem capacidade?", speaker="Diogo")
    assert "Gustavo, o criador da Kaya" in backend.messages[0]["content"]
    assert route.query == "o Gustavo, o criador da Kaya, tem capacidade?"


def test_the_persona_knows_who_made_it():
    to_others = render_maintainer_clause(CONFIG)
    to_him = render_maintainer_clause(CONFIG, to_maintainer=True)
    assert to_others.startswith("Foste feito pelo Gustavo e corres no PC dele")
    assert "teu chefe, patrão, dono, criador" in to_others
    assert "é ele quem te está a escrever agora" in to_him
    assert "{maintainer}" not in to_others + to_him


def test_member_cards_forbid_inferred_jobs_habits_and_networks():
    suffix = build_member_prompt_suffix({"members": [{"name": "Bruno", "key_facts": ["Bruno loves football."]}]})
    assert "nem deduzas o emprego, os hábitos, as redes sociais" in suffix


def test_retrieved_lines_are_what_people_said_not_facts_about_them():
    from src.chat.retriever import ConversationRetriever

    retriever = ConversationRetriever.__new__(ConversationRetriever)
    text = retriever.format_context([{"text": "Bruno: quem trabalhava na acme era a ana"}])
    assert "cada linha é o que essa pessoa disse" in text
    assert "não é um facto sobre quem o disse" in text
