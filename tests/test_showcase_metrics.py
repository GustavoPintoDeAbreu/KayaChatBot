"""The showcase page's injected numbers: what the page gets, and what it must never get.

labwatch's LAN feed is unreachable here (and from the browser), so the tests pin both the
feed mapping and the graceful fallback — flows and prompt with no metric bars, and no lab-wide
numbers standing in for a project's own. The feed mapping
is pinned both ways: an unreachable feed still yields a page, and a feed that answers lands on
the landing page's own project ids."""
import re
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chat import showcase_metrics

EXPECTED_STAT_LABELS_PT = {
    "notes added, all time": "notas adicionadas, desde sempre",
    "ideas sent, all time": "ideias enviadas, desde sempre",
    "average idea to pull request, minutes": "minutos em média da ideia ao pull request",
    "local code Claude graded pass, %": "% de código local aprovado pelo Claude",
    "fixes shipped by the medic, all time": "correções enviadas pelo médico, desde sempre",
    "subtitle files translated, all time": "ficheiros de legendas traduzidos, desde sempre",
    "items followed": "artigos seguidos",
    "messages Kaya read this week": "mensagens que Kaya leu esta semana",
    "replies Kaya sent this week": "respostas que Kaya enviou esta semana",
    "voice replies this week": "respostas por voz esta semana",
    "episodes made this week": "episódios feitos esta semana",
    "episodes generated, all time": "episódios feitos, desde sempre",
    "price sweeps this week": "pesquisas de preços esta semana",
    "deal alerts sent this week": "alertas de promoções esta semana",
}

ENGLISH_ONLY_WORDS = (r"\bthe\b", r"\band\b", r"\bfrom\b", r"\bby\b", r"\bor\b", r"\bwith\b",
                      r"\bidea\b", r"\bin total\b", r"\badded\b", r"\bsent\b", r"\btext\b",
                      r"\bvoice\b", r"\bpicks\b", r"\bwrites\b", r"\basks\b", r"\builds\b",
                      r"\breviews\b", r"\bopens\b")


def _english_leftovers(text: str) -> list:
    return [pattern for pattern in ENGLISH_ONLY_WORDS if re.search(pattern, text)]


def test_inject_adds_one_script_before_body_close():
    html = "<html><body><p>x</p></body></html>"
    out = showcase_metrics.inject(html, showcase_metrics.build_view(None))
    assert "window.KAYA_SHOWCASE" in out
    assert out.count("window.KAYA_SHOWCASE") == 1
    assert out.index("</script>") < out.rindex("</body>")


def test_inject_is_idempotent():
    html = "<html><body><p>x</p></body></html>"
    view = showcase_metrics.build_view(None)
    once = showcase_metrics.inject(html, view)
    assert showcase_metrics.inject(once, view) == once


def test_build_view_without_feed_carries_flows_and_prompt():
    view = showcase_metrics.build_view(None)
    assert set(view["flows"]) == {
        "kaya", "lab-assistant", "lab-notes", "idea-pipeline", "lab-medic", "labwatch",
        "flight-deals", "price-scout", "series-watch", "game-alerts", "redditcast", "subs",
        "paywatch", "personal-hub", "lab-browser",
    }
    assert "lab" not in view
    assert view["products"] == {}
    assert view["prompt"]["labwatch"] == showcase_metrics.LABWATCH_PROMPT


def test_build_view_maps_feed_products_to_landing_ids_and_localises():
    payload = {
        "lab": {"stats": {"electricity_kwh_this_week": 41.5}},
        "products": [
            {"name": "Kaya", "stats": [{"label": "replies Kaya sent this week", "value": 12}]},
            {"name": "labwatch", "stats": []},
        ],
    }
    view = showcase_metrics.build_view(payload)
    assert view["products"]["kaya"]["stats"] == [
        {"en": "replies Kaya sent this week",
         "pt": "respostas que Kaya enviou esta semana",
         "value": 12,
         "scope": "week"}
    ]
    assert "labwatch" not in view["products"]


def test_lab_wide_totals_never_stand_in_for_a_project():
    """A card with no counters of its own once got the lab's tokens, so every project showed the
    same numbers; the lab totals are not in the view at all now."""
    payload = {
        "lab": {"stats": {"local_ai_tokens_generated_this_week": 123456}},
        "products": [{"name": "redditcast", "stats": [{"label": "episodes made this week", "value": 1}]}],
    }
    view = showcase_metrics.build_view(payload)
    assert "lab" not in view
    assert set(view["products"]) == {"redditcast"}
    assert "123456" not in showcase_metrics.render_injection(view)


def test_render_injection_escapes_html():
    rendered = showcase_metrics.render_injection({"prompt": {"labwatch": "</script><script>alert(1)</script>"}})
    assert rendered.count("</script>") == 1
    assert "</script><script>alert" not in rendered
    assert "\\u003c/script\\u003e" in rendered


def test_fetch_public_returns_none_when_unreachable():
    assert showcase_metrics.fetch_public(url="http://127.0.0.1:1/", timeout=0.2) is None


@pytest.fixture
def offline_feed(monkeypatch):
    """An unreachable feed, counted: the point of the failure cache is fewer fetches."""
    monkeypatch.setattr(showcase_metrics, "_CACHE", None)
    monkeypatch.setattr(showcase_metrics, "_FAILED_AT", None)
    counts = {"get": 0}

    def down(*args, **kwargs):
        counts["get"] += 1
        raise showcase_metrics.httpx.ConnectError("labwatch is down")

    monkeypatch.setattr(showcase_metrics.httpx, "get", down)
    return counts


def test_a_failed_fetch_is_cached_so_the_next_load_is_fast(offline_feed):
    """While labwatch is down a page load must not wait the fetch timeout every time."""
    assert showcase_metrics.fetch_public() is None
    assert showcase_metrics.fetch_public() is None
    assert showcase_metrics.fetch_public() is None
    assert offline_feed["get"] == 1


@pytest.fixture
def stale_then_down(monkeypatch):
    """A cache holding numbers older than CACHE_TTL_S, with the feed now down."""
    import time

    monkeypatch.setattr(showcase_metrics, "_FAILED_AT", None)
    good = {"lab": {"stats": {"electricity_kwh_this_week": 12.0}}}
    monkeypatch.setattr(showcase_metrics, "_CACHE", (time.time() - showcase_metrics.CACHE_TTL_S - 1, good))
    counts = {"get": 0}

    def down(*args, **kwargs):
        counts["get"] += 1
        raise showcase_metrics.httpx.ConnectError("labwatch is down")

    monkeypatch.setattr(showcase_metrics.httpx, "get", down)
    return good, counts


def test_the_last_good_numbers_survive_a_failure(stale_then_down):
    """Past the success TTL the feed is asked again; if it is down, the stale numbers go out —
    and the failure is cached, so the next load does not pay the timeout either."""
    good, counts = stale_then_down
    assert showcase_metrics.fetch_public() == good
    assert counts["get"] == 1
    assert showcase_metrics.fetch_public() == good
    assert counts["get"] == 1


def test_stat_labels_translate_fully_into_european_portuguese():
    """2026-11-12: the PT deck labels were mixed-language ('nota added, in total',
    'episódios feitos, in total'); each feed label now carries a full pt-PT translation."""
    assert showcase_metrics.STAT_LABEL_PT == EXPECTED_STAT_LABELS_PT


def test_the_portuguese_deck_carries_no_english_leftovers():
    """'Kaya lê e classifies by intention' and 'nota added, in total' shipped because nothing
    checked the PT strings. Every PT stat label and flow step is scanned for English-only
    words; loanwords the page itself uses (pull request, sandbox, browser) stay allowed."""
    for english, pt in showcase_metrics.STAT_LABEL_PT.items():
        assert pt != english, english
        assert not _english_leftovers(pt), (english, pt)
    for project, flows in showcase_metrics.FLOWS.items():
        assert set(flows) == {"en", "pt"}, project
        assert len(flows["pt"]) == len(flows["en"]), project
        for en_step, pt_step in zip(flows["en"], flows["pt"]):
            assert pt_step != en_step, (project, en_step)
            assert not _english_leftovers(pt_step), (project, pt_step)


def test_the_repaired_portuguese_flows_read_as_the_page_does():
    assert showcase_metrics.FLOWS["kaya"]["pt"] == [
        "Um amigo escreve no grupo de WhatsApp",
        "Kaya lê e classifica pela intenção",
        "Procura no histórico do grupo",
        "Responde, por texto ou por voz",
    ]
    assert showcase_metrics.FLOWS["lab-assistant"]["pt"] == [
        "Chega um áudio, por DM",
        "É transcrito em casa",
        "Kaya escolhe a ferramenta",
        "Responde, ou espera pelo teu sim",
    ]
    assert showcase_metrics.FLOWS["lab-notes"]["pt"] == [
        "Ditas uma nota, ou escreves, no WhatsApp",
        "É transcrita",
        "Vai para a página certa do caderno",
    ]
    assert showcase_metrics.FLOWS["idea-pipeline"]["pt"] == [
        "Envio uma ideia pelo telemóvel, por voz ou texto",
        "Um agente de planeamento faz as perguntas e escreve um plano",
        "O agente de implementação constrói-o numa sandbox",
        "Um agente de QA revê-o e abre um pull request para eu rever",
    ]


def test_the_dry_run_reads_a_served_page():
    served = (
        '<!doctype html><html><head><title>Kaya, a home lab of AI agents</title></head><body>'
        '<div class="card-in"><b>12 projects</b></div>'
        '<div class="card-in"><b>12 projetos</b></div>'
        '<article class="project" data-project="kaya"></article>'
        '<article class="project" data-project="kaya"></article>'
        '<script>window.KAYA_SHOWCASE = {"products":{"kaya":{"stats":['
        '{"en":"notes added, all time","pt":"notas adicionadas, desde sempre","value":3,"scope":"all"}]}}};</script>'
        '</body></html>'
    )
    assert showcase_metrics._at_a_glance(served) == {"en": 12, "pt": 12}
    assert showcase_metrics._project_cards(served) == 1
    script = showcase_metrics._showcase_script(served)
    assert script["products"]["kaya"]["stats"][0]["pt"] == "notas adicionadas, desde sempre"


def test_the_dry_run_survives_every_source_being_down(capsys, monkeypatch):
    monkeypatch.setattr(showcase_metrics, "_CACHE", None)
    monkeypatch.setattr(showcase_metrics, "_FAILED_AT", None)

    def down(*args, **kwargs):
        raise showcase_metrics.httpx.ConnectError("everything is down")

    monkeypatch.setattr(showcase_metrics.httpx, "get", down)
    assert showcase_metrics.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "feed unreachable" in out
    assert "local page" in out
    assert "live page" in out and "unreachable" in out
