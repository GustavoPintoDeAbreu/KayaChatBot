"""The showcase page's injected numbers: what the page gets, and what it must never get.

labwatch's LAN feed is unreachable here (and from the browser), so the tests pin both the
feed mapping and the graceful fallback — flows and prompt with no metric bars, and no lab-wide
numbers standing in for a project's own. The feed mapping
is pinned both ways: an unreachable feed still yields a page, and a feed that answers lands on
the landing page's own project ids."""
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chat import showcase_metrics


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
         "value": 12}
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
