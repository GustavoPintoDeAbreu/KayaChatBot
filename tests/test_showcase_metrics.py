"""The showcase page's injected numbers: what the page gets, and what it must never get.

labwatch's LAN feed is unreachable here (and from the browser), so the tests pin both the
feed mapping and the graceful fallback — flows and prompt with no metric bars. The feed mapping
is pinned both ways: an unreachable feed still yields a page, and a feed that answers lands on
the landing page's own project ids."""
import sys
from pathlib import Path

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
    assert view["lab"]["stats"] == []
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
    bars = {row["key"]: row for row in view["lab"]["stats"]}
    electricity = bars["electricity_kwh_this_week"]
    assert electricity["value"] == 41.5
    assert electricity["unit"] == " kWh"
    assert electricity["en"] == "electricity this week"
    assert electricity["pt"] == "eletricidade esta semana"


def test_render_injection_escapes_html():
    rendered = showcase_metrics.render_injection({"prompt": {"labwatch": "</script><script>alert(1)</script>"}})
    assert rendered.count("</script>") == 1
    assert "</script><script>alert" not in rendered
    assert "\\u003c/script\\u003e" in rendered


def test_fetch_public_returns_none_when_unreachable():
    assert showcase_metrics.fetch_public(url="http://127.0.0.1:1/", timeout=0.2) is None
