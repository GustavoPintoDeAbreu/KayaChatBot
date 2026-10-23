"""The public showcase page: what it may say, and what it must not.

The page is public — it is handed to the group and can be forwarded anywhere —
so it carries no real member names, no infrastructure details and no login.
`whatsapp_server` cannot be imported here — it builds the engine at module
scope (`engine = get_engine(config)`), which loads the model — so the routes
are checked by parsing the source.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chat import static_assets

LANDING = BASE_DIR / "src" / "chat" / "static" / "landing.html"
SERVER = BASE_DIR / "src" / "chat" / "whatsapp_server.py"
ASSETS = BASE_DIR / "src" / "chat" / "static" / "showcase"

LINKEDIN = "https://www.linkedin.com/in/gustavo-abreu-7687b0133/"

SHOWN = ("kaya", "lab-assistant", "idea-pipeline", "lab-medic", "labwatch", "lab-notes",
         "redditcast", "series-watch", "flight-deals", "price-scout", "game-alerts", "subs")
HIDDEN = ("lab-audit", "lab audit", "lab-infra", "house-hunt", "house hunt", "daily-git",
          "daily git", "photoshop", "comfyui")
INFRA_TERMS = ("gemma", "qwen", "ollama", "llama", "whisper", "piper", "kokoro", "bge-m3",
               "chromadb", "gradio", "fastapi", "uvicorn", "sqlite", "docker", "waha", "noweb",
               "searxng", "broker", "grok", "xai", "azure", "openai", "cloudflare", "tunnel",
               "tailscale", "raspberry", "pi5", "rtx", "3090", "nvidia", "ryzen", "gpu",
               "localhost", ".local", "wake-on-lan", "shutdown", "firewall", "trivy", "gitleaks",
               "osv-scanner", "lynis", "rkhunter", "chkrootkit", "rustdesk", "ssh", "sudo",
               "proxy", "serpapi", "evomi", "tmdb", "mcp", "/idea", "/homelab", "/status",
               "/app", "07:00", "kaya-prod", "sigmakayachat")
INFRA_PATTERNS = (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", r"\b\d+\.\d+\.\d+\b", r"\bv\d+(?:\.\d+)+\b",
                  r"\b\d{1,3}[bB]\b", r"\+\d{6,}", r"\b\d{9,}\b", r"[\w.+-]+@[\w-]+\.\w+",
                  r"\bPi\b", r"\bLAN\b", r"\bports?\b")

MEMBERS_FILE = BASE_DIR / "data" / "group_members.json"


def _real_names() -> list:
    """The members, read from their own profiles rather than copied.

    This list used to be hardcoded, and had gone stale: two members joined the
    group and the guard did not know about them, so the one test standing between
    a real name and a public page had a hole in it for exactly those two. Reading
    the file means the guard cannot fall behind the group again.
    """
    import json

    if not MEMBERS_FILE.exists():  # gitignored; CI checkouts may not have it
        return []
    data = json.loads(MEMBERS_FILE.read_text(encoding="utf-8"))
    names = []
    for member in data.get("members", []):
        names.append(member["name"])
        names.extend(member.get("aliases", []))
    return sorted({n for n in names if len(n) > 2})


# The landing page is public — it is handed to the group and can be forwarded
# anywhere — so it uses invented names throughout.
REAL_NAMES = _real_names()


@pytest.fixture(scope="module")
def landing() -> str:
    return LANDING.read_text(encoding="utf-8")


def _without_code(text: str) -> str:
    """The visible text: style and script blocks removed."""
    return re.sub(r"(?is)<(style|script)\b.*?</\1>", " ", text)


def _public_texts() -> list:
    """Every file a visitor can read from the public page."""
    texts = [(LANDING.name, LANDING.read_text(encoding="utf-8"))]
    for path in sorted(ASSETS.iterdir()):
        texts.append((path.name, path.read_text(encoding="utf-8")))
    return texts


def test_landing_page_exists(landing):
    assert len(landing) > 8_000, "landing page looks truncated"


def test_english_by_default_with_a_portuguese_switch(landing):
    assert 'id="page" data-lang="en"' in landing
    assert 'class="doc" data-lang="en"' in landing
    assert 'class="doc" data-lang="pt"' in landing
    assert 'data-value="en" aria-pressed="true"' in landing
    assert 'data-value="pt" aria-pressed="false"' in landing


def test_every_shown_project_has_a_card_in_both_languages(landing):
    for project in SHOWN:
        assert landing.count(f'data-project="{project}"') == 2, project
    assert len(re.findall(r'data-project="', landing)) == 2 * len(SHOWN)


def test_hidden_projects_are_not_mentioned():
    for name, text in _public_texts():
        lowered = text.lower()
        leaked = [term for term in HIDDEN if term in lowered]
        assert not leaked, f"{name} mentions hidden project(s): {leaked}"


def test_no_infrastructure_details_are_published():
    for name, text in _public_texts():
        body = _without_code(text).replace(LINKEDIN, "")
        # The page references its own images by path (e.g. /assets/idea-pipeline.svg);
        # those filenames are not infrastructure, so drop them before the checks.
        body = re.sub(r'src="/assets/[^"]+"', " ", body)
        lowered = body.lower()
        leaked = [term for term in INFRA_TERMS if term in lowered]
        assert not leaked, f"{name} publishes infrastructure detail(s): {leaked}"
        for pattern in INFRA_PATTERNS:
            match = re.search(pattern, body)
            assert match is None, f"{name} matches {pattern!r}: {match.group(0)!r}"


def test_only_linkedin_is_external(landing):
    hrefs = re.findall(r'href="(https?://[^"]+)"', landing)
    assert hrefs, "the page must link somewhere"
    assert all(href == LINKEDIN for href in hrefs), hrefs
    linkedin_count = landing.count(f'href="{LINKEDIN}"')
    safe_count = landing.count(f'href="{LINKEDIN}" target="_blank" rel="noopener noreferrer"')
    assert linkedin_count == safe_count
    assert not re.search(r'src="http', landing)


def test_images_are_local_assets_that_exist(landing):
    srcs = re.findall(r'<img[^>]+src="([^"]+)"', landing)
    assert len(srcs) >= 8
    referenced = set()
    for src in srcs:
        assert src.startswith("/assets/"), src
        name = src[len("/assets/"):]
        assert static_assets.asset_path(ASSETS, name) is not None, src
        referenced.add(name)
    on_disk = {p.name for p in ASSETS.iterdir()}
    assert referenced == on_disk, (on_disk - referenced, referenced - on_disk)
    alts = re.findall(r'<img[^>]+alt="([^"]*)"', landing)
    assert all(alt for alt in alts)


def test_assets_are_inert_and_small():
    for path in sorted(ASSETS.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert len(text.encode("utf-8")) < 200_000, path.name
        lowered = text.lower()
        for forbidden in ("<script", "<foreignobject", "<image", "href=", "<style"):
            assert forbidden not in lowered, (path.name, forbidden)
        assert re.search(r"\son\w+\s*=", text) is None, path.name


def test_the_friends_chat_login_is_gone(landing):
    for marker in ("/app", "Log in", "Entrar"):
        assert marker not in landing, marker


def test_the_page_does_not_poll_the_pc(landing):
    assert "/status" not in landing
    assert "fetch(" not in landing


def test_no_real_member_names_are_published():
    for name, text in _public_texts():
        leaked = [n for n in REAL_NAMES if re.search(rf"(?<![\w-]){n}(?![\w-])", text)]
        assert not leaked, f"{name} publishes real member name(s): {leaked}"


def test_whatsapp_server_no_longer_mounts_the_chat():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "mount_gradio_app"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("gradio", "src.chat.web_app"), node.module
            assert "gradio" not in names
    source = SERVER.read_text(encoding="utf-8")
    assert '"/app"' not in source
    assert "url=/app" not in source
    assert "_web_credentials" not in source


def test_whatsapp_server_serves_the_showcase_assets():
    source = SERVER.read_text(encoding="utf-8")
    assert '@app.get("/assets/{name}"' in source
    assert "static_assets.asset_path(" in source
