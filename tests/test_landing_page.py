"""The public showcase page: what it may say, and what it must not.

The page is public — it is handed to the group and can be forwarded anywhere —
so it carries no real member names, no infrastructure details and no login.
`whatsapp_server` cannot be imported here — it builds the engine at module
scope (`engine = get_engine(config)`), which loads the model — so the routes
are checked by parsing the source.
"""
import shutil
import ast
import re
import subprocess
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

SCRIPTS = ("scene.js", "motion.js")


def _css(landing: str) -> str:
    """The page's style block."""
    return re.search(r"(?is)<style>(.*?)</style>", landing).group(1)


def _script(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")

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
    on_disk = {p.name for p in ASSETS.iterdir() if p.suffix != ".js"}
    assert referenced == on_disk, (on_disk - referenced, referenced - on_disk)
    alts = re.findall(r'<img[^>]+alt="([^"]*)"', landing)
    assert all(alt for alt in alts)


def test_assets_are_inert_and_small():
    for path in sorted(ASSETS.iterdir()):
        if path.suffix == ".js":  # scripts have their own tests
            continue
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
    assert "static_assets.headers_for(path)" in source


def test_scripts_are_local_assets_that_exist(landing):
    srcs = re.findall(r'<script[^>]*\ssrc="([^"]+)"[^>]*>', landing)
    assert srcs == ["/assets/scene.js", "/assets/motion.js"]
    for src in srcs:
        assert static_assets.asset_path(ASSETS, src[len("/assets/"):]) is not None, src
    assert {p.name for p in ASSETS.iterdir() if p.suffix == ".js"} == set(SCRIPTS)
    for name in SCRIPTS:
        tag = next(line for line in landing.splitlines() if f'src="/assets/{name}"' in line)
        assert " defer" in tag, name


def test_scripts_stay_on_the_page():
    banned = ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "EventSource",
              "import(", "eval(", "new Function", "innerHTML", "outerHTML",
              "insertAdjacentHTML", "document.write", "http://", "https://",
              "localStorage", "document.cookie")
    for name in SCRIPTS:
        text = _script(name)
        assert len(text.encode("utf-8")) < 40_000, name
        for word in banned:
            assert word not in text, (name, word)


def test_motion_is_optional(landing):
    css = _css(landing)
    assert "prefers-reduced-motion" in css
    guard = "matchMedia('(prefers-reduced-motion: reduce)')"
    assert guard in landing
    for name in SCRIPTS:
        assert guard in _script(name)
    assert 'class="stage-fallback"' in landing
    assert "<canvas" in landing
    assert "scene-on" in _script("scene.js")
    assert "scene-on" in css


def test_no_shadows_or_blurs(landing):
    css = _css(landing).lower()
    for word in ("shadow", "blur", "filter"):
        assert word not in css, word
        for name in SCRIPTS:
            assert word not in _script(name).lower(), (name, word)


def test_animations_use_transforms_only(landing):
    css = _css(landing)
    for value in re.findall(r"transition:([^;}]+)", css):
        stripped = re.sub(r"\([^)]*\)", "", value)
        for part in stripped.split(","):
            part = part.strip()
            assert part == "none" or part.startswith("transform"), (css, part)
    assert css.count("@keyframes") == 1
    assert "@keyframes curtain-lift{to{transform:translate3d(0,-101%,0)}}" in css
    for name in SCRIPTS:
        for style in re.findall(r"\.style\.(\w+)", _script(name)):
            assert style in {"transform", "setProperty", "width", "height"}, (name, style)


def test_curtain_never_blocks_the_page(landing):
    css = _css(landing)
    assert re.search(r"\.curtain\{[^}]*pointer-events:none", css)
    curtain_tag = re.search(r"<div id=\"curtain\"[^>]*>", landing)
    assert curtain_tag is not None
    assert 'id="curtain"' in curtain_tag.group(0)
    assert 'aria-hidden="true"' in curtain_tag.group(0)
    assert ".curtain{display:none}" in css


def test_the_language_switch_tells_the_scene(landing):
    assert "kaya:lang" in landing
    for name in SCRIPTS:
        assert "kaya:lang" in _script(name)


def test_console_easter_egg():
    assert "console.log('%cKaya'" in _script("motion.js")


def test_scripts_parse():
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this machine to syntax-check the scripts")
    for name in SCRIPTS:
        result = subprocess.run([node, "--check", str(ASSETS / name)], capture_output=True)
        assert result.returncode == 0, (name, result.stderr.decode())
