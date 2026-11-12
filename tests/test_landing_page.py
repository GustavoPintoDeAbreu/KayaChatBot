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
GITHUB = "https://github.com/GustavoPintoDeAbreu"

SCRIPTS = ("scene.js", "motion.js", "metrics.js")


def _css(landing: str) -> str:
    """The page's style block."""
    return re.search(r"(?is)<style>(.*?)</style>", landing).group(1)


def _script(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")

SHOWN = ("kaya", "lab-assistant", "idea-pipeline", "lab-medic", "labwatch", "lab-notes",
         "redditcast", "series-watch", "flight-deals", "price-scout", "game-alerts", "subs",
         "paywatch", "personal-hub", "lab-browser")
HIDDEN = ("lab-audit", "lab audit", "lab-infra", "house-hunt", "house hunt", "daily-git",
          "daily git", "photoshop", "comfyui")
# Model and framework names (Gemma, Qwen, FastAPI, SQLite...) may be named since 2026-10-27
# (Gustavo: the detail cards are for talking shop); hardware, addresses, versions, model
# sizes and the lab's internal services still may not.
INFRA_TERMS = ("gradio", "waha", "noweb",
               "broker", "grok", "xai", "azure", "openai", "cloudflare", "tunnel",
               "tailscale", "pi5", "rtx", "3090", "nvidia", "ryzen", "gpu",
               "localhost", ".local", "wake-on-lan", "shutdown", "firewall", "trivy", "gitleaks",
               "osv-scanner", "lynis", "rkhunter", "chkrootkit", "rustdesk", "ssh", "sudo",
               "proxy", "serpapi", "evomi", "tmdb", "/idea", "/homelab", "/status",
               "/app", "07:00", "kaya-prod", "sigmakayachat")
INFRA_PATTERNS = (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", r"\b\d+\.\d+\.\d+\b", r"\bv\d+(?:\.\d+)+\b",
                  r"\b\d{1,3}[bB]\b", r"\+\d{6,}", r"\b\d{9,}\b", r"[\w.+-]+@[\w-]+\.\w+",
                  r"\bLAN\b", r"\bports?\b")

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

# The page is the owner's own site, so it credits its author by real name in the byline, meta
# description and footer. That credit is intentional and public (it links to his LinkedIn/GitHub),
# not a group-member leak, so the guard below strips these exact author forms before matching.
AUTHOR_CREDIT = ("Gustavo Abreu", "GustavoPintoDeAbreu", "gustavo-abreu")


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


def test_the_glance_card_counts_the_projects_the_page_shows(landing):
    """2026-11-12: the card said 12 while the page showed 15 project cards. The number is
    derived from the page's own project list now, so a new card without a new count fails."""
    shown = set(re.findall(r'data-project="([a-z-]+)"', landing))
    assert len(shown) >= 12
    assert f"<b>{len(shown)} projects</b>" in landing
    assert f"<b>{len(shown)} projetos</b>" in landing


def test_hidden_projects_are_not_mentioned():
    for name, text in _public_texts():
        lowered = text.lower()
        leaked = [term for term in HIDDEN if term in lowered]
        assert not leaked, f"{name} mentions hidden project(s): {leaked}"


def test_no_infrastructure_details_are_published():
    for name, text in _public_texts():
        body = _without_code(text).replace(LINKEDIN, "").replace(GITHUB, "")
        # The page references its own images by path (e.g. /assets/idea-pipeline.svg);
        # those filenames are not infrastructure, so drop them before the checks.
        body = re.sub(r'src="/assets/[^"]+"', " ", body)
        lowered = body.lower()
        leaked = [term for term in INFRA_TERMS if term in lowered]
        assert not leaked, f"{name} publishes infrastructure detail(s): {leaked}"
        for pattern in INFRA_PATTERNS:
            match = re.search(pattern, body)
            assert match is None, f"{name} matches {pattern!r}: {match.group(0)!r}"


def test_only_linkedin_and_github_are_external(landing):
    hrefs = re.findall(r'href="(https?://[^"]+)"', landing)
    assert hrefs, "the page must link somewhere"
    assert all(href in (LINKEDIN, GITHUB) for href in hrefs), hrefs
    for profile in (LINKEDIN, GITHUB):
        total = landing.count(f'href="{profile}"')
        safe = landing.count(f'href="{profile}" target="_blank" rel="noopener noreferrer"')
        assert total == safe
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


def test_the_idea_prompt_is_not_baked_into_the_page(landing):
    """The labwatch /idea prompt is server-injected (owner-approved), never static: a reviewer
    reading the page's source must be able to tell real feed data from an error."""
    assert "Build me a self-hosted dashboard" not in landing


def test_no_real_member_names_are_published():
    for name, text in _public_texts():
        body = text
        for author in AUTHOR_CREDIT:
            body = body.replace(author, "")
        leaked = [n for n in REAL_NAMES if re.search(rf"(?<![\w-]){n}(?![\w-])", body)]
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


def test_only_the_pi_gateway_serves_the_showcase():
    """2026-11-02: the PC app's copy of "/" and "/assets/<name>" was unreachable from the
    internet (Cloudflare routes everything to the Pi) and nothing on the LAN used it."""
    source = SERVER.read_text(encoding="utf-8")
    assert '@app.get("/"' not in source and '"/assets/{name}"' not in source
    assert "showcase_metrics" not in source and "landing.html" not in source
    gateway = (BASE_DIR / "src" / "gateway" / "app.py").read_text(encoding="utf-8")
    assert '@app.get("/assets/{name}")' in gateway and "static_assets.asset_path(" in gateway


def test_scripts_are_local_assets_that_exist(landing):
    srcs = re.findall(r'<script[^>]*\ssrc="([^"]+)"[^>]*>', landing)
    assert srcs == ["/assets/scene.js", "/assets/motion.js", "/assets/metrics.js"]
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


CONTACT_EN_BULLETS = ("Website full-stack development", "Foundational data architecture for companies",
                      "Hardware &amp; IoT project integration", "Everything multilingual and cross-platform when needed")
CONTACT_PT_BULLETS = ("Desenvolvimento web full-stack", "Arquitetura de data fundacional para empresas",
                      "Integração de projetos de hardware e IoT", "Tudo multilingue e multiplatform quando eu quiser")


def test_how_i_build_heading_names_the_motivation(landing):
    """2026-11-08: Gustavo asked for the section to say what motivates it, not just how."""
    for heading in ("How I build and what motivates me", "Como construo e o que me motiva"):
        assert f'<h2 data-scrub="slide">{heading}</h2>' in landing, heading


def test_get_in_touch_is_a_plain_form_posting_to_the_gateway(landing):
    """No JS on the page submits it (the CSP has no fetch and the scripts stay inert), so the
    submit is a form whose action is this origin's /contact — never an external endpoint."""
    for heading in ("Get in touch", "Fala comigo"):
        assert f'<h2 data-scrub="slide">{heading}</h2>' in landing, heading
    assert landing.count('<form class="contact-form" action="/contact" method="post">') == 2
    assert not re.search(r"<form[^>]*action=\"https?://", landing)
    for field in ("name", "email", "message"):
        assert landing.count(f'name="{field}"') == 2, field
    assert landing.count('<input class="hp" name="website"') == 2
    assert landing.count('name="proof"') == 2
    assert landing.count('value="PROOF-TOKEN"') == 2, "the gateway stamps a token here on every serve"
    assert "<button type=\"submit\">Send</button>" in landing
    assert "<button type=\"submit\">Enviar</button>" in landing


def test_the_contact_offer_box_lists_what_he_can_do(landing):
    for label in ("I can help with", "Posso ajuda com"):
        assert f"<b>{label}</b>" in landing, label
    for bullet in CONTACT_EN_BULLETS + CONTACT_PT_BULLETS:
        assert f"<li>{bullet}</li>" in landing, bullet


def test_the_contact_form_carries_no_owner_email(landing):
    """The form takes the visitor's email; the owner's stays server-side (CONTACT_EMAIL_TO)."""
    assert "mailto:" not in landing
    assert re.search(r"[\w.+-]+@[\w-]+\.\w+", _without_code(landing)) is None


def test_console_easter_egg():
    assert "console.log('%cKaya'" in _script("motion.js")


def test_scripts_parse():
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this machine to syntax-check the scripts")
    for name in SCRIPTS:
        result = subprocess.run([node, "--check", str(ASSETS / name)], capture_output=True)
        assert result.returncode == 0, (name, result.stderr.decode())


def test_one_colour_scheme_and_no_switch(landing):
    """2026-11-02: only the blue scheme stays; the 1-4 switch went with the other three."""
    assert "data-scheme" not in landing
    assert 'data-set="scheme"' not in landing
    assert "kaya:scheme" not in landing
    for name in SCRIPTS:
        assert "kaya:scheme" not in _script(name), name
    assert "--teal:#3edbe3" in _css(landing)


def test_clicking_a_card_opens_nothing():
    """2026-11-02: the full-screen detail panel is gone; the hover overlay is what stays."""
    for name in SCRIPTS:
        text = _script(name)
        assert "detail-panel" not in text, name
        assert "closest('.project')" not in text, name


def test_project_rows_slide_to_their_own_metrics():
    text = _script("metrics.js")
    assert "deck-toggle" in text and "deck-track" in text
    assert "translate3d(-50%,0,0)" in text
    assert "inert" in text
    assert "data.lab" not in text
    assert "products" in text and "flows" in text


def test_every_project_has_details_in_both_languages(landing):
    for project in SHOWN:
        assert landing.count(f'data-detail="{project}"') == 2, project
    for label in ("How it works", "Built with", "Como funciona", "Feito com"):
        assert landing.count(label) == len(SHOWN), label


def test_how_it_fits_names_the_orchestrator_and_two_outcomes(landing):
    assert landing.count('class="outcome outcome-product"') == 2
    assert "the orchestrator" in landing and "o orquestrador" in landing
    for label in ("App usage", "Development", "Usar as apps", "Desenvolvimento"):
        assert label in landing, label
    assert "Notes and ideas take their shape" not in landing
    assert "As notas e as ideias ganham forma" not in landing


def test_the_wordmark_is_a_sigma_not_a_k(landing):
    assert "M14 14H44V70L100 14H146L88 90" not in landing
    assert "M14 32H146M8 90H146M14 148H146M14 32L8 90L14 148" not in landing
    wordmarks = re.findall(r'<svg class="wordmark".*?</svg>', landing, flags=re.S)
    assert len(wordmarks) == 2
    outer = "M146 46L146 32L102 32L58 32L14 32L49 61L84 90L49 119L14 148L58 148L102 148L146 148L146 134"
    inner = "M120 44L80 44L40 44L60 90L40 136L80 136L120 136"
    for mark in wordmarks:
        assert mark.count(f'd="{outer}"') == 1
        assert mark.count(f'd="{inner}"') == 1
        nets = re.findall(r'<g class="km-net".*?</g>', mark, flags=re.S)
        assert len(nets) == 1
        assert nets[0].count("<path d=") == 3
        assert mark.count('class="km-node"') == 17
        assert mark.count('class="km-dot"') == 3
        assert mark.count('r="3.5"') == 16
        assert mark.count('r="4.5"') == 1
        assert mark.count('r="6"') == 3
        assert mark.count('r="16"') == 3


def test_group_titles(landing):
    for title in ("Interactive agents", "Builder agents", "Scout agents", "Side projects",
                  "Agentes interativos", "Agentes construtores", "Agentes batedores", "Projetos paralelos"):
        assert f'<p class="group-title">{title}</p>' in landing, title


def test_the_wordmark_is_drawn_per_language(landing):
    # gradient ids are per copy, so the hidden language never steals the visible one's
    for lang in ("en", "pt"):
        for name in ("kg", "wg", "glow"):
            assert landing.count(f'id="{name}-{lang}"') == 1

