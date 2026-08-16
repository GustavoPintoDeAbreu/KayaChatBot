"""The public landing page, and the password that must sit in front of the chat.

`whatsapp_server` cannot be imported here — it builds the engine at module scope
(`engine = get_engine(config)`), which loads the model. The mount is therefore
checked by parsing the source, which is enough to catch the regression that
actually happened: `mount_gradio_app` was called without `auth=` for as long as
it existed, so KAYA_WEB_USER/KAYA_WEB_PASS were set in the deployed environment
and silently ignored, leaving Cloudflare Access as the only thing in front of the
group's private memory.
"""
import ast
import re
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent
LANDING = BASE_DIR / "src" / "chat" / "static" / "landing.html"
SERVER = BASE_DIR / "src" / "chat" / "whatsapp_server.py"

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


def test_landing_page_exists(landing):
    assert len(landing) > 10_000, "landing page looks truncated"


def test_login_button_points_at_the_app(landing):
    assert 'href="/app"' in landing, "the landing page must offer a way in"


def test_the_toggle_says_in_detail(landing):
    assert ">In detail<" in landing
    assert ">Advanced<" not in landing


def test_both_languages_and_both_levels_are_present(landing):
    for level in ("simple", "advanced"):
        for lang in ("en", "pt"):
            assert f'class="doc" data-level="{level}" data-lang="{lang}"' in landing


def test_no_real_member_names_are_published(landing):
    leaked = [n for n in REAL_NAMES if re.search(rf"(?<![\w-]){n}(?![\w-])", landing)]
    assert not leaked, f"real member name(s) on the public page: {leaked}"


def test_the_report_commands_are_documented(landing):
    """The collection channel is useless if nobody is told it exists."""
    for command in ("/bug", "/feedback"):
        assert landing.count(f"<code>{command}</code>") >= 2, (
            f"{command} must appear in both the English and Portuguese pages")


def test_page_is_self_contained(landing):
    """No external requests: the page must render with the network off."""
    assert not re.search(r'(?:src|href)="https?://', landing)


def _mount_call() -> ast.Call:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mount_gradio_app"):
            return node
    raise AssertionError("mount_gradio_app call not found in whatsapp_server")


def test_the_chat_is_mounted_behind_auth():
    """The argument whose absence made production unauthenticated."""
    kwargs = {kw.arg for kw in _mount_call().keywords}
    assert "auth" in kwargs, "mount_gradio_app must be given auth="


def test_the_chat_is_not_mounted_at_the_root():
    """`/` belongs to the public landing page; the chat lives under /app."""
    path = next(kw.value for kw in _mount_call().keywords if kw.arg == "path")
    assert path.value == "/app"
