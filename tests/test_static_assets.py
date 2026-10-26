"""The public showcase assets: one flat directory, plain names, nothing that
resolves outside it. Imported by the Pi gateway, so stdlib only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import static_assets


def test_a_plain_asset_name_resolves(tmp_path):
    (tmp_path / "demo.svg").write_text("<svg></svg>")
    path = static_assets.asset_path(tmp_path, "demo.svg")
    assert path == (tmp_path / "demo.svg").resolve()
    assert static_assets.media_type(path) == "image/svg+xml"


def test_every_allowed_type_has_its_media_type(tmp_path):
    for name, expected in (("a.png", "image/png"), ("a.webp", "image/webp"),
                           ("a.jpg", "image/jpeg"), ("a.js", "text/javascript")):
        (tmp_path / name).write_text("x")
        assert static_assets.media_type(static_assets.asset_path(tmp_path, name)) == expected


def test_names_that_are_not_plain_assets_are_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.svg").write_text("<svg></svg>")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.svg").write_text("<svg></svg>")
    for name in ("notes.txt", ".hidden.svg", "sub/inner.svg", "../demo.svg",
                 "Demo.svg", "demo.SVG", "-demo.svg", "", "demo.svg\x00",
                 "demo.mjs", "demo.js.map", "demo.JS"):
        assert static_assets.asset_path(tmp_path, name) is None, name


def test_a_missing_file_is_refused(tmp_path):
    assert static_assets.asset_path(tmp_path, "nope.svg") is None


def test_a_symlink_out_of_the_directory_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.svg").write_text("<svg></svg>")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "link.svg").symlink_to(outside / "secret.svg")
    assert static_assets.asset_path(assets, "link.svg") is None


def test_a_script_gets_its_own_headers(tmp_path):
    (tmp_path / "a.js").write_text("var one = 1;")
    headers = static_assets.headers_for(static_assets.asset_path(tmp_path, "a.js"))
    assert headers["Content-Security-Policy"] == "default-src 'none'"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "Cache-Control" in headers


def test_images_keep_their_headers_and_get_a_copy(tmp_path):
    (tmp_path / "demo.svg").write_text("<svg></svg>")
    path = static_assets.asset_path(tmp_path, "demo.svg")
    assert static_assets.headers_for(path) == static_assets.ASSET_HEADERS
    headers = static_assets.headers_for(path)
    headers["Cache-Control"] = "mutated"
    assert static_assets.ASSET_HEADERS["Cache-Control"] == "public, max-age=3600"
