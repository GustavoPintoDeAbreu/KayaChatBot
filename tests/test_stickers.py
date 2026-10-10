"""Stickers: the archive, frame extraction, and the review page's scoring."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.chat import stickers
from src.testing import review_page


def webp(frames=1):
    images = [Image.new("RGBA", (32, 32), (index * 60, 0, 0, 255)) for index in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="WEBP", save_all=frames > 1, append_images=images[1:],
                   duration=100)
    return buffer.getvalue()


def config_for(tmp_path):
    return {"chat": {"stickers": {"archive_dir": str(tmp_path / "stickers")}}}


def test_archive_writes_once_under_the_hash(tmp_path):
    config = config_for(tmp_path)
    sha = "pByTpQNIF_-kC8cfkac6LeApmUMguipO5sbUaBzVieM"
    path = stickers.archive(b"first", sha, config)
    assert path.read_bytes() == b"first"
    stickers.archive(b"second", sha, config)
    assert path.read_bytes() == b"first"
    assert stickers.is_archived(config, sha)


def test_archive_refuses_an_unsafe_hash(tmp_path):
    config = config_for(tmp_path)
    assert stickers.archive(b"x", "../../etc/passwd", config) is None
    assert stickers.archive(b"x", "", config) is None
    assert not (tmp_path / "stickers").exists()


def test_frames_static_and_animated():
    assert len(stickers.frames(webp(1))) == 1
    assert len(stickers.frames(webp(6))) == 3
    assert stickers.frames(b"not an image") == []


def test_review_scoring(tmp_path):
    (tmp_path / "items.json").write_text(json.dumps([
        {"id": "1", "candidates": [{"key": "A", "text": "a"}, {"key": "B", "text": "b"}]},
        {"id": "2", "candidates": [{"key": "A", "text": "a"}, {"key": "B", "text": "b"}]},
    ]))
    (tmp_path / "ratings.json").write_text(json.dumps({
        "1": {"A": {"rating": "right"}, "B": {"rating": "wrong", "invented": True}},
        "2": {"A": {"rating": "partly"}, "B": {"rating": "right"}},
    }))
    scores = review_page.score(tmp_path)
    assert scores["A"]["right_rate"] == 0.5 and scores["A"]["invented"] == 0
    assert scores["B"]["invented"] == 1 and scores["B"]["wrong"] == 1


def test_review_page_hides_the_variant_keys():
    page = review_page.render(
        [{"id": "1", "title": "t", "candidates": [{"key": "A", "text": "photo prompt"},
                                                   {"key": "C", "text": "context"}]}],
        {}, "Stickers", "rate")
    assert "photo prompt" in page and ">A<" not in page and ">C<" not in page
