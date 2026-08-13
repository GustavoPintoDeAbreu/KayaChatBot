"""An edit must not be able to win by refusing to edit.

"Make these guys kissing" and "put a gun in the left guy's hand" both came back
pixel-identical to the photo they were given, were delivered as successes, and
when asked the bot said the request had been "blocked by content filters" —
which nothing in the pipeline knows.

The cause was the selection rule: pick_best returned the highest ArcFace
similarity to the source face, so among N takes the one that changed LEAST
always won. The bake-off had already fixed this for scoring (cde9829); the
production selector never got it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import face_utils

# Measured over the 40-cell standard grid: real Kontext edits score 0.048 (the
# weakest) to 0.44, an unchanged image at most 0.0003 through the delivery path.
THRESHOLD = 0.012


def solid(value, size=(64, 64)):
    from PIL import Image

    return Image.new("RGB", size, (value, value, value))


def half_and_half(left, right, size=(64, 64)):
    from PIL import Image

    image = Image.new("RGB", size, (left, left, left))
    image.paste(Image.new("RGB", (size[0] // 2, size[1]), (right, right, right)))
    return image


# ── the change metric ────────────────────────────────────────────────────────
def test_an_identical_image_scores_zero():
    source = solid(120)
    assert face_utils.change_ratio(source, source) == pytest.approx(0.0, abs=1e-6)


def test_jpeg_recompression_is_not_an_edit():
    """The delivery path re-encodes at quality 92; that must not read as change."""
    import io

    from PIL import Image

    source = half_and_half(30, 200)
    buffer = io.BytesIO()
    source.save(buffer, "JPEG", quality=92)
    with Image.open(io.BytesIO(buffer.getvalue())) as recompressed:
        assert face_utils.change_ratio(source, recompressed) < THRESHOLD


def test_a_real_change_scores_well_above_the_threshold():
    assert face_utils.change_ratio(solid(0), solid(255)) == pytest.approx(1.0, abs=1e-6)
    assert face_utils.change_ratio(solid(100), solid(140)) > THRESHOLD


def test_an_unscoreable_pair_is_not_called_a_no_op():
    """Failing to measure must not be reported as "nothing changed"."""
    assert face_utils.change_ratio(None, None) == 1.0


# ── the selection rule ───────────────────────────────────────────────────────
class FakeLikeness:
    """Stands in for ArcFace: the no-op scores highest, as it does in reality.

    Keyed by id() because PIL Images are not hashable.
    """

    def __init__(self, scores):
        self.scores = {id(image): score for image, score in scores}

    def __call__(self, reference, image):
        return self.scores.get(id(image))


def test_a_no_op_loses_to_a_real_edit(monkeypatch):
    """The whole bug in one assertion: the unchanged take scored best on
    likeness, so it was the one that got sent."""
    source = solid(120)
    untouched, edited = solid(120), solid(200)
    monkeypatch.setattr(face_utils, "likeness",
                        FakeLikeness([(untouched, 0.99), (edited, 0.62)]))

    best, seed, score = face_utils.pick_best(
        None, [(untouched, 1), (edited, 2)], source=source, noop_threshold=THRESHOLD)

    assert best is edited
    assert seed == 2


def test_the_most_alike_real_edit_still_wins(monkeypatch):
    """Disqualifying no-ops must not throw away the likeness contest itself."""
    source = solid(120)
    closer, further = solid(190), solid(240)
    monkeypatch.setattr(face_utils, "likeness",
                        FakeLikeness([(closer, 0.80), (further, 0.40)]))

    best, _, _ = face_utils.pick_best(
        None, [(further, 1), (closer, 2)], source=source, noop_threshold=THRESHOLD)

    assert best is closer


def test_all_no_ops_still_return_something(monkeypatch):
    """Nothing to choose between; the caller notices via change_ratio and says
    so rather than being handed no image at all."""
    source = solid(120)
    first, second = solid(120), solid(121)
    monkeypatch.setattr(face_utils, "likeness",
                        FakeLikeness([(first, 0.99), (second, 0.98)]))

    best, _, _ = face_utils.pick_best(
        None, [(first, 1), (second, 2)], source=source, noop_threshold=THRESHOLD)

    assert best is first
    assert face_utils.change_ratio(source, best) < THRESHOLD


def test_threshold_zero_keeps_the_old_behaviour(monkeypatch):
    source = solid(120)
    untouched, edited = solid(120), solid(200)
    monkeypatch.setattr(face_utils, "likeness",
                        FakeLikeness([(untouched, 0.99), (edited, 0.62)]))

    best, _, _ = face_utils.pick_best(
        None, [(untouched, 1), (edited, 2)], source=source, noop_threshold=0.0)

    assert best is untouched


def test_no_source_means_no_disqualification(monkeypatch):
    """The bench calls pick_best without a source; it must still work."""
    first, second = solid(120), solid(200)
    monkeypatch.setattr(face_utils, "likeness",
                        FakeLikeness([(first, 0.99), (second, 0.62)]))

    best, _, _ = face_utils.pick_best(None, [(first, 1), (second, 2)])

    assert best is first


def test_an_unscoreable_candidate_set_falls_back(monkeypatch):
    """A missing detector costs the choice, not the picture."""
    first, second = solid(120), solid(200)
    monkeypatch.setattr(face_utils, "likeness", FakeLikeness([]))

    best, seed, score = face_utils.pick_best(None, [(first, 1), (second, 2)])

    assert best is first and seed == 1 and score is None


def test_no_candidates_is_an_error():
    with pytest.raises(ValueError):
        face_utils.pick_best(None, [])
