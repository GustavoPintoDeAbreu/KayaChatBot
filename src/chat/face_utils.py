"""Face detection for image editing: framing the shot, and picking the best take.

Two problems, one detector.

**Framing.** Kontext works at about one megapixel. A photo is scaled to fit that
before the transformer ever sees it, so a face keeps only the share of the frame
it already had: a 4000px group shot with a 400px face arrives with a face about
100px across, and a hundred pixels do not carry a person's identity. The
bake-off shows exactly this — likeness held at 0.55-0.67 on the solo portraits
and collapsed to 0.03-0.18 on the group shots. Tightening the crop around the
faces before the resize is the cheapest way to spend those pixels on the part
that has to survive.

**Choosing.** Editing is stochastic and the failures are not subtle: the same
prompt returns a good likeness on one seed and a stranger on the next. With a
detector already loaded, N candidates can be scored against the source face and
the best one sent, instead of sending whichever seed came up.

Everything here degrades to a no-op. InsightFace is CPU-only and optional; if it
is missing or finds nothing, the image is used exactly as it arrives, which is
the behaviour this module replaced.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The same pack scripts/pick_bench_photos.py scores the reference photos with, so
# a likeness measured at runtime means the same thing as one from the bake-off.
MODEL_PACK = "buffalo_l"

_analyser = None
_analyser_lock = threading.Lock()
_unavailable = False


def get_analyser():
    """The shared InsightFace analyser, or None if it cannot be loaded.

    CPU only: the card is busy holding a 20GB diffusion pipeline, and detection
    on one photo is well under a second.
    """
    global _analyser, _unavailable
    if _analyser is not None or _unavailable:
        return _analyser
    with _analyser_lock:
        if _analyser is not None or _unavailable:
            return _analyser
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=MODEL_PACK, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _analyser = app
        except Exception as exc:  # noqa: BLE001 — a missing detector must not stop an edit
            logger.warning("face detection unavailable (%s); continuing without it", exc)
            _unavailable = True
    return _analyser


def detect(image) -> List[Any]:
    """Faces in a PIL image, largest first. Empty on any failure."""
    app = get_analyser()
    if app is None:
        return []
    try:
        import numpy as np

        frame = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PIL RGB → cv2 BGR
        faces = app.get(frame)
    except Exception as exc:  # noqa: BLE001
        logger.warning("face detection failed (%s)", exc)
        return []
    faces.sort(key=lambda f: _area(f.bbox), reverse=True)
    return faces


def _area(bbox) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def reference_embedding(image) -> Optional[Any]:
    """The identity to preserve: the largest face in the source photo."""
    faces = detect(image)
    if not faces:
        return None
    import numpy as np

    return np.asarray(faces[0].normed_embedding, dtype="float32")


def likeness(reference, image) -> Optional[float]:
    """Cosine similarity between ``reference`` and the closest face in ``image``.

    Matches the *same person*, not simply the biggest face in the output. On a
    group shot those are frequently different people, which makes the number
    measure framing rather than identity.
    """
    if reference is None:
        return None
    faces = detect(image)
    if not faces:
        return None
    import numpy as np

    scores = [
        float(np.dot(reference, np.asarray(f.normed_embedding, dtype="float32")))
        for f in faces
    ]
    return max(scores)


def frame_for_faces(image, min_ratio: float = 0.22, target_ratio: float = 0.32):
    """Crop towards the faces when they are too small to survive the resize.

    Returns ``(image, was_cropped)``. The crop keeps the original aspect ratio,
    so the picture is tightened rather than reshaped and the Kontext bucket it
    lands in does not change. It always contains every detected face — cropping
    to one person in a group photo would answer a different request.
    """
    faces = detect(image)
    if not faces:
        return image, False

    width, height = image.size
    largest = max(faces, key=lambda f: _area(f.bbox))
    face_width = float(largest.bbox[2] - largest.bbox[0])
    if face_width <= 0 or face_width / width >= min_ratio:
        return image, False

    left = min(float(f.bbox[0]) for f in faces)
    top = min(float(f.bbox[1]) for f in faces)
    right = max(float(f.bbox[2]) for f in faces)
    bottom = max(float(f.bbox[3]) for f in faces)

    # Wide enough to give the face its share of the frame, and never tighter than
    # the faces themselves plus a little air.
    wanted = max(face_width / target_ratio, (right - left) * 1.25)
    crop_width = min(float(width), wanted)
    crop_height = min(float(height), crop_width * height / width)
    # Clamping the height may have forced a narrower box; keep the ratio honest.
    crop_width = min(crop_width, crop_height * width / height)
    if crop_width >= width * 0.98:
        return image, False

    centre_x = (left + right) / 2
    centre_y = (top + bottom) / 2
    box_left = _clamp(centre_x - crop_width / 2, 0, width - crop_width)
    box_top = _clamp(centre_y - crop_height / 2, 0, height - crop_height)
    cropped = image.crop((
        int(box_left), int(box_top),
        int(box_left + crop_width), int(box_top + crop_height),
    ))
    logger.info(
        "framed for faces: %dx%d → %dx%d (face %.0f%% → %.0f%% of width)",
        width, height, cropped.width, cropped.height,
        100 * face_width / width, 100 * face_width / cropped.width,
    )
    return cropped, True


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, max(low, high)))


def fit_to_kontext(image):
    """Resize to the nearest resolution Kontext was trained on, in ONE resample.

    The pipeline snaps its input to a bucket from ``PREFERRED_KONTEXT_RESOLUTIONS``
    by aspect ratio regardless. Pre-scaling to 1024 first meant every photo was
    resampled twice, spending detail before the VAE ever saw the face, so the
    bucket is picked here and the pipeline is told not to do it again.
    """
    from PIL import Image
    from diffusers.pipelines.flux.pipeline_flux_kontext import (
        PREFERRED_KONTEXT_RESOLUTIONS)

    aspect = image.width / image.height
    _, width, height = min(
        (abs(aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    if (image.width, image.height) == (width, height):
        return image
    return image.resize((width, height), Image.LANCZOS)


def prepare_source(path: str, face_crop: bool = True, min_ratio: float = 0.22,
                   target_ratio: float = 0.32):
    """Open a photo, tighten it around the faces if they are small, bucket it.

    Returns ``(image_for_the_model, reference_embedding)``. The embedding comes
    from the FULL-resolution original, where the face has the most pixels, so it
    is the strongest available statement of who has to come back out.
    """
    from PIL import Image, ImageOps

    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    reference = reference_embedding(original)
    framed = original
    if face_crop:
        framed, _ = frame_for_faces(
            original, min_ratio=min_ratio, target_ratio=target_ratio)
    return fit_to_kontext(framed), reference


def pick_best(reference, candidates: Sequence[Tuple[Any, int]]):
    """Return ``(image, seed, score)`` for the candidate closest to ``reference``.

    Falls back to the first candidate whenever nothing can be scored, so a missing
    detector costs the choice, not the picture.
    """
    if not candidates:
        raise ValueError("no candidates to choose from")
    scored = []
    for image, seed in candidates:
        score = likeness(reference, image)
        scored.append((image, seed, score))
    usable = [row for row in scored if row[2] is not None]
    if not usable:
        image, seed, _ = scored[0]
        return image, seed, None
    return max(usable, key=lambda row: row[2])
