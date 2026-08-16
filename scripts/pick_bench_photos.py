#!/usr/bin/env python3
"""Pick the input photos for the image-editing bake-off.

The whole bench turns on one question — does the edited photo still look like
the person? — so the inputs have to be photos where identity is unambiguous to
begin with. A blurry crowd shot cannot distinguish a model that preserves
likeness from one that invents a face.

Selection is by face geometry, not by eye: ArcFace detection over the export,
keeping shots with a large, confidently-detected, roughly frontal face. Each
kept photo also gets its reference identity embedding written alongside it —
that vector is what every candidate's output is scored against later.

    kaya_chatbot_env/bin/python scripts/pick_bench_photos.py \
        --media "$HOME/Downloads/WhatsApp kaya media" --count 8

Outputs land in data/bench_photos/ (gitignored — these are real faces).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

MIN_FACE_PX = 120        # a face smaller than this cannot be judged for likeness
MIN_DET_SCORE = 0.70
MAX_YAW_RATIO = 0.35     # nose offset within the face box; rejects hard profiles
MIN_OBJECT_PX = 512      # --no-faces has no face to bound quality, so bound size


def load_analyser(det_size: int = 640):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def frontality(face) -> float:
    """0 = dead centre, higher = more turned away.

    Uses the nose landmark's horizontal offset from the box centre, normalised by
    box width — enough to drop profiles without pulling in a pose estimator.
    """
    box = face.bbox
    width = max(float(box[2] - box[0]), 1.0)
    nose_x = float(face.kps[2][0])
    centre_x = (float(box[0]) + float(box[2])) / 2.0
    return abs(nose_x - centre_x) / width


def score_photo(app, path: Path) -> Dict[str, Any] | None:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    primary = faces[0]
    box = primary.bbox
    face_px = min(float(box[2] - box[0]), float(box[3] - box[1]))
    yaw = frontality(primary)

    if face_px < MIN_FACE_PX or float(primary.det_score) < MIN_DET_SCORE or yaw > MAX_YAW_RATIO:
        return None

    return {
        "file": path.name,
        "faces": len(faces),
        "face_px": round(face_px, 1),
        "det_score": round(float(primary.det_score), 3),
        "yaw": round(yaw, 3),
        "width": image.shape[1],
        "height": image.shape[0],
        "embedding": np.asarray(primary.normed_embedding, dtype=np.float32).tolist(),
    }


# "No face detected" is NOT "no people": the first pass of this selection picked
# eight photos that all turned out to be someone shot from behind, which is a
# person edit wearing a disguise and would score preservation as if it were a
# still life. The vision model has already described most of this export into
# data/media_descriptions.json, and a description is a far better witness than a
# face detector — it says "um homem", "uma pessoa de costas", "duas raparigas"
# whether or not a face is visible.
_PEOPLE_WORDS = re.compile(
    r"\b(pessoas?|homem|homens|mulher|mulheres|rapaz|rapazes|rapariga|raparigas|"
    r"jovem|jovens|crian[çc]as?|beb[ée]|senhor|senhora|casal|amigos?|grupo de|"
    r"indiv[íi]duo|figura humana|rosto|cara[s]?\b|selfie|retrato|"
    r"person|people|man|men|woman|women|boy|girl|child|children)\b",
    re.IGNORECASE,
)


def looks_peopleless(description: str) -> bool:
    """Whether a vision description describes a scene with nobody in it."""
    return bool(description) and not _PEOPLE_WORDS.search(description)


def score_object_photo(app, path: Path,
                       descriptions: Dict[str, str] | None = None) -> Dict[str, Any] | None:
    """The inverse selection: photos with NOBODY in them.

    The face grid is the reason an object edit was never measured. Every input was
    picked for having a large frontal face and every edit transformed a person, so
    the bench could not see what happened when the group sent a photo of four
    energy drinks on a shop counter — an identity clause about a face that was not
    there, a crop that framed nothing, and best-of-N ranked by a similarity score
    that did not exist.

    Two gates, because either alone is wrong: no detectable face, AND a vision
    description that does not mention a person. A photo with no description is
    skipped rather than guessed at — there are hundreds of candidates and only
    eight are needed, so the cheap thing to do with an unknown is drop it.
    """
    import cv2

    if descriptions is not None:
        description = descriptions.get(path.name, "")
        if not looks_peopleless(description):
            return None

    image = cv2.imread(str(path))
    if image is None:
        return None
    if app.get(image):
        return None
    height, width = image.shape[:2]
    if min(height, width) < MIN_OBJECT_PX:
        return None
    return {
        "file": path.name,
        "faces": 0,
        "width": width,
        "height": height,
        # Bigger is better here for the same reason a big face was: a thumbnail
        # cannot distinguish a model that preserved the scene from one that did not.
        "pixels": width * height,
        "description": (descriptions or {}).get(path.name, "")[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick bake-off input photos by face quality.")
    parser.add_argument("--media", required=True, help="WhatsApp media export directory")
    parser.add_argument("--out", default="")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--scan-limit", type=int, default=0, help="0 = scan every image")
    parser.add_argument("--no-faces", action="store_true",
                        help="pick photos with NOBODY in them, for the object grid")
    parser.add_argument("--descriptions", default="data/media_descriptions.json",
                        help="vision descriptions, used by --no-faces to reject "
                             "people the face detector cannot see")
    args = parser.parse_args()
    out_default = "data/bench_objects" if args.no_faces else "data/bench_photos"
    args.out = args.out or out_default

    base_dir = Path(__file__).parent.parent
    media = Path(args.media).expanduser()
    out_dir = (base_dir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The vision model has already read most of this export. For the object grid
    # its descriptions are the gate that a face detector cannot be: a person shot
    # from behind has no face and is still a person.
    descriptions: Dict[str, str] | None = None
    if args.no_faces:
        source = Path(args.descriptions)
        if not source.is_absolute():
            source = base_dir / args.descriptions
        if not source.exists():
            raise SystemExit(f"no descriptions at {source} — needed to tell a still "
                             f"life from someone photographed from behind")
        descriptions = json.loads(source.read_text(encoding="utf-8"))
        print(f"{len(descriptions)} vision descriptions from {source.name}")

    images = sorted(p for p in media.glob("IMG-*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.scan_limit:
        images = images[: args.scan_limit]
    print(f"scanning {len(images)} images from {media}")

    app = load_analyser()
    kept: List[Dict[str, Any]] = []
    for index, path in enumerate(images, 1):
        if index % 100 == 0:
            print(f"  {index}/{len(images)} scanned, {len(kept)} usable")
        try:
            result = (score_object_photo(app, path, descriptions) if args.no_faces
                      else score_photo(app, path))
        except Exception as exc:  # noqa: BLE001 — one bad JPEG must not end the scan
            print(f"  ! {path.name}: {exc}")
            continue
        if result:
            kept.append(result)

    bar = "have nobody in them" if args.no_faces else "passed the face-quality bar"
    print(f"{len(kept)} photos {bar}")
    if not kept:
        print("nothing usable — loosen the thresholds or check the media path")
        return

    if args.no_faces:
        # Biggest first, for the same reason the face grid takes the biggest
        # faces — but at most one per day. Sorting on size alone returned eight
        # frames of the same afternoon on the same boat, which measures one scene
        # eight times instead of eight scenes once.
        kept.sort(key=lambda r: r["pixels"], reverse=True)
        chosen, seen_days = [], set()
        for record in kept:
            day = record["file"][4:12]
            if day in seen_days:
                continue
            seen_days.add(day)
            chosen.append(record)
            if len(chosen) == args.count:
                break
        if len(chosen) < args.count:
            chosen += [r for r in kept if r not in chosen][: args.count - len(chosen)]
    else:
        # Prefer big, confident, frontal faces; then spread across solo and group
        # shots, since a group photo is a much harder edit than a portrait.
        kept.sort(key=lambda r: (r["face_px"] * r["det_score"], -r["yaw"]), reverse=True)
        solo = [r for r in kept if r["faces"] == 1]
        group = [r for r in kept if r["faces"] > 1]
        half = max(args.count // 2, 1)
        chosen = (solo[:half] + group[: args.count - half])[: args.count]
        if len(chosen) < args.count:
            extra = [r for r in kept if r not in chosen]
            chosen += extra[: args.count - len(chosen)]

    import shutil

    # Clear a previous selection: leftovers from an earlier run share the pNN_
    # numbering and would be silently scored as if they were part of this set.
    for stale in out_dir.iterdir():
        if stale.is_file():
            stale.unlink()

    manifest = []
    for position, record in enumerate(chosen, 1):
        source = media / record["file"]
        stem = f"p{position:02d}_{source.stem}"
        shutil.copy2(source, out_dir / f"{stem}{source.suffix.lower()}")
        # No reference embedding for an object photo: there is no identity to
        # score against, which is the whole point of that grid.
        if "embedding" in record:
            np.save(out_dir / f"{stem}.npy",
                    np.asarray(record["embedding"], dtype=np.float32))
        entry = {k: v for k, v in record.items() if k != "embedding"}
        entry["id"] = stem
        entry["path"] = str(out_dir / f"{stem}{source.suffix.lower()}")
        manifest.append(entry)
        if args.no_faces:
            print(f"  {stem}  {record['width']}x{record['height']}  no faces")
        else:
            print(f"  {stem}  faces={record['faces']}  face_px={record['face_px']}  "
                  f"det={record['det_score']}  yaw={record['yaw']}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    suffix = "" if args.no_faces else " + reference embeddings"
    print(f"\n✓ {len(manifest)} photos{suffix} in {out_dir}")


if __name__ == "__main__":
    main()
