#!/usr/bin/env python3
"""Render the landing page to PNGs, one per level/language, for sharing.

The landing page is one HTML file holding four documents behind two toggles
(simple/in-detail × EN/PT). WhatsApp cannot show a toggle, so each combination is
flattened into its own file and screenshotted.

    kaya_chatbot_env/bin/python scripts/render_landing_shots.py
    kaya_chatbot_env/bin/python scripts/render_landing_shots.py --only simple_en,advanced_en
    kaya_chatbot_env/bin/python scripts/render_landing_shots.py --out ~/Desktop/kaya-explainer

Three things are load-bearing, each learned by getting it wrong:

* **The toggle script must be overridden, not just the markup.** `apply()` runs on
  load and rewrites the wrapper from its own `state`, so setting `data-level` on
  `#page` alone is undone before the screenshot.
* **The switch bar is hidden with an inline `display:none`**, not the `hidden`
  attribute — `.switches{display:flex}` beats the UA stylesheet's `[hidden]`, so
  the chips stayed visible in the first batch that went out.
* **Chrome screenshots the window, not the content.** Rendering at a tall window
  leaves thousands of blank pixels, so each shot is cropped up from the bottom to
  the last row that differs from the page background.

Needs `google-chrome` on PATH and Pillow (both already present for the image
pipeline).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LANDING = BASE_DIR / "src" / "chat" / "static" / "landing.html"
DEFAULT_OUT = BASE_DIR / "reports" / "landing"

# (filename stem, level, language). `advanced` is the internal data-value; the
# button reads "In detail".
VARIANTS = [
    ("simple_en", "simple", "en"),
    ("simple_pt", "simple", "pt"),
    ("advanced_en", "advanced", "en"),
    ("advanced_pt", "advanced", "pt"),
]

WIDTH = 900           # CSS pixels; ×2 device scale = 1800px, sharp on a phone
SCALE = 2
TALL = 20000          # taller than any variant; cropped back down afterwards


def flatten(html: str, level: str, lang: str) -> str:
    """One document, no toggles, ready to screenshot."""
    html = html.replace(
        '<div class="wrap" id="page" data-level="simple" data-lang="en">',
        f'<div class="wrap" id="page" data-level="{level}" data-lang="{lang}">')
    html = re.sub(r"var state = \{[^}]*\}",
                  f"var state = {{ level: '{level}', lang: '{lang}' }}", html)
    html = html.replace('<div class="switches" role="group"',
                        '<div class="switches" style="display:none" role="group"')
    return html


def shoot(page: Path, out: Path) -> None:
    subprocess.run(
        ["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
         "--hide-scrollbars", "--virtual-time-budget=6000",
         f"--force-device-scale-factor={SCALE}",
         f"--window-size={WIDTH},{TALL}", f"--screenshot={out}", f"file://{page}"],
        check=True, capture_output=True,
    )


def crop_to_content(path: Path, pad: int = 120) -> tuple[int, int]:
    """Trim the blank tail Chrome leaves below the content. Returns (w, h)."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    background = image.getpixel((width - 3, height - 3))
    pixels = image.load()
    last = height - 1
    while last > 0:
        if not all(pixels[x, last] == background for x in range(0, width, 17)):
            break
        last -= 8
    bottom = min(height, last + pad)
    image.crop((0, 0, width, bottom)).save(path, optimize=True)
    return width, bottom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--only", default="", help="comma-separated stems to render")
    args = parser.parse_args()

    if shutil.which("google-chrome") is None:
        print("google-chrome not found on PATH", file=sys.stderr)
        return 1

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = LANDING.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        for stem, level, lang in VARIANTS:
            if wanted and stem not in wanted:
                continue
            page = Path(tmp) / f"{stem}.html"
            page.write_text(flatten(source, level, lang), encoding="utf-8")
            target = out_dir / f"{stem}.png"
            shoot(page, target)
            width, height = crop_to_content(target)
            size_mb = target.stat().st_size / 1e6
            print(f"{stem:12s} {width}x{height:<6d} {size_mb:.2f} MB  {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
