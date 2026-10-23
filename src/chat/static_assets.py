"""Images for the public showcase page: one flat directory, a fixed set of types.

Stdlib only: the Pi gateway imports it (tests/gateway/test_torch_free.py, test_pi_image.py)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Union

ASSET_TYPES: Dict[str, str] = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
}
ASSET_HEADERS: Dict[str, str] = {
    "Cache-Control": "public, max-age=3600",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
}
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.(?:svg|png|webp|jpg)")


def asset_path(directory: Union[str, Path], name: str) -> Optional[Path]:
    """The file ``name`` directly inside ``directory``, or None if the name is not a plain
    lowercase asset name, the file is missing, or it resolves anywhere else."""
    if not _NAME.fullmatch(name):
        return None
    base = Path(directory).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        return None
    return path


def media_type(path: Path) -> str:
    """The Content-Type for an asset returned by asset_path."""
    return ASSET_TYPES[path.suffix]
