#!/usr/bin/env python3
"""Read-only check that the uv migration's external sources are really there.

The pip-to-uv change (idea/96) swaps install machinery, not pins — but every
pin must still resolve where the Dockerfiles now resolve it, and ghcr must
actually serve the pinned uv for both build archs (the PC image is amd64, the
Pi image is arm64). This reads the two requirements files with the same rule
tests/test_requirements_pins.py uses, then asks each source:

- every `==` pin: GET https://pypi.org/pypi/<name>/<version>/json (a wildcard
  pin like `fastapi==0.115.*` checks the project's release list instead);
- every `>=` floor: GET /pypi/<name>/json and compare with packaging.version;
- torch 2.6.0 on the cu124 index the Dockerfile still points at;
- ghcr's manifest for astral-sh/uv:0.12.19 (on 401, retry with a public token).

Prints per source how much was read and the first lines, so a green run shows
the sources answered rather than just a count. Exits non-zero if a reachable
source reports a missing pin or the uv manifest lacks an arm64 layer, or if no
source could be reached at all; a host that is simply unreachable (no network,
a proxy refusing it) is listed under NOT REACHED and does not by itself fail
the run — a refused host is not evidence a pin is missing. Nothing is
downloaded to disk and nothing is installed.

    kaya_chatbot_env/bin/python scripts/check_uv_pins.py
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from typing import Dict, List, Tuple


class _Unreachable(Exception):
    """The source could not be contacted at all (no network, proxy refused)."""


class _PinMissing(Exception):
    """A reachable source says a pin does not resolve where the build resolves it."""


try:
    from packaging.version import Version
except ImportError:
    # The venv pins packaging; a bare system python3 may not have it. Floors
    # here are plain numeric versions, so a tuple of parts is enough to order
    # them — this is a comparison fallback, not a PEP 440 parser.
    def Version(text: str) -> Tuple:
        return tuple(int(part) if part.isdigit() else part for part in text.split("."))

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILES = [
    ROOT / "requirements.txt",
    ROOT / "deploy/pi/requirements.txt",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
UV_TAG = "0.12.19"
UV_MANIFEST_URL = f"https://ghcr.io/v2/astral-sh/uv/manifests/{UV_TAG}"


def _fetch(url: str, headers: Dict[str, str] | None = None) -> Tuple[int, bytes]:
    """GET a URL and report (status, body); an HTTP status comes back as its
    code so callers can retry on 401 or call the pin missing, while a host
    that never answers raises _Unreachable."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        raise _Unreachable(f"{url}: {error}")


def _pins(path: Path) -> List[Tuple[str, str]]:
    """Pins from a requirements file: the same one-line rule as
    tests/test_requirements_pins.py:_requirements."""
    pins = []
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"(===|==|>=|<=|~=|!=)", line)
        if not match:
            continue
        name = line[:match.start()].strip()
        pins.append((name, line[match.start():].strip()))
    return pins


def _check_pypi_pin(name: str, specifier: str) -> str:
    """Resolve one pin against PyPI and report what was read."""
    operator = specifier[:3] if specifier.startswith("===") else specifier[:2]
    version = specifier[len(operator):]
    if operator == "==" and version.endswith(".*"):
        status, body = _fetch(f"https://pypi.org/pypi/{name}/json")
        if status != 200:
            raise _PinMissing(f"{name}{specifier}: PyPI answered {status}")
        releases = sorted(json.loads(body).get("releases", {}))
        prefix = version[:-2]
        matches = [release for release in releases if release.startswith(prefix)]
        if not matches:
            raise _PinMissing(f"{name}=={version}: no release matches the wildcard")
        return f"{name} =={version}: {len(releases)} releases read, {len(matches)} match ({matches[-1]} latest)"
    if operator in ("==", "==="):
        status, body = _fetch(f"https://pypi.org/pypi/{name}/{version}/json")
        if status != 200:
            raise _PinMissing(f"{name}{specifier}: PyPI answered {status}")
        info = json.loads(body)["info"]
        return f"{name} =={version}: {len(body)} bytes read, project {info['name']} at {info['version']}"
    status, body = _fetch(f"https://pypi.org/pypi/{name}/json")
    if status != 200:
        raise _PinMissing(f"{name}{specifier}: PyPI answered {status}")
    latest = Version(json.loads(body)["info"]["version"])
    floor = Version(version)
    if latest < floor:
        raise _PinMissing(f"{name}{operator}{version}: PyPI latest is {latest} below the floor")
    return f"{name} {operator}{version}: latest {latest} satisfies the floor"


def _check_torch_index() -> str:
    """Count torch-2.6.0 links on the cu124 index page the Dockerfile uses."""
    status, body = _fetch(TORCH_INDEX_URL)
    if status != 200:
        raise _PinMissing(f"torch==2.6.0 on {TORCH_INDEX_URL}: answered {status}")
    text = body.decode("utf-8", errors="replace")
    hits = text.count("torch-2.6.0")
    if hits == 0:
        raise _PinMissing(f"torch==2.6.0 not on {TORCH_INDEX_URL}")
    first_lines = [line.strip() for line in text.splitlines() if "torch-2.6.0" in line][:3]
    return f"{TORCH_INDEX_URL}: {len(body)} bytes read, {hits} torch-2.6.0 links ({first_lines[0]})"


def _check_uv_manifest() -> str:
    """ghcr manifest for astral-sh/uv:{tag}: must carry amd64 AND arm64 layers."""
    headers = {"Accept": "application/vnd.oci.image.manifest.v1+json"}
    status, body = _fetch(UV_MANIFEST_URL, headers)
    if status == 401:
        token_status, token_body = _fetch("https://ghcr.io/token?scope=public")
        if token_status != 200:
            raise _PinMissing(f"uv:{UV_TAG} on ghcr: token endpoint answered {token_status}")
        headers["Authorization"] = f"Bearer {json.loads(token_body)['token']}"
        status, body = _fetch(UV_MANIFEST_URL, headers)
    if status != 200:
        raise _PinMissing(f"uv:{UV_TAG} on ghcr: answered {status}")
    manifest = json.loads(body)
    layers = manifest.get("images") or manifest.get("manifests") or []
    architectures = sorted({layer.get("architecture", "?") for layer in layers})
    if "arm64" not in architectures:
        raise _PinMissing(f"arm64 layer missing for ghcr astral-sh/uv:{UV_TAG}: {architectures}")
    return f"ghcr astral-sh/uv:{UV_TAG}: {len(body)} bytes read, layers {architectures} ({UV_MANIFEST_URL})"


def main() -> int:
    report = []
    missing = []
    unreached = []
    reached = 0

    def ask(label: str, check) -> None:
        nonlocal reached
        try:
            report.append(check())
            reached += 1
        except _Unreachable as error:
            unreached.append(f"{label}: {error}")
        except _PinMissing as error:
            missing.append(f"{label}: {error}")

    for path in REQUIREMENTS_FILES:
        pins = _pins(path)
        report.append(f"{path.relative_to(ROOT)}: {len(pins)} pins")
        for name, specifier in pins:
            ask(f"{name}{specifier}", lambda name=name, specifier=specifier: _check_pypi_pin(name, specifier))
    ask("torch cu124 index", _check_torch_index)
    ask("ghcr uv manifest", _check_uv_manifest)

    print("\n".join(report))
    if unreached:
        print(f"\nNOT REACHED ({len(unreached)} sources):")
        print("\n".join(unreached))
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        print("\n".join(missing))
        return 1
    if not reached:
        print("\nNOTHING VERIFIED: no source could be reached.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
