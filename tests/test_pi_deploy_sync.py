"""deploy_pi.sh syncs only an allow-list to the Pi; every path the Pi Dockerfile COPYs must be on it.

2026-11-28: #200 added `COPY lab.yaml` (the Pi /help tap) but the rsync list had no lab.yaml, so the
gateway build failed on the Pi ("/lab.yaml": not found) while CI, which builds from the full repo, was green.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copied_sources():
    out = []
    for line in (ROOT / "deploy/pi/Dockerfile").read_text().splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(.+)", line)
        if m:
            parts = m.group(1).rstrip("\\").split()
            out += [p for p in parts[:-1] if not p.startswith("--")]
    return out


def _covered(path, includes):
    for inc in includes:
        if inc.endswith("/***") and (path + "/").startswith(inc[:-3]):
            return True
        if inc == path:
            return True
    return False


def test_every_pi_dockerfile_copy_is_synced():
    script = (ROOT / "scripts/deploy_pi.sh").read_text()
    includes = re.findall(r"--include '([^']+)'", script)
    sources = _copied_sources()
    assert "lab.yaml" in sources and "config.yaml" in sources
    missing = [s for s in sources if not _covered(s.rstrip("/"), includes)]
    assert missing == [], f"deploy_pi.sh does not sync {missing} to the Pi"
