"""The Pi image copies files one by one, so every module the gateway imports must be listed.

On 2026-10-10 the gateway started importing src/chat/channels.py (through
whatsapp_adapter) and src/chat/birthdays.py; neither was in deploy/pi/Dockerfile,
and the deployed gateway crash-looped with ModuleNotFoundError until it was fixed.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _copied() -> set:
    text = (ROOT / "deploy/pi/Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    paths = set()
    for line in text.splitlines():
        if line.startswith("COPY "):
            paths.update(part for part in line.split()[1:-1] if not part.startswith("--"))
    return paths


def _local_imports(path: Path) -> set:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.startswith("src."):
                found.add(name)
    return found


def _module_file(name: str) -> Path:
    path = ROOT / Path(*name.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def test_every_module_the_gateway_imports_is_in_the_pi_image():
    copied = _copied()
    seen, queue = set(), [p for p in (ROOT / "src/gateway").glob("*.py")]
    missing = []
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        relative = str(path.relative_to(ROOT))
        if not any(relative == c or relative.startswith(c.rstrip("/") + "/") for c in copied):
            missing.append(relative)
        queue.extend(_module_file(name) for name in _local_imports(path))
    missing = [m for m in missing if not re.fullmatch(r"src(/chat|/data)?/__init__\.py", m)]
    assert not missing, f"add to deploy/pi/Dockerfile: {missing}"
