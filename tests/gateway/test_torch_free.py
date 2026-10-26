"""The Pi cannot run the model stack; the gateway must never import it."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent


def test_the_gateway_imports_nothing_heavy():
    code = ("import sys, src.gateway.app; "
            "heavy = [m for m in ('torch', 'sentence_transformers', 'chromadb', 'transformers', "
            "'faster_whisper') if m in sys.modules]; "
            "assert not heavy, heavy")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_pi_image_copies_every_module_the_gateway_imports():
    """The Pi Dockerfile copies an explicit list of files. A new import that is not on it
    passes every other test (they run in the full repo) and crash-loops the gateway on the
    Pi: src.chat.enrichment did exactly that on 2026-10-26 (KayaChatBot#137)."""
    import re

    code = ("import sys, json, src.gateway.app; "
            "print(json.dumps(sorted(m.__file__ for n, m in sys.modules.items() "
            "if n.startswith('src.') and getattr(m, '__file__', None))))")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    imported = {str(Path(path).resolve().relative_to(REPO.resolve())) for path in json.loads(result.stdout)}
    dockerfile = (REPO / "deploy" / "pi" / "Dockerfile").read_text()
    copied = set()
    for block in re.findall(r"^COPY ((?:[^\n]*\\\n)*[^\n]*)", dockerfile, re.MULTILINE):
        words = block.replace("\\\n", " ").split()
        copied.update(words[:-1])
    def covered(path: str) -> bool:
        return any(path == src or path.startswith(src.rstrip("/") + "/") for src in copied)
    missing = sorted(path for path in imported if not covered(path) and not path.endswith("__init__.py"))
    assert not missing, f"deploy/pi/Dockerfile does not copy: {missing}"
