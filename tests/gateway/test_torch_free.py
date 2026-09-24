"""The Pi cannot run the model stack; the gateway must never import it."""
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
