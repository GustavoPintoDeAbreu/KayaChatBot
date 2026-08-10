"""Image generation and editing, run out of process.

The bot never imports a diffusion pipeline. It shells out to
``scripts/imagegen_worker.py``, which loads the model, produces one image and
exits — see that file for why in-process was rejected (20GB of VRAM that does not
come back, and a card held hostage between requests).

What lives here is everything the *bot* needs to know: whether the feature is
available, which card it may use, how long to wait, and the guarantee that only
one job runs at a time. A second concurrent job would OOM the card and lose both.

Consent: editing puts a real member's face in an invented scene. `allowed_scopes`
gates which chats may ask — by default only the shared group, whose members chose
to have a bot in the room, and never an arbitrary DM from someone who happens to
know the number.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
WORKER = BASE_DIR / "scripts" / "imagegen_worker.py"

# One job at a time: the editor takes ~20GB and a second one would OOM the card
# and lose both images. Held for the whole subprocess, which is why every caller
# runs it off the webhook thread.
_job_lock = threading.Lock()


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("imagegen", {}) or {}


def is_available(config: Dict[str, Any]) -> bool:
    return bool(_config(config).get("enabled", False)) and WORKER.exists()


def is_busy() -> bool:
    """Whether a job is already running — the bot says so rather than queueing."""
    return _job_lock.locked()


def allowed_here(config: Dict[str, Any], scope: str) -> bool:
    """Whether this chat may ask for an image at all."""
    icfg = _config(config)
    if not icfg.get("enabled", False):
        return False
    allowed = icfg.get("allowed_scopes")
    if not allowed:
        return True
    return scope in allowed


def run(config: Dict[str, Any], prompt: str, mode: str = "generate",
        image_path: Optional[str] = None) -> Optional[bytes]:
    """Produce one image. Returns PNG bytes, or None on any failure.

    Blocking and slow — minutes for an edit. Callers must run it off the thread
    that answers the webhook.
    """
    if not is_available(config):
        return None
    if mode == "edit" and not image_path:
        logger.warning("image edit requested with no source photo")
        return None

    icfg = _config(config)
    timeout = float(icfg.get("timeout_seconds", 1800))
    env_device = str(icfg.get("device", "")).strip()

    with _job_lock:
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.png"
            # sys.executable, not the venv path: in the container there is no
            # kaya_chatbot_env and the interpreter is /usr/bin/python.
            command = [
                sys.executable, str(WORKER),
                "--mode", mode, "--prompt", prompt, "--out", str(out_path),
            ]
            if image_path:
                command += ["--image", str(image_path)]
            if icfg.get("editor"):
                command += ["--editor", str(icfg["editor"])]

            env = dict(os.environ)
            if env_device:
                # The LLM holds the other card; pinning keeps the two apart.
                env["CUDA_VISIBLE_DEVICES"] = env_device
            # The diffusion weights are ~120GB and live outside the app's HF_HOME
            # (which holds only the tokenizer). In the container they arrive as a
            # read-only mount of the host cache.
            image_home = os.environ.get("KAYA_IMAGE_HF_HOME") or icfg.get("hf_home")
            if image_home:
                env["HF_HOME"] = str(image_home)
                env["HF_HUB_OFFLINE"] = "1"

            try:
                result = subprocess.run(command, cwd=str(BASE_DIR), env=env,
                                        capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("image worker timed out after %.0fs", timeout)
                return None

            if result.returncode != 0:
                logger.warning("image worker failed (%s): %s",
                               result.returncode, (result.stderr or "")[-500:])
                return None
            if not out_path.exists():
                logger.warning("image worker reported success but wrote nothing")
                return None

            data = out_path.read_bytes()
            logger.info("image %s done in %.0fs (%d bytes)", mode,
                        time.time() - started, len(data))
            return data
