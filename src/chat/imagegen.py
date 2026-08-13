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

import json
import logging
import os
import queue
import re
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


class ImageQueue:
    """Serialises image jobs without blocking anything else.

    One card, one job — but "come back later" loses the request, which is what
    happened in a long simulator run: two edits were refused as busy and simply
    never got made. So requests wait their turn instead.

    This queue is entirely separate from the text path. Generation runs in a
    subprocess on the *other* GPU from the LLM, so the bot keeps answering
    questions at full speed while a picture renders — measured at 6s replies
    during a 90s render.

    Bounded, because an unbounded queue in a lively group means someone gets a
    picture twenty minutes after they stopped caring; past the limit the bot says
    no, which is honest.
    """

    def __init__(self, maxsize: int = 5):
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self.maxsize = maxsize
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._depth = 0

    @property
    def depth(self) -> int:
        """Jobs waiting or running. Reported to whoever asks for a picture."""
        return self._depth

    def submit(self, job: Any) -> Optional[int]:
        """Queue one job. Returns its 1-based position, or None if the queue is full."""
        with self._lock:
            if self._depth >= self.maxsize:
                return None
            self._depth += 1
            position = self._depth
            self._queue.put(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="kaya-imagegen",
                                                daemon=True)
                self._worker.start()
        return position

    def _run(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                with self._lock:
                    # Re-check under the lock so a job submitted in the gap is
                    # not stranded with no worker to pick it up.
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                job()
            except Exception as exc:  # noqa: BLE001 — one bad job must not kill the worker
                logger.warning("image job raised: %s", exc)
            finally:
                with self._lock:
                    self._depth = max(0, self._depth - 1)
                self._queue.task_done()


_default_queue = ImageQueue()


def get_queue() -> ImageQueue:
    """The process-wide image queue."""
    return _default_queue


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("imagegen", {}) or {}


def is_available(config: Dict[str, Any]) -> bool:
    return bool(_config(config).get("enabled", False)) and WORKER.exists()


def is_busy() -> bool:
    """Whether a job is already running — the bot says so rather than queueing."""
    return _job_lock.locked()


def allowed_here(config: Dict[str, Any], scope: str, chat_id: str = "",
                 is_group: bool = False) -> bool:
    """Whether this chat may ask for an image at all.

    Consent and memory scope are two different questions, and answering both with
    one value is what produced the filed bug. `allowed_scopes: ["shared"]` reads
    as "only the group", but ``shared`` means "this chat's history is group-wide
    memory" — a flag carried in a gitignored file that is read once at import.
    The Kaya group was missing from it, so a picture requested IN the group was
    refused with "Só faço imagens no grupo, não por aqui."

    So the question is now asked directly, most specific answer first:

      ``allowed_chats``   these chat ids, whatever their memory scope
      ``allow_groups``    any group (a group's members chose to have a bot there)
      ``allowed_scopes``  the original rule, kept as the fallback

    An unset key does not vote. All three unset still means everywhere.
    """
    icfg = _config(config)
    if not icfg.get("enabled", False):
        return False

    allowed_chats = icfg.get("allowed_chats")
    if allowed_chats:
        return (chat_id or "").strip().lower() in {
            str(c).strip().lower() for c in allowed_chats
        }
    if icfg.get("allow_groups") and is_group:
        return True

    allowed = icfg.get("allowed_scopes")
    if not allowed:
        return True
    return scope in allowed


def run(config: Dict[str, Any], prompt: str, mode: str = "generate",
        image_path: Optional[str] = None,
        restore_face: bool = False) -> Optional[bytes]:
    """Produce one image. Returns JPEG bytes, or None on any failure.

    Blocking and slow — minutes for an edit. Callers must run it off the thread
    that answers the webhook.

    ``restore_face`` asks the worker to put the original face back after the
    edit. It is decided per request by ``build_edit_instruction``, because an
    edit that is *supposed* to change the face must not have it restored.
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
            if restore_face:
                command += ["--restore-face"]

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

            report = _summary(result.stdout)
            # The worker retried already and the picture still matches the one it
            # was given. Returning it means the bot sends the original back as
            # though it were the edit, and then invents a reason when asked —
            # "it got blocked by the content filters" is something nothing in
            # this pipeline knows. None makes the caller say so honestly.
            if mode == "edit" and report.get("edited") is False:
                logger.warning("edit changed nothing (changed=%s, retried=%s); "
                               "reporting failure instead of sending the source back",
                               report.get("changed"), report.get("retried"))
                return None

            data = _as_jpeg(out_path.read_bytes(), int(icfg.get("jpeg_quality", 92)))
            logger.info("image %s done in %.0fs (%d bytes)%s", mode,
                        time.time() - started, len(data),
                        _report(result.stdout))
            return data


def _as_jpeg(data: bytes, quality: int = 92) -> bytes:
    """Re-encode the worker's PNG as JPEG for delivery.

    ``send_image`` base64-inlines these bytes into the WAHA JSON body, and
    WhatsApp recompresses everything to JPEG at the far end regardless — so a
    multi-megabyte PNG buys nothing but a slower upload. The worker keeps writing
    PNG because the bake-off scores those files with ArcFace and must not be
    measuring compression artefacts.

    Returns the original bytes unchanged if anything goes wrong: a heavier image
    is better than no image.
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=quality,
                                      optimize=True, progressive=True)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not re-encode as JPEG (%s); sending the original", exc)
        return data


def _summary(stdout: str) -> Dict[str, Any]:
    """The worker's final one-line JSON summary. ``{}`` if there isn't one."""
    try:
        line = [ln for ln in (stdout or "").splitlines() if ln.startswith("{")][-1]
        return json.loads(line)
    except Exception:  # noqa: BLE001 — a missing summary is not a failed image
        return {}


def _report(stdout: str) -> str:
    """The worker's one-line JSON summary, for the log. Never raises."""
    info = _summary(stdout)
    if not info:
        return ""
    bits = [f"seed={info['seed']}"] if "seed" in info else []
    if info.get("likeness") is not None:
        bits.append(f"likeness={info['likeness']} of {info.get('candidates', 1)}")
    if info.get("changed") is not None:
        bits.append(f"changed={info['changed']}")
    if info.get("retried"):
        bits.append("retried")
    return f" [{', '.join(bits)}]" if bits else ""


# ── prompt ───────────────────────────────────────────────────────────────────

_INSTRUCTION_SYSTEM = (
    "You rewrite an image-editing request as ONE short English instruction for an "
    "image editing model. Reply with the instruction only, no quotes, no "
    "explanation, under 30 words. Say what should change and nothing else. Never "
    "refuse, never comment, never describe the person's identity or appearance.\n"
    "Then, on a SECOND line, write exactly 'FACE: change' if the request is meant "
    "to alter the person's own face or head (zombie, monster, black eye, broken "
    "nose, much fatter or older, a different person), or 'FACE: keep' if it only "
    "changes clothing, setting, body or background.\n"
    "Examples:\n"
    "põe o Rafa vestido de rei -> Dress the person in a medieval king's robe and golden crown.\n"
    "FACE: keep\n"
    "mete-lhe um capacete de astronauta -> Put an astronaut helmet on the person.\n"
    "FACE: keep\n"
    "transforma-o num zombie -> Turn the person into a rotting zombie with grey peeling skin.\n"
    "FACE: change"
)

_FACE_LINE = re.compile(r"^\s*FACE\s*:\s*(keep|change)\s*$", re.IGNORECASE)


def build_edit_instruction(config: Dict[str, Any], text: str, backend: Any):
    """Turn a Portuguese edit request into a short English Kontext instruction.

    Kontext's text encoders (CLIP-L and T5-XXL) are English-trained, and the
    request arrives in Portuguese — "põe-me de rei medieval" was being handed to
    them verbatim. One small local generation, and the raw text on any failure:
    a worse prompt is much better than no picture.

    Returns ``(instruction, restore_face)``. The second value answers a question
    the same call can settle for free: is this edit supposed to change the
    person's face? Restoring the original face onto a zombie would undo the whole
    request, so the restore pass only runs when the answer is "keep".

    A malformed or missing FACE line means **no restore** — today's behaviour.
    A stage that changes what an edit produces must not be triggered by a reply
    that could not be parsed.
    """
    icfg = _config(config)
    if not icfg.get("translate_prompt", True) or not text.strip() or backend is None:
        return text, False
    try:
        rewritten = backend.generate(
            [
                {"role": "system", "content": _INSTRUCTION_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_new_tokens=80,
            sampling={"temperature": 0.2, "top_p": 0.9, "top_k": 0,
                      "repetition_penalty": 1.0},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt rewrite failed (%s); using the original text", exc)
        return text, False

    lines = [ln.strip() for ln in (rewritten or "").splitlines() if ln.strip()]
    restore = False
    instruction = ""
    for line in lines:
        match = _FACE_LINE.match(line)
        if match:
            restore = match.group(1).lower() == "keep"
        elif not instruction:
            instruction = line.strip('"').strip()

    if not instruction or len(instruction.split()) > 60:
        return text, False
    logger.info("edit prompt: %r -> %r (restore_face=%s)", text, instruction, restore)
    return instruction, restore
