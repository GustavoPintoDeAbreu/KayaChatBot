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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
WORKER = BASE_DIR / "scripts" / "imagegen_worker.py"

# One job at a time: the editor takes ~20GB and a second one would OOM the card
# and lose both images. Held for the whole subprocess, which is why every caller
# runs it off the webhook thread.
_job_lock = threading.Lock()

DEFAULT_LEASE_PATH = BASE_DIR / "data" / "gpu0_lease.json"

# What an edit is about, decided per request by build_edit_instruction. It picks
# the editor and gates the identity machinery. `person` is the fallback for every
# failure path, because that is the behaviour that shipped.
SUBJECT_PERSON = "person"


def lease_path(config: Dict[str, Any]) -> Path:
    """Where the maintenance lease lives, honouring ``chat.imagegen.lease_file``."""
    custom = _config(config).get("lease_file")
    if not custom:
        return DEFAULT_LEASE_PATH
    path = Path(custom)
    return path if path.is_absolute() else (BASE_DIR / custom)


def read_lease(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The active maintenance lease, or None if there is none or it has expired.

    Image generation is the only thing that uses GPU0 — serving, Whisper and
    vision all live on the other card — so pausing it frees a whole 24GB while
    the bot keeps answering. That is what makes heavy GPU work (regenerating
    member profiles, a bake-off) possible without taking prod down.

    Expiry is the safety. A crashed maintenance job cannot leave image
    generation dead forever: the worst case is that it resumes by itself when
    the lease runs out. A missing or unreadable file means not paused, which is
    the right way to fail.
    """
    try:
        path = lease_path(config)
        if not path.exists():
            return None
        lease = json.loads(path.read_text(encoding="utf-8"))
        if float(lease.get("expires_at", 0)) <= time.time():
            return None
        return lease
    except Exception as exc:  # noqa: BLE001 — a broken lease must not stop the bot
        logger.warning("could not read the GPU lease (%s); treating as free", exc)
        return None


def is_paused(config: Dict[str, Any]) -> bool:
    return read_lease(config) is not None


def pause_remaining(config: Dict[str, Any]) -> int:
    """Whole minutes left on the lease, rounded up. 0 when not paused."""
    lease = read_lease(config)
    if not lease:
        return 0
    seconds = float(lease.get("expires_at", 0)) - time.time()
    return max(1, int(seconds // 60) + (1 if seconds % 60 else 0))


def acquire_lease(config: Dict[str, Any], reason: str, ttl_seconds: float,
                  holder: str = "") -> Dict[str, Any]:
    """Claim GPU0. Writes the lease; the caller still has to wait for `is_busy`."""
    lease = {
        "holder": holder or f"pid:{os.getpid()}",
        "reason": reason,
        "acquired_at": time.time(),
        "expires_at": time.time() + float(ttl_seconds),
    }
    path = lease_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lease, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return lease


def release_lease(config: Dict[str, Any]) -> bool:
    """Hand GPU0 back. The queue drains on its own from there."""
    path = lease_path(config)
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("could not release the GPU lease: %s", exc)
        return False


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

    def __init__(self, maxsize: int = 5, config: Optional[Dict[str, Any]] = None):
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self.maxsize = maxsize
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._depth = 0
        # Needed only to read the maintenance lease. The queue is process-wide
        # and built at import, before any config exists, so the first caller
        # that has one hands it over via ``configure``.
        self.config: Dict[str, Any] = config or {}

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config or {}

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
            # Checked BEFORE dequeuing, so a paused job keeps its place in the
            # queue and `depth` stays honest for the "tenho N à frente" line.
            # The worker must not take the exit branch below while paused with
            # work waiting, or nothing would be left to drain on release.
            if is_paused(self.config):
                time.sleep(2.0)
                continue
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


def _keep_output(config: Dict[str, Any], out_path: Path, report: Dict[str, Any]) -> None:
    """Copy the render and what produced it into ``data/imagegen_log/``.

    The worker writes into a TemporaryDirectory that is deleted the moment the
    bytes are read, so when the group said an edit came out badly there was
    nothing left to look at: no output, no source, and no record of the prompt.
    Off by default; ``keep_outputs`` is how many renders to retain.
    """
    icfg = _config(config)
    keep = int(icfg.get("keep_outputs", 0) or 0)
    if keep <= 0:
        return
    try:
        directory = Path(icfg.get("keep_outputs_dir") or (BASE_DIR / "data" / "imagegen_log"))
        directory.mkdir(parents=True, exist_ok=True)
        # Seed in the name because the stamp is only second-resolution, and
        # because it is the one value that reproduces the render.
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        stem = directory / f"{stamp}_{report.get('mode', 'edit')}_{report.get('seed', 0)}"
        shutil.copyfile(out_path, stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        # Oldest first. Keyed off the sidecar, so pruning can only ever delete
        # renders this function wrote — never anything else left in the directory.
        ours = sorted(directory.glob("*.json"))
        for stale in ours[:-keep]:
            stale.with_suffix(".png").unlink(missing_ok=True)
            stale.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 — a debugging aid never breaks a render
        logger.warning("could not keep the render: %s", exc)


def resolve_editor(config: Dict[str, Any], subject: str) -> str:
    """Which editor answers this request.

    A single editor was chosen by a bake-off whose every photo was picked for
    having a large frontal face and whose every edit transformed a person, so it
    was chosen on identity preservation alone. That is the right criterion for
    "põe o Rafa de rei" and the wrong one for "turn these cans into something
    else" — there is no identity to preserve. ``editors`` maps subject to editor;
    ``editor`` stays the fallback and the answer when the mapping is absent.
    """
    icfg = _config(config)
    mapping = icfg.get("editors") or {}
    return str(mapping.get(subject) or icfg.get("editor") or "").strip()


def run(config: Dict[str, Any], prompt: str, mode: str = "generate",
        image_path: Optional[str] = None,
        restore_face: bool = False, heavy: bool = False,
        subject: str = SUBJECT_PERSON,
        on_report: Optional[Callable[[Dict[str, Any]], None]] = None) -> Optional[bytes]:
    """Produce one image. Returns JPEG bytes, or None on any failure.

    Blocking and slow — minutes for an edit. Callers must run it off the thread
    that answers the webhook.

    ``restore_face`` asks the worker to put the original face back after the
    edit. It is decided per request by ``build_edit_instruction``, because an
    edit that is *supposed* to change the face must not have it restored.

    ``heavy`` says this edit changes what the people are doing rather than how
    they look, and comes from the same call. Those need more guidance: "make
    these two kiss" at the costume-swap setting came back as the original photo.

    ``subject`` comes from the same call again, and decides both the editor and
    whether the identity machinery runs at all. Everything that protects a
    likeness — the identity clause, the crop that frames faces, best-of-N scored
    by ArcFace — is dead weight on a photo of a shop counter, and the clause is
    actively harmful there.
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
            editor = resolve_editor(config, subject)
            if editor:
                command += ["--editor", editor]
            if restore_face:
                command += ["--restore-face"]
            if heavy:
                command += ["--guidance-scale",
                            str(icfg.get("heavy_guidance_scale", 3.5))]
            # Nobody in the picture, nothing to preserve. The worker drops the
            # clause on its own once insightface confirms zero faces; this says
            # the same thing from the request side, so an edit the model called
            # an object edit is not framed around, or ranked by, a face.
            if mode == "edit" and subject != SUBJECT_PERSON:
                command += ["--no-identity-clause", "--no-face-crop",
                            "--candidates", "1"]

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
            report.update({"mode": mode, "subject": subject, "editor": editor,
                           "instruction": prompt,
                           "seconds": round(time.time() - started, 1)})
            if on_report is not None:
                # Everything the worker decided — the final prompt with whatever
                # clause was appended, the editor, the seed, the likeness — used
                # to exist only as a logger.info on a logger nobody configures,
                # so the one edit the group complained about left no trace at all.
                try:
                    on_report(dict(report))
                except Exception as exc:  # noqa: BLE001 — telemetry never breaks a render
                    logger.warning("image report callback failed: %s", exc)
            _keep_output(config, out_path, report)
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
    "refuse, never comment, never describe anyone's identity or appearance.\n"
    "Then, on a SECOND line, write exactly 'SUBJECT: person' if the edit is about "
    "a person or people in the photo, 'SUBJECT: object' if it is about a thing in "
    "the photo, or 'SUBJECT: scene' if it is about the setting, background, "
    "lighting or style of the whole picture.\n"
    "Then, on a THIRD line, write exactly 'FACE: change' if the request is meant "
    "to alter a person's own face or head (zombie, monster, black eye, broken "
    "nose, much fatter or older, a different person), or 'FACE: keep' otherwise.\n"
    "Then, on a FOURTH line, write exactly 'EDIT: heavy' if the request changes "
    "what is happening in the picture — a pose, a position, how things or people "
    "interact, adding or removing something — or 'EDIT: light' if it only dresses, "
    "recolours or restyles what is already there.\n"
    "Examples:\n"
    "põe o Rafa vestido de rei -> Dress the person in a medieval king's robe and golden crown.\n"
    "SUBJECT: person\n"
    "FACE: keep\n"
    "EDIT: light\n"
    "transforma-o num zombie -> Turn the person into a rotting zombie with grey peeling skin.\n"
    "SUBJECT: person\n"
    "FACE: change\n"
    "EDIT: light\n"
    "põe estes dois a beijarem-se -> Make the two men kiss each other on the lips.\n"
    "SUBJECT: person\n"
    "FACE: keep\n"
    "EDIT: heavy\n"
    "transform the monster cans into dildos -> Replace the energy drink cans on the counter with dildos.\n"
    "SUBJECT: object\n"
    "FACE: keep\n"
    "EDIT: heavy\n"
    "mete o carro vermelho -> Change the car's colour to red.\n"
    "SUBJECT: object\n"
    "FACE: keep\n"
    "EDIT: light\n"
    "mete isto a nevar -> Make it snow over the whole scene.\n"
    "SUBJECT: scene\n"
    "FACE: keep\n"
    "EDIT: light"
)

_SUBJECT_LINE = re.compile(r"^\s*SUBJECT\s*:\s*(person|object|scene)\s*$", re.IGNORECASE)
_FACE_LINE = re.compile(r"^\s*FACE\s*:\s*(keep|change)\s*$", re.IGNORECASE)
_EDIT_LINE = re.compile(r"^\s*EDIT\s*:\s*(light|heavy)\s*$", re.IGNORECASE)


def build_edit_instruction(config: Dict[str, Any], text: str, backend: Any):
    """Turn a Portuguese edit request into a short English Kontext instruction.

    Kontext's text encoders (CLIP-L and T5-XXL) are English-trained, and the
    request arrives in Portuguese — "põe-me de rei medieval" was being handed to
    them verbatim. One small local generation, and the raw text on any failure:
    a worse prompt is much better than no picture.

    Returns ``(instruction, restore_face, heavy, subject)``. The extra values
    answer three questions the same call settles for free.

    What is this edit even about? The whole edit path used to assume a person,
    down to the few-shot examples here, and "transform the monster cans into
    dildos" on a photo of a shop counter was rewritten as a person edit and then
    handed an instruction to preserve a face that was not in the picture.
    ``subject`` selects the editor and turns the identity machinery off.

    Is this edit supposed to change the person's face? Restoring the original
    face onto a zombie would undo the whole request, so the restore pass only
    runs when the answer is "keep".

    And how far does the picture have to move? A costume swap and "make these
    two kiss" were being asked for at the same guidance, and the second one came
    back unchanged. A pose or interaction change needs to be pushed harder.

    A malformed or missing FACE line means **no restore**, a malformed EDIT line
    means the ordinary guidance, and a malformed SUBJECT line means ``person`` —
    all three are today's behaviour. A stage that changes what an edit produces
    must not be triggered by a reply that could not be parsed.
    """
    icfg = _config(config)
    if not icfg.get("translate_prompt", True) or not text.strip() or backend is None:
        return text, False, False, SUBJECT_PERSON
    try:
        rewritten = backend.generate(
            [
                {"role": "system", "content": _INSTRUCTION_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_new_tokens=96,
            sampling={"temperature": 0.2, "top_p": 0.9, "top_k": 0,
                      "repetition_penalty": 1.0},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt rewrite failed (%s); using the original text", exc)
        return text, False, False, SUBJECT_PERSON

    lines = [ln.strip() for ln in (rewritten or "").splitlines() if ln.strip()]
    restore = False
    heavy = False
    subject = SUBJECT_PERSON
    instruction = ""
    for line in lines:
        face = _FACE_LINE.match(line)
        edit = _EDIT_LINE.match(line)
        subj = _SUBJECT_LINE.match(line)
        if face:
            restore = face.group(1).lower() == "keep"
        elif edit:
            heavy = edit.group(1).lower() == "heavy"
        elif subj:
            subject = subj.group(1).lower()
        elif not instruction:
            instruction = line.strip('"').strip()

    if not instruction or len(instruction.split()) > 60:
        return text, False, False, SUBJECT_PERSON
    logger.info("edit prompt: %r -> %r (subject=%s, restore_face=%s, heavy=%s)",
                text, instruction, subject, restore, heavy)
    return instruction, restore, heavy, subject
