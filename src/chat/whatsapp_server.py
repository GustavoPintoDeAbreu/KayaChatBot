"""WhatsApp bridge server: WAHA webhook + the Gradio UI in one process.

Run this *instead of* ``web_app.py`` when WhatsApp is enabled. It loads the model
once (shared with the mounted Gradio UI via the ``get_engine`` singleton), exposes
``POST /whatsapp/webhook`` for WAHA to push inbound messages to, and sends replies
back through a ``WahaClient``. Generation runs in a threadpool and the webhook
returns ``200`` immediately so WAHA's webhook does not time out while the GPU works.

Mock mode (``KAYA_WHATSAPP_MOCK=1`` or ``whatsapp.mock_mode: true``) swaps in
``MockWahaClient`` so the entire flow runs with no real number — replies are
captured and readable at ``GET /whatsapp/outbox``. This is what
``scripts/whatsapp_simulator.py`` drives.

    KAYA_WHATSAPP_MOCK=1 kaya_chatbot_env/bin/python -m src.chat.whatsapp_server
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt
from src.chat.gpu_lock import GpuBusyError, gpu_section
from src.chat.whatsapp_adapter import WhatsAppAdapter
from src.chat.waha_client import WahaClient, MockWahaClient
from src.chat import metrics
from src.chat import feedback
from src.chat.web_search import CITATION_PREFIX

# WAHA event names that carry an emoji reaction (engine-dependent spelling).
_REACTION_EVENTS = ("message.reaction", "reaction")

_docker_cfg = "/app/config.yaml"
_local_cfg = str(Path(__file__).parent.parent.parent / "config.yaml")
config_path = _docker_cfg if os.path.exists(_docker_cfg) else _local_cfg
config = load_config(config_path)

_wcfg = config.setdefault("whatsapp", {})

# Merge real phone->name mappings from a gitignored local file (PII stays out of
# git). Keys are bare phone numbers or full JIDs; see config.yaml whatsapp.contacts.
import json as _json

_contacts_path = Path(config_path).parent / "data" / "whatsapp_contacts.json"
if _contacts_path.exists():
    try:
        _local_contacts = _json.loads(_contacts_path.read_text(encoding="utf-8"))
        _wcfg["contacts"] = {**(_wcfg.get("contacts") or {}), **_local_contacts}
        print(f"✓ Loaded {len(_local_contacts)} WhatsApp contact name(s) from {_contacts_path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not read {_contacts_path}: {exc}")

# Let the adapter learn phone -> member mappings from real traffic: group_members.json
# has names and aliases but no phone numbers, so the map cannot be generated ahead of
# time. Matching WhatsApp's pushName against an alias fills it in as people talk.
_wcfg["contacts_file"] = str(_contacts_path)
try:
    _members_file = config.get("data", {}).get("group_members_file")
    if _members_file:
        _mpath = Path(_members_file)
        if not _mpath.is_absolute():
            _mpath = Path(config_path).parent / _members_file
        if _mpath.exists():
            _members = _json.loads(_mpath.read_text(encoding="utf-8"))
            _members = _members.get("members", _members) if isinstance(_members, dict) else _members
            _aliases = {}
            for _m in _members:
                _name = _m.get("name") if isinstance(_m, dict) else str(_m)
                if not _name:
                    continue
                _aliases[_name.lower()] = _name
                for _a in (_m.get("aliases") or []) if isinstance(_m, dict) else []:
                    _aliases[str(_a).lower()] = _name
            _wcfg["member_aliases"] = _aliases
            print(f"✓ Loaded {len(_aliases)} member name/alias(es) for speaker resolution")
except Exception as exc:  # noqa: BLE001
    print(f"⚠️  Could not load member aliases: {exc}")

# Which chats count as GROUP-WIDE memory, from a gitignored local file (a chat id
# is still an identifier, so it stays out of git like the contacts and whitelist).
# Shape: {"shared_chats": ["1203...@g.us"]}. Without this the group's own history
# is private to it — safe, but it loses the shared memory the bot exists for.
_scopes_path = Path(config_path).parent / "data" / "whatsapp_shared_chats.json"
if _scopes_path.exists():
    try:
        _shared = _json.loads(_scopes_path.read_text(encoding="utf-8"))
        _wcfg["shared_chats"] = list(
            {*(_wcfg.get("shared_chats") or []), *(_shared.get("shared_chats") or [])}
        )
        print(f"✓ Loaded {len(_wcfg['shared_chats'])} shared-memory chat id(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not read {_scopes_path}: {exc}")

# Merge the DM anti-spam whitelist from a gitignored local file (PII stays out of
# git). Shape: {"allowed": ["351XXXXXXXXX", ...]}. Only used when
# whatsapp.whitelist.enabled is true; see config.yaml whatsapp.whitelist.
_whitelist_path = Path(config_path).parent / "data" / "whatsapp_whitelist.json"
if _whitelist_path.exists():
    import json as _json

    try:
        _wl = _json.loads(_whitelist_path.read_text(encoding="utf-8"))
        _allowed = _wl.get("allowed", _wl) if isinstance(_wl, dict) else _wl
        _wcfg.setdefault("whitelist", {})
        merged = list({*(_wcfg["whitelist"].get("allowed") or []), *(_allowed or [])})
        _wcfg["whitelist"]["allowed"] = merged
        print(f"✓ Loaded {len(_allowed or [])} WhatsApp whitelist number(s) from {_whitelist_path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not read {_whitelist_path}: {exc}")

# Where /bug and /feedback reports are announced. A bare phone or a full JID; the
# adapter normalises it. Env-only for the same reason as the whitelist above — it
# is somebody's real number, so it lives in .env rather than config.yaml.
_report_to = os.environ.get("KAYA_REPORT_JID", "").strip()
if _report_to:
    _wcfg["report_to"] = _report_to
    print("✓ Bug/feedback reports will be announced by DM")

MOCK_MODE = os.environ.get("KAYA_WHATSAPP_MOCK", "").lower() in ("1", "true", "yes") or bool(
    _wcfg.get("mock_mode", False)
)
WEBHOOK_TOKEN = os.environ.get("KAYA_WHATSAPP_WEBHOOK_TOKEN") or _wcfg.get("webhook_token", "")

# Engine (one model load) + the prompt policy for WhatsApp (uncensored per config).
engine = get_engine(config)
_system_prompt = build_system_prompt(
    config, config_path, include_uncensored=config.get("chat", {}).get("uncensored_mode", False)
)


def _responder(message: str, speaker: str, recent_lines, scope=None,
               exclude_from=None, summary: str = ""):
    """Answer one message, returning the text AND how it was routed.

    ``respond`` (rather than ``generate_reply``) so the adapter can act on routed
    commands — switching a chat to voice replies, clearing its context — which are
    executed in code rather than generated.

    ``scope`` limits which chat's long-term memory may be retrieved, and
    ``exclude_from`` stops retrieval re-injecting the recent turns the prompt
    already carries verbatim. ``summary`` is this chat's rolling summary of the
    turns that have already scrolled out of that window.
    """
    return engine.respond(
        message, speaker, recent_lines, _system_prompt,
        scope=scope, exclude_from=exclude_from, summary=summary,
    )


if MOCK_MODE:
    print("⚠️  WhatsApp bridge in MOCK mode — replies are captured, not sent to WhatsApp.")
    waha_client = MockWahaClient()
else:
    waha_client = WahaClient(
        base_url=os.environ.get("KAYA_WAHA_URL") or _wcfg.get("waha_base_url", "http://waha:3000"),
        session=_wcfg.get("waha_session", "default"),
        api_key=os.environ.get("KAYA_WAHA_API_KEY") or _wcfg.get("waha_api_key"),
    )

def _tts(text: str):
    """Synthesise a voice note, or None if voice replies are unavailable.

    Injected into the adapter so it stays free of TTS imports. Piper runs on CPU
    (~28x realtime), so speaking never competes with the GPU that is answering.
    """
    from src.chat import tts

    if not tts.is_available(config):
        return None
    return tts.synthesize_voice_note(text, config)


def _speech_text(text: str) -> str:
    """The spoken form of a written reply (see tts.sanitize_for_speech)."""
    from src.chat import tts

    return tts.sanitize_for_speech(text, citation_prefix=CITATION_PREFIX)


def _stt(url: str, mimetype: str):
    """Transcribe an incoming voice note, or None if STT is unavailable."""
    from src.chat import stt

    if not stt.is_available(config):
        return None
    return stt.transcribe_url(
        url, mimetype, config,
        api_key=os.environ.get("KAYA_WAHA_API_KEY", ""),
        # WAHA reports its files as localhost:3000, which is unreachable from
        # this container; rewrite to the address we actually talk to it on.
        waha_base_url=os.environ.get("KAYA_WAHA_URL") or _wcfg.get("waha_base_url", ""),
    )


def _fetch_media(url: str, mimetype: str):
    """Download an inbound photo to a temp path, or None. Same localhost rewrite
    as voice notes: WAHA describes its files by its own hostname."""
    import tempfile

    import httpx

    from src.chat.stt import rewrite_media_url

    url = rewrite_media_url(
        url, os.environ.get("KAYA_WAHA_URL") or _wcfg.get("waha_base_url", ""))
    api_key = os.environ.get("KAYA_WAHA_API_KEY", "")
    try:
        headers = {"X-Api-Key": api_key} if api_key else {}
        with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.content
    except Exception as exc:  # noqa: BLE001 — a failed fetch must not raise
        print(f"⚠️  could not fetch image {url}: {exc}")
        return None

    suffix = ".png" if "png" in (mimetype or "") else ".jpg"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.write(payload)
    handle.close()
    return handle.name


def _describe(url: str, mimetype: str):
    """Read an inbound photo with the serving model, or None if vision is off."""
    from src.chat import vision

    if not vision.is_available(config):
        return None
    return vision.describe_url(
        url, mimetype, config,
        api_key=os.environ.get("KAYA_WAHA_API_KEY", ""),
        waha_base_url=os.environ.get("KAYA_WAHA_URL") or _wcfg.get("waha_base_url", ""),
    )


def _imagegen(mode: str, prompt: str, image_path=None):
    """Make one image, or None. Blocking — the adapter calls it off-thread.

    An edit request is rewritten into a short English instruction first: Kontext's
    text encoders are English-trained and the request arrives in Portuguese. The
    rewrite is one small local generation and falls back to the raw text.
    """
    from src.chat import imagegen

    # Generation from scratch has no original face to restore.
    restore_face = False
    if mode == "edit":
        with gpu_section(config):
            # The same call also answers whether this edit is meant to change the
            # person's face — restoring the original face onto a zombie would undo
            # the request.
            prompt, restore_face = imagegen.build_edit_instruction(
                config, prompt, engine.backend)
    return imagegen.run(config, prompt, mode=mode, image_path=image_path,
                        restore_face=restore_face)


from src.chat.summary import SummaryWriter

# Rolling per-chat summary of what has scrolled out of the verbatim window.
# Shares the engine's backend, so no second model is loaded.
_summary_writer = SummaryWriter(config, engine.backend)

adapter = WhatsAppAdapter(_responder, waha_client, config,
                          tts_synthesize=_tts, speech_text=_speech_text,
                          transcribe=_stt,
                          image_generate=_imagegen, fetch_media=_fetch_media,
                          describe_image=_describe,
                          summary_writer=_summary_writer)
# Ignore any backlog WAHA replays after a reconnect — only answer fresh messages.
adapter.ignore_before_ts = int(time.time())

app = FastAPI(title="Kaya WhatsApp bridge")


@app.get("/whatsapp/health")
def health():
    return {"status": "ok", "mock": MOCK_MODE, "bot_jid": adapter.bot_jid}


@app.post("/whatsapp/ingest")
def ingest_now():
    """Fold logged messages into the vector store, in THIS process. Mock only.

    The simulator needs recall of something said earlier in the same run, and mock
    mode deliberately skips the ingest scheduler. Running the ingester as a
    separate process does not work: this process's ChromaDB client keeps its own
    view, so the app goes on answering "não tenho essa informação" about a fact
    that is sitting in the store. Production never has this problem — its
    scheduler runs in-process, against the same client.
    """
    if not MOCK_MODE:
        raise HTTPException(status_code=404, detail="ingest is only available in mock mode")
    from src.data.ingest import run_ingest

    results = run_ingest(config)
    return {"scopes": len(results),
            "chunks": sum(r.get("chunks", 0) for r in results),
            "messages": sum(r.get("messages", 0) for r in results)}


@app.get("/whatsapp/outbox")
def outbox():
    """In mock mode, return everything the bot 'sent' (for the simulator/tests)."""
    if isinstance(waha_client, MockWahaClient):
        return {"sent": waha_client.sent}
    raise HTTPException(status_code=404, detail="outbox is only available in mock mode")


def _log_reaction_feedback(fb: dict) -> None:
    """Persist a 👍/👎 emoji reaction on a bot reply as user feedback."""
    if not fb:
        return
    feedback.log_rating(
        source="whatsapp",
        rating=fb["rating"],
        user_message=fb.get("user_text", ""),
        assistant_response=fb.get("reply", ""),
        is_group=bool(fb.get("is_group")),
    )


def _process_reaction(event: dict):
    try:
        _log_reaction_feedback(adapter.handle_reaction(event))
    except Exception as exc:  # noqa: BLE001 — never crash the webhook worker
        print(f"⚠️  WhatsApp reaction handler error: {exc}")


def _log_interaction_metrics(result: dict, t0: float) -> None:
    """Log one answered message to the metrics sink (shared by the live + mock paths)."""
    # Command confirmations ("volto a responder por texto") are bookkeeping, not
    # conversation — don't pollute the interaction metrics with them.
    if result and result.get("reply") and not result.get("command"):
        metrics.log_interaction(
            source="whatsapp",
            user_message=result.get("user_text", ""),
            assistant_response=result["reply"],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            is_group=bool(result.get("is_group")),
            web_search_used=bool(result.get("citation")),
            delivered_as=result.get("delivered_as", "text"),
            spoken_text=result.get("spoken_text", ""),
            # route_mode, retrieval and who the reply named — see Reply.telemetry.
            **(result.get("telemetry") or {}),
        )


def _process(event: dict):
    if event.get("event") in _REACTION_EVENTS:
        _process_reaction(event)
        return
    t0 = time.perf_counter()
    try:
        result = adapter.handle_event(event, system_prompt=_system_prompt)
        _log_interaction_metrics(result, t0)
    except GpuBusyError:
        print("⚠️  GPU busy — dropped a WhatsApp message rather than queueing it.")
    except Exception as exc:  # noqa: BLE001 — never crash the webhook worker
        print(f"⚠️  WhatsApp handler error: {exc}")


def _preload_audio_models() -> None:
    """Warm Whisper in the background so the first voice note is not slow.

    Loading large-v3 takes ~30s, and it happens lazily on first use — which made
    the first voice note of a session take 40-46s end to end, well past the
    latency budget. Doing it at startup moves that cost off the user's path.
    """
    from src.chat import stt

    if not stt.is_available(config):
        return

    def _warm() -> None:
        try:
            import time as _t
            t0 = _t.time()
            stt._load(config)
            print(f"✓ Whisper preloaded in {_t.time() - t0:.0f}s")
        except Exception as exc:  # noqa: BLE001 — warming is best-effort
            print(f"⚠️  Whisper preload failed: {exc}")

    threading.Thread(target=_warm, name="kaya-whisper-warm", daemon=True).start()


def _start_ingest_scheduler() -> None:
    """Catch up on what was missed while down, then keep folding in new messages.

    Runs in a daemon thread, never at message time: embedding competes with
    answering for the GPU, and a reply must not wait on it. Ingest is idempotent
    (chunk ids derive from message ids), so a crash mid-run is safe to repeat.
    """
    icfg = (_wcfg.get("ingest") or {})
    if not icfg.get("on_boot", True) and not icfg.get("interval_minutes", 0):
        return

    from src.data.ingest import run_ingest

    def _loop() -> None:
        if icfg.get("on_boot", True):
            try:
                run_ingest(config)
            except Exception as exc:  # noqa: BLE001 — ingestion must never take the bot down
                print(f"⚠️  Boot ingest failed: {exc}")
        interval = float(icfg.get("interval_minutes", 0) or 0)
        if interval <= 0:
            return
        while True:
            time.sleep(interval * 60)
            try:
                run_ingest(config)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Periodic ingest failed: {exc}")

    threading.Thread(target=_loop, name="kaya-ingest", daemon=True).start()
    print(
        f"✓ Ingestion scheduled (boot={icfg.get('on_boot', True)}, "
        f"every {icfg.get('interval_minutes', 0)} min)"
    )


if not MOCK_MODE:
    _preload_audio_models()
    _start_ingest_scheduler()


@app.post("/whatsapp/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_webhook_token: str = Header(default=""),
):
    if WEBHOOK_TOKEN and x_webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid webhook token")
    event = await request.json()
    if os.environ.get("KAYA_WHATSAPP_DEBUG"):
        import json as _json
        print(f"[wpp-debug] raw event: {_json.dumps(event, ensure_ascii=False)[:3000]}", flush=True)
    # Generation is slow; in mock mode we await it so the simulator/tests see the
    # reply, but in production we ack immediately and generate in the background so
    # WAHA's webhook doesn't time out and retry (which would duplicate replies).
    if MOCK_MODE:
        if event.get("event") in _REACTION_EVENTS:
            fb = await run_in_threadpool(adapter.handle_reaction, event)
            _log_reaction_feedback(fb)
            return {"handled": fb is not None, **(fb or {})}
        t0 = time.perf_counter()
        result = await run_in_threadpool(adapter.handle_event, event, _system_prompt)
        _log_interaction_metrics(result, t0)
        return {"handled": result is not None, **(result or {})}
    background_tasks.add_task(_process, event)
    return {"handled": True}


def _web_credentials() -> Optional[Tuple[str, str]]:
    """The username/password pair guarding the chat UI, or None if unset.

    Same precedence as web_app's own launch path: the environment wins over
    config.yaml, so the deployed secret never has to live in the repo.
    """
    env_user = os.environ.get("KAYA_WEB_USER")
    env_pass = os.environ.get("KAYA_WEB_PASS")
    if env_user and env_pass:
        return (env_user, env_pass)
    cfg_auth = (config.get("chat", {}) or {}).get("web_auth")
    return tuple(cfg_auth) if cfg_auth else None


# The public landing page. "/" explains what the bot is and does, to anyone, with
# no login — it is the page the group is given. The chat itself lives behind a
# password at "/app", which is what the page's button points at.
_LANDING = Path(__file__).parent / "static" / "landing.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> HTMLResponse:
    try:
        return HTMLResponse(_LANDING.read_text(encoding="utf-8"))
    except OSError:
        # Never strand the visitor: without the page, send them to the login.
        return HTMLResponse('<meta http-equiv="refresh" content="0; url=/app">', 200)


# Mount the Gradio UI at "/app" so one process serves the web chat, the landing
# page and the WhatsApp webhook on the same model. Importing web_app reuses the
# engine.
#
# `auth` is the load-bearing argument. It was absent for as long as this mount
# existed, and because prod runs THIS module — not web_app's __main__ — the
# KAYA_WEB_USER/KAYA_WEB_PASS pair was set in the environment and silently
# ignored: /config served the whole app to anyone who asked. Cloudflare Access
# was the only thing in front of it. Do not drop this argument.
try:
    import gradio as gr
    from src.chat.web_app import demo

    _auth = _web_credentials()
    if _auth is None:
        print("⚠️  No KAYA_WEB_USER/KAYA_WEB_PASS — the chat UI is UNAUTHENTICATED.")
    app = gr.mount_gradio_app(app, demo, path="/app", auth=_auth)
except Exception as exc:  # noqa: BLE001 — the webhook must work even if the UI fails
    print(f"⚠️  Could not mount Gradio UI: {exc}")


if __name__ == "__main__":
    import uvicorn

    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("KAYA_WEB_PORT") or config.get("chat", {}).get("web_server_port", 7860))
    uvicorn.run(app, host=server_name, port=server_port)
