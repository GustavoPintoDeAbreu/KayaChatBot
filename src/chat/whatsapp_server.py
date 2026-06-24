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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt
from src.chat.gpu_lock import GpuBusyError
from src.chat.whatsapp_adapter import WhatsAppAdapter
from src.chat.waha_client import WahaClient, MockWahaClient
from src.chat import metrics
from src.chat.web_search import CITATION_PREFIX

_docker_cfg = "/app/config.yaml"
_local_cfg = str(Path(__file__).parent.parent.parent / "config.yaml")
config_path = _docker_cfg if os.path.exists(_docker_cfg) else _local_cfg
config = load_config(config_path)

_wcfg = config.setdefault("whatsapp", {})

# Merge real phone->name mappings from a gitignored local file (PII stays out of
# git). Keys are bare phone numbers or full JIDs; see config.yaml whatsapp.contacts.
_contacts_path = Path(config_path).parent / "data" / "whatsapp_contacts.json"
if _contacts_path.exists():
    import json as _json

    try:
        _local_contacts = _json.loads(_contacts_path.read_text(encoding="utf-8"))
        _wcfg["contacts"] = {**(_wcfg.get("contacts") or {}), **_local_contacts}
        print(f"✓ Loaded {len(_local_contacts)} WhatsApp contact name(s) from {_contacts_path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not read {_contacts_path}: {exc}")

# Merge the DM anti-spam whitelist from a gitignored local file (PII stays out of
# git). Shape: {"allowed": ["351913227550", ...]}. Only used when
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

MOCK_MODE = os.environ.get("KAYA_WHATSAPP_MOCK", "").lower() in ("1", "true", "yes") or bool(
    _wcfg.get("mock_mode", False)
)
WEBHOOK_TOKEN = os.environ.get("KAYA_WHATSAPP_WEBHOOK_TOKEN") or _wcfg.get("webhook_token", "")

# Engine (one model load) + the prompt policy for WhatsApp (uncensored per config).
engine = get_engine(config)
_system_prompt = build_system_prompt(
    config, config_path, include_uncensored=config.get("chat", {}).get("uncensored_mode", False)
)


def _responder(message: str, speaker: str, recent_lines):
    return engine.generate_reply(message, speaker, recent_lines, _system_prompt)


if MOCK_MODE:
    print("⚠️  WhatsApp bridge in MOCK mode — replies are captured, not sent to WhatsApp.")
    waha_client = MockWahaClient()
else:
    waha_client = WahaClient(
        base_url=os.environ.get("KAYA_WAHA_URL") or _wcfg.get("waha_base_url", "http://waha:3000"),
        session=_wcfg.get("waha_session", "default"),
        api_key=os.environ.get("KAYA_WAHA_API_KEY") or _wcfg.get("waha_api_key"),
    )

adapter = WhatsAppAdapter(_responder, waha_client, config)
# Ignore any backlog WAHA replays after a reconnect — only answer fresh messages.
adapter.ignore_before_ts = int(time.time())

app = FastAPI(title="Kaya WhatsApp bridge")


@app.get("/whatsapp/health")
def health():
    return {"status": "ok", "mock": MOCK_MODE, "bot_jid": adapter.bot_jid}


@app.get("/whatsapp/outbox")
def outbox():
    """In mock mode, return everything the bot 'sent' (for the simulator/tests)."""
    if isinstance(waha_client, MockWahaClient):
        return {"sent": waha_client.sent}
    raise HTTPException(status_code=404, detail="outbox is only available in mock mode")


def _process(event: dict):
    t0 = time.perf_counter()
    try:
        result = adapter.handle_event(event, system_prompt=_system_prompt)
        if result and result.get("reply") and result.get("command") != "clear":
            metrics.log_interaction(
                source="whatsapp",
                user_message=result.get("user_text", ""),
                assistant_response=result["reply"],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                is_group=bool(result.get("is_group")),
                web_search_used=CITATION_PREFIX in result["reply"],
            )
    except GpuBusyError:
        print("⚠️  GPU busy — dropped a WhatsApp message rather than queueing it.")
    except Exception as exc:  # noqa: BLE001 — never crash the webhook worker
        print(f"⚠️  WhatsApp handler error: {exc}")


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
        result = await run_in_threadpool(adapter.handle_event, event, _system_prompt)
        return {"handled": result is not None, **(result or {})}
    background_tasks.add_task(_process, event)
    return {"handled": True}


# Mount the existing Gradio UI at "/" so one process serves both the web chat and
# the WhatsApp webhook on the same model. Importing web_app reuses the engine.
try:
    import gradio as gr
    from src.chat.web_app import demo

    app = gr.mount_gradio_app(app, demo, path="/")
except Exception as exc:  # noqa: BLE001 — the webhook must work even if the UI fails
    print(f"⚠️  Could not mount Gradio UI: {exc}")


if __name__ == "__main__":
    import uvicorn

    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("KAYA_WEB_PORT") or config.get("chat", {}).get("web_server_port", 7860))
    uvicorn.run(app, host=server_name, port=server_port)
