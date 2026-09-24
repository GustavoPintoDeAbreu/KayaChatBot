"""The Pi gateway's two apps: webhook -> journal, offline reply, media, status."""
import asyncio
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.monitor import PcMonitor, PcState

BOT = "351900000000@c.us"
GROUP = "120363000000000000@g.us"
ALICE = "351911111111@c.us"
POWER = {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5,
         "shutdown": {"sun": "23:00", "mon": "23:00", "tue": "23:00", "wed": "23:00",
                      "thu": "23:00", "fri": "02:00", "sat": "02:00"}}


class Clock:
    def __init__(self, start: float = 1_790_000_000.0):
        self.value = start

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def rig(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "whatsapp": {"group": {"respond_on_mention": True, "respond_on_reply": True},
                     "whitelist": {"enabled": True, "allowed": ["351911111111"]}},
        "power": POWER,
    }))
    landing = tmp_path / "landing.html"
    landing.write_text("<html>kaya</html>")
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_LANDING_HTML": str(landing), "GATEWAY_WHITELIST": str(tmp_path / "none.json"),
        "KAYA_RELAY_TOKEN": "relay", "KAYA_WAHA_API_KEY": "waha-key",
        "KAYA_WHATSAPP_WEBHOOK_TOKEN": "hook",
    })
    clock = Clock()
    monitor = PcMonitor(settings.pc_url, now=clock, probe=lambda: PcState.ONLINE)
    monitor.observe(PcState.ONLINE)
    sent, fetched = [], {"fail": False}

    async def fetch_media(url):
        if fetched["fail"]:
            raise OSError("gone")
        return b"OggS-bytes", "audio/ogg"

    gateway = Gateway(settings, monitor=monitor, now=clock,
                      send_text=lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)),
                      fetch_media=fetch_media)
    return gateway, clock, monitor, sent, fetched


def _message(message_id, *, chat=GROUP, sender=ALICE, body="olá", mention=False, media=None,
             timestamp=1_790_000_000):
    payload = {"id": message_id, "from": chat, "body": body, "timestamp": timestamp,
               "notifyName": "Alice", "mentionedIds": [BOT] if mention else []}
    if chat.endswith("@g.us"):
        payload["participant"] = sender
    if media:
        payload.update({"hasMedia": True, "media": media})
    return {"event": "message", "me": {"id": BOT}, "payload": payload}


def _post(gateway, event, token="hook"):
    client = TestClient(gateway.internal_app)
    return client.post("/waha/webhook", json=event, headers={"X-Webhook-Token": token})


def test_a_mention_is_journaled_as_addressed_once(rig):
    gateway, *_ = rig
    first = _post(gateway, _message("m1", mention=True)).json()
    assert first["addressed"] is True
    assert _post(gateway, _message("m1", mention=True)).json() == {"duplicate": True}
    assert gateway.journal.pending_count() == 1


def test_plain_chatter_is_journaled_but_not_addressed(rig):
    gateway, *_ = rig
    assert _post(gateway, _message("m2")).json()["addressed"] is False
    assert gateway.journal.get(1).event["payload"]["body"] == "olá"


def test_voice_note_bytes_are_kept_even_when_the_fetch_fails(rig):
    gateway, _, _, _, fetched = rig
    media = {"url": "http://localhost:3000/api/files/default/v.oga", "mimetype": "audio/ogg"}
    seq = _post(gateway, _message("v1", body="", media=media)).json()["seq"]
    stored = gateway.journal.get(seq)
    assert stored.media_status == "stored"
    assert Path(stored.media_path).read_bytes() == b"OggS-bytes"
    fetched["fail"] = True
    seq = _post(gateway, _message("v2", body="", media=media)).json()["seq"]
    assert gateway.journal.get(seq).media_status == "failed"


def test_offline_reply_only_when_the_pc_is_really_off(rig):
    gateway, clock, monitor, sent, _ = rig
    _post(gateway, _message("d1", chat=ALICE))
    assert sent == []
    monitor.observe(PcState.OFFLINE)
    clock.value += 120
    _post(gateway, _message("d2", chat=ALICE, timestamp=int(clock.value)))
    _post(gateway, _message("d3", chat=ALICE, timestamp=int(clock.value)))
    assert len(sent) == 1
    chat, text, reply_to = sent[0]
    assert chat == ALICE and reply_to == "d2" and "Volto" in text


def test_webhook_token_is_enforced(rig):
    gateway, *_ = rig
    assert _post(gateway, _message("x"), token="wrong").status_code == 401


def test_media_endpoint_checks_key_and_name(rig):
    gateway, *_ = rig
    media = {"url": "http://localhost:3000/api/files/default/v.oga", "mimetype": "audio/ogg"}
    seq = _post(gateway, _message("v3", body="", media=media)).json()["seq"]
    client = TestClient(gateway.internal_app)
    assert client.get(f"/media/{seq}/v.oga").status_code == 401
    assert client.get(f"/media/{seq}/other.oga", headers={"X-Api-Key": "waha-key"}).status_code == 404
    ok = client.get(f"/media/{seq}/v.oga", headers={"X-Api-Key": "waha-key"})
    assert ok.status_code == 200 and ok.content == b"OggS-bytes"


def test_going_down_needs_the_relay_token(rig):
    gateway, _, monitor, _, _ = rig
    client = TestClient(gateway.internal_app)
    assert client.post("/pc/going-down").status_code == 401
    assert client.post("/pc/going-down", headers={"X-Relay-Token": "relay"}).json() == {"ok": True}
    assert monitor.state is PcState.GOING_DOWN


def test_public_app_serves_the_page_and_a_reduced_status(rig):
    gateway, *_ = rig
    client = TestClient(gateway.public_app)
    assert "kaya" in client.get("/").text
    status = client.get("/status")
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["pc_online"] is True and "journal" not in status.json()
    assert client.get("/docs").status_code == 404
    assert client.post("/waha/webhook", json={}).status_code in (404, 405)


def test_background_tasks_start_and_stop(rig):
    gateway, *_ = rig

    async def cycle():
        await gateway.start_background()
        await asyncio.sleep(0.05)
        await asyncio.wait_for(gateway.stop_background(), timeout=10)

    asyncio.run(cycle())
