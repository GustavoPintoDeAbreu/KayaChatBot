"""The public showcase page's "Get in touch" form: a visitor's note reaches the owner,
over WhatsApp and/or email, and nothing else leaves (no owner address on the page, no send
for a bot or an empty submit)."""
import sys
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.contact import (ContactNotifier, ContactThrottle, check_proof, clean_contact_fields,
                                 is_honeypot, make_proof, proof_key, render_thanks)

OWNER_NUMBER = "351900000000"
OWNER = f"{OWNER_NUMBER}@c.us"
SECRET_EMAIL = "gustavo@private-domain-example.invalid"

_sessions: list = []


class _FakeSMTP:
    """Stands in for the network: records what a session would have sent, and can fail."""

    def __init__(self, host, port, *args):
        self.host, self.port = host, port
        self.sent = []
        _sessions.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def starttls(self, **_):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        self.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    """Patch the transport: the notifier must build the message and use STARTTLS, but no test
    connects to anything."""
    _sessions.clear()
    monkeypatch.setattr("src.gateway.contact.smtplib.SMTP", _FakeSMTP)
    return _sessions


@pytest.fixture
def clock():
    """The gateway's clock, settable: the dwell and the 72 h expiry are about time and a test
    must not sleep for either."""
    return [1_700_000_000.0]


@pytest.fixture
def rig(tmp_path, clock):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"whatsapp": {"group": {"respond_on_mention": True}}}))
    landing = tmp_path / "landing.html"
    landing.write_text('<html><form class="contact-form" action="/contact" method="post">'
                       '<input type="hidden" name="proof" value="PROOF-TOKEN"></form></html>')
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_LANDING_HTML": str(landing), "GATEWAY_WHITELIST": str(tmp_path / "none.json"),
        "GATEWAY_ASSETS_DIR": str(tmp_path / "assets"),
        "KAYA_RELAY_TOKEN": "relay",
        # Nothing contact-specific is set: IDEA_OWNER_NUMBERS is what the taps already know, and
        # the DM target is derived from it.
        "IDEA_OWNER_NUMBERS": OWNER_NUMBER,
        "CONTACT_SMTP_HOST": "smtp.private-domain-example.invalid", "CONTACT_SMTP_PORT": "587",
        "CONTACT_SMTP_USER": "", "CONTACT_SMTP_PASSWORD": "",
        "CONTACT_EMAIL_TO": SECRET_EMAIL, "CONTACT_EMAIL_FROM": "",
    })

    sent = []
    gateway = Gateway(settings, send_text=lambda chat, text, reply_to=None: sent.append((chat, text)),
                      now=lambda: clock[0])
    client = TestClient(gateway.public_app)
    return gateway, client, sent


def _note(name="Alice", email="visitor@elsewhere-example.invalid", message="Can you help with a site?"):
    return {"name": name, "email": email, "message": message}


def _browser(client, clock, pause=5.0, headers=None, **fields):
    """Submits the way a browser does: it fetched the page, so it holds the token that page was
    handed, and it paused past the dwell before posting."""
    page = client.get("/").text
    match = re.search(r'name="proof" value="([^"]*)"', page)
    assert match, "the served page must carry a token the route will accept"
    clock[0] += pause
    return client.post("/contact", data={**fields, "proof": match.group(1)}, headers=headers or {})


def test_a_full_note_is_cleaned_and_kept():
    assert clean_contact_fields(_note()) == _note()
    assert clean_contact_fields({"name": "  Alice  ", "email": "a@b.co", "message": " hi "}) == {
        "name": "Alice", "email": "a@b.co", "message": "hi"}
    long = {"name": "n" * 300, "email": "a@b.co", "message": "m" * 5000}
    kept = clean_contact_fields(long)
    assert (len(kept["name"]), len(kept["message"])) == (120, 2000), long


def test_a_half_note_is_not_a_visitor():
    for missing in ("name", "email", "message"):
        form = {k: v for k, v in _note().items() if k != missing}
        assert clean_contact_fields(form) is None, missing
    assert clean_contact_fields({"name": "x", "email": "no-at-sign", "message": "hi"}) is None
    assert clean_contact_fields({"name": "x", "email": "me@nodot", "message": "hi"}) is None


def test_the_honeypot_field_marks_a_bot():
    assert not is_honeypot(_note())
    assert is_honeypot({**_note(), "website": "http://spam-example.invalid"})
    assert is_honeypot({"name": "x", "website": "  ", "email": "a@b.co", "message": "hi"}) is False


def test_the_proof_token_is_one_we_minted_and_needs_a_dwell():
    key = proof_key("relay")
    token = make_proof(key, 1_000.0)
    assert check_proof(key, token, 1_003.5) is True
    assert check_proof(key, token, 1_002.9) is False              # nobody typed that fast
    assert check_proof(key, token, 1_000.0 + 259_201.0) is False   # a page kept for weeks
    assert check_proof(key, "1000." + "a" * 16, 1_003.0) is False  # forged signature
    assert check_proof(key, "", 1_003.0) is False                  # curl has nothing to echo
    assert check_proof(proof_key("other"), token, 1_003.0) is False


def test_the_throttle_cools_down_one_ip_and_caps_the_hour(smtp):
    throttle = ContactThrottle(cooldown_seconds=60, max_per_window=3, window_seconds=3_600)
    assert throttle.allow("1.2.3.4", 1_000.0) is True
    assert throttle.allow("1.2.3.4", 1_030.0) is False  # inside the cooldown
    assert throttle.allow("9.9.9.9", 1_030.0) is True   # another visitor is unaffected
    assert throttle.allow("1.2.3.4", 1_070.0) is True
    assert throttle.allow("1.2.3.4", 1_140.0) is True
    assert throttle.allow("1.2.3.4", 1_210.0) is False  # hour cap reached
    assert throttle.allow("1.2.3.4", 5_000.0) is True   # the window has rolled over


def test_the_throttle_caps_a_busy_ip_across_a_day():
    """The shipped limits: a two-minute cooldown, five an hour, twenty a day."""
    throttle = ContactThrottle(cooldown_seconds=0, max_per_window=99, max_per_day=3)
    for offset in (0, 10, 20):
        assert throttle.allow("8.8.8.8", 1_000.0 + offset) is True
    assert throttle.allow("8.8.8.8", 1_030.0) is False       # the day cap
    assert throttle.allow("8.8.8.8", 1_000.0 + 86_400) is True  # the day has rolled over


def test_the_throttle_table_stays_flat_under_a_flood():
    """A distributed flood must not grow memory: the table is bounded and pruned."""
    flood = ContactThrottle(max_ips=4)
    for index in range(50):
        flood.allow(f"9.9.{index}.1", 2_000.0)
    assert len(flood._hits) <= flood._max_ips + 1


def test_the_dm_target_needs_nothing_new_to_fill():
    """WhatsApp is the zero-config channel: the taps' owner numbers already say who to DM."""
    def quiet(chat, text):
        pass

    assert ContactNotifier(send_text=quiet, owner_numbers=(OWNER_NUMBER,)).contact_chat == OWNER
    assert ContactNotifier(send_text=quiet, owner_numbers=("351900000001", "351900000000")).contact_chat == OWNER
    assert ContactNotifier(send_text=quiet, contact_chat="120363000000000000@g.us",
                           owner_numbers=(OWNER_NUMBER,)).contact_chat == "120363000000000000@g.us"
    assert ContactNotifier(send_text=quiet).contact_chat == ""


def test_a_note_reaches_both_channels(smtp):
    notifier = ContactNotifier(send_text=lambda chat, text: None, contact_chat=OWNER,
                               smtp_host="smtp.private-domain-example.invalid",
                               email_to=SECRET_EMAIL)
    sent = notifier.notify("Alice", "visitor@elsewhere-example.invalid", "Can you help with a site?")
    assert sent == {"whatsapp": True, "email": True}, sent
    session = _sessions[-1]
    assert (session.host, session.port) == ("smtp.private-domain-example.invalid", 587)
    message = session.sent[0]
    assert message["To"] == SECRET_EMAIL and message["Subject"].startswith("Website contact from ")
    assert "visitor@elsewhere-example.invalid" in message.as_string()


def test_a_broken_channel_never_eats_the_other(smtp):
    def boom(chat, text):
        raise OSError("WAHA is down")

    notifier = ContactNotifier(send_text=boom, contact_chat=OWNER, smtp_host="smtp-host",
                               email_to=SECRET_EMAIL)
    assert notifier.notify("Alice", "visitor@elsewhere-example.invalid", "hi") == {
        "whatsapp": False, "email": True}

    def boom_send(session):
        raise OSError("smtp refused")

    class _BrokenSMTP(_FakeSMTP):
        def send_message(self, msg):
            raise OSError("smtp refused")

    notifier = ContactNotifier(send_text=lambda chat, text: None, contact_chat=OWNER,
                               smtp_host="smtp-host", email_to=SECRET_EMAIL)
    import src.gateway.contact as contact_mod
    original = contact_mod.smtplib.SMTP
    contact_mod.smtplib.SMTP = _BrokenSMTP
    try:
        assert notifier.notify("Alice", "visitor@elsewhere-example.invalid", "hi") == {
            "whatsapp": True, "email": False}
    finally:
        contact_mod.smtplib.SMTP = original


def test_an_unconfigured_channel_is_simply_skipped(smtp):
    whatsapp_only = ContactNotifier(send_text=lambda chat, text: None,
                                    owner_numbers=(OWNER_NUMBER,))
    assert whatsapp_only.notify("A", "v@e.co", "hi") == {"whatsapp": True, "email": False}
    assert _sessions == []

    email_only = ContactNotifier(send_text=lambda chat, text: None, contact_chat="",
                                 smtp_host="smtp-host", email_to=SECRET_EMAIL)
    assert email_only.notify("A", "v@e.co", "hi") == {"whatsapp": False, "email": True}

    nothing = ContactNotifier(send_text=lambda chat, text: None)
    assert nothing.notify("A", "v@e.co", "hi") == {"whatsapp": False, "email": False}


def test_the_thanks_page_says_nothing_about_the_owner():
    for lang in ("en", "pt"):
        page = render_thanks(lang)
        assert SECRET_EMAIL not in page and "@" not in page
        assert f'<html lang="{lang}">' in page
    assert "note is on its way" in render_thanks("en")
    assert "responde logo" in render_thanks("pt")


def test_the_public_app_relays_a_valid_note(rig, clock):
    gateway, client, sent = rig
    response = _browser(client, clock, **_note())
    assert response.status_code == 200
    assert "on its way" in response.text
    assert SECRET_EMAIL not in response.text
    assert sent == [(OWNER, "🌔 Website contact\n\nName: Alice\nEmail: visitor@elsewhere-example.invalid\n\nCan you help with a site?")]


def test_the_public_app_sends_nothing_for_a_bot(rig, clock):
    """Even a bot that scraped the page gets the thank-you: nothing tells it it was dropped."""
    _, client, sent = rig
    response = _browser(client, clock, **_note(), website="http://spam-example.invalid")
    assert response.status_code == 200, "a bot must not be tipped off"
    assert sent == []


def test_the_public_app_answers_a_half_note_with_400(rig, clock):
    _, client, sent = rig
    response = _browser(client, clock, name="Alice")
    assert response.status_code == 400
    assert sent == []


def test_the_public_app_silently_drops_a_runaway_ip(rig, clock):
    gateway, client, sent = rig
    first = _browser(client, clock, **_note())
    again = _browser(client, clock, **_note())
    assert (first.status_code, again.status_code) == (200, 200)
    assert len(sent) == 1, "the second submit from the same visitor was dropped"


def test_the_public_app_answers_in_the_visitor_language(rig, clock):
    _, client, _ = rig
    response = _browser(client, clock, headers={"accept-language": "pt-BR,pt;q=0.9"}, **_note())
    assert "responde logo" in response.text


def test_the_public_app_refuses_a_giant_or_off_body(rig):
    """The body is parsed by hand (no python-multipart in the Pi image), so it must be capped
    and a non-form submission must not sneak through."""
    _, client, sent = rig
    assert client.post("/contact", data={"name": "A" * 40_000}).status_code == 400
    response = client.post("/contact", json=_note())
    assert response.status_code == 200, "an off-body submit is dropped silently, like a bot's"
    assert sent == []


def test_the_public_app_drops_a_note_that_carries_no_page_token(rig):
    """A plain curl loop has nothing to echo back: it gets the thank-you and nothing is sent."""
    _, client, sent = rig
    response = client.post("/contact", data=_note())
    assert response.status_code == 200
    assert sent == []


def test_the_public_app_drops_a_submit_that_is_too_fast(rig, clock):
    """No human types a name, an email and a message in under three seconds."""
    _, client, sent = rig
    response = _browser(client, clock, pause=1.0, **_note())
    assert response.status_code == 200
    assert sent == []


def test_the_public_app_drops_a_page_saved_for_weeks(rig, clock):
    """A token replayed long after the page was handed out is stale, not a visitor."""
    _, client, sent = rig
    response = _browser(client, clock, pause=259_201.0, **_note())
    assert response.status_code == 200
    assert sent == []


def test_the_counters_only_live_on_the_internal_status(rig, clock):
    gateway, client, _ = rig
    _browser(client, clock, **_note())
    client.post("/contact", data=_note())
    _browser(client, clock, **_note(), website="http://spam-example.invalid")
    assert client.get("/status").status_code == 404, "the public listener never serves the counters"
    assert gateway.status()["contact"] == {"sent": 1, "dropped_proof": 1, "dropped_honeypot": 1,
                                           "dropped_throttle": 0, "rejected": 0}
