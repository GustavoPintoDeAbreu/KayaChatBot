"""The public showcase page's "Get in touch" form: a visitor's note reaches the owner,
over WhatsApp and/or email, and nothing else leaves (no owner address on the page, no send
for a bot or an empty submit)."""
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.contact import ContactNotifier, ContactThrottle, clean_contact_fields, is_honeypot, render_thanks

OWNER = "351900000000@c.us"
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
def rig(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"whatsapp": {"group": {"respond_on_mention": True}}}))
    landing = tmp_path / "landing.html"
    landing.write_text('<html><form class="contact-form" action="/contact" method="post"></form></html>')
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_LANDING_HTML": str(landing), "GATEWAY_WHITELIST": str(tmp_path / "none.json"),
        "GATEWAY_ASSETS_DIR": str(tmp_path / "assets"),
        "KAYA_RELAY_TOKEN": "relay",
        "GATEWAY_CONTACT_CHAT": OWNER,
        "CONTACT_SMTP_HOST": "smtp.private-domain-example.invalid", "CONTACT_SMTP_PORT": "587",
        "CONTACT_SMTP_USER": "", "CONTACT_SMTP_PASSWORD": "",
        "CONTACT_EMAIL_TO": SECRET_EMAIL, "CONTACT_EMAIL_FROM": "",
    })

    sent = []
    gateway = Gateway(settings, send_text=lambda chat, text, reply_to=None: sent.append((chat, text)))
    client = TestClient(gateway.public_app)
    return gateway, client, sent


def _note(name="Alice", email="visitor@elsewhere-example.invalid", message="Can you help with a site?"):
    return {"name": name, "email": email, "message": message}


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


def test_the_throttle_cools_down_one_ip_and_caps_the_hour(smtp):
    throttle = ContactThrottle(cooldown_seconds=60, max_per_window=3, window_seconds=3_600)
    assert throttle.allow("1.2.3.4", 1_000.0) is True
    assert throttle.allow("1.2.3.4", 1_030.0) is False  # inside the cooldown
    assert throttle.allow("9.9.9.9", 1_030.0) is True   # another visitor is unaffected
    assert throttle.allow("1.2.3.4", 1_070.0) is True
    assert throttle.allow("1.2.3.4", 1_140.0) is True
    assert throttle.allow("1.2.3.4", 1_210.0) is False  # hour cap reached
    assert throttle.allow("1.2.3.4", 5_000.0) is True   # the window has rolled over


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
    whatsapp_only = ContactNotifier(send_text=lambda chat, text: None, contact_chat=OWNER)
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


def test_the_public_app_relays_a_valid_note(rig):
    gateway, client, sent = rig
    response = client.post("/contact", data=_note())
    assert response.status_code == 200
    assert "on its way" in response.text
    assert SECRET_EMAIL not in response.text
    assert sent == [(OWNER, "🌔 Website contact\n\nName: Alice\nEmail: visitor@elsewhere-example.invalid\n\nCan you help with a site?")]


def test_the_public_app_sends_nothing_for_a_bot(rig):
    _, client, sent = rig
    response = client.post("/contact", data={**_note(), "website": "http://spam-example.invalid"})
    assert response.status_code == 200, "a bot must not be tipped off"
    assert sent == []


def test_the_public_app_answers_a_half_note_with_400(rig):
    _, client, sent = rig
    response = client.post("/contact", data={"name": "Alice"})
    assert response.status_code == 400
    assert sent == []


def test_the_public_app_silently_drops_a_runaway_ip(rig):
    gateway, client, sent = rig
    first = client.post("/contact", data=_note())
    again = client.post("/contact", data=_note())
    assert (first.status_code, again.status_code) == (200, 200)
    assert len(sent) == 1, "the second submit from the same visitor was dropped"


def test_the_public_app_answers_in_the_visitor_language(rig):
    _, client, _ = rig
    response = client.post("/contact", data=_note(), headers={"accept-language": "pt-BR,pt;q=0.9"})
    assert "responde logo" in response.text
