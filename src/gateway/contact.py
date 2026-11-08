"""The public showcase page's "Get in touch" form: relay a visitor's note to the owner.

The page is static and its CSP allows ``form-action 'self'`` (see ``static_assets.page_csp``),
so a visitor's submit POSTs here at ``/contact`` on the gateway's public app. The owner's note
is forwarded two ways, both to the owner only: a WhatsApp text into his DM (the same WAHA sender
the taps use) and an email over SMTP. The owner's own address is a server-side secret
(``CONTACT_EMAIL_TO``), never on the page; the email in the form is the *visitor's*.

Stdlib only: the Pi gateway imports this and must stay torch-free (tests/gateway/test_torch_free.py).
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_NAME = 120
_MAX_EMAIL = 254
_MAX_MESSAGE = 2000


def clean_contact_fields(form) -> Optional[dict]:
    """The form's name/email/message, trimmed to their caps; None if any is missing.

    ``form`` is the posted ``application/x-www-form-urlencoded`` body. A browser enforces the
    required attributes, so a submit that arrives without one of these is not a real visitor.
    """
    name = str(form.get("name") or "").strip()[:_MAX_NAME]
    email = str(form.get("email") or "").strip()[:_MAX_EMAIL]
    message = str(form.get("message") or "").strip()[:_MAX_MESSAGE]
    if not name or not message:
        return None
    _, _, domain = email.partition("@")
    if not email or "@" not in email or not domain or "." not in domain:
        return None
    return {"name": name, "email": email, "message": message}


def is_honeypot(form) -> bool:
    """Whether the hidden field was filled — a bot, not a person. Treated as a silent no-op."""
    return bool(str(form.get("website") or "").strip())


class ContactThrottle:
    """A per-IP cooldown so one visitor cannot flood the owner's WhatsApp or email inbox.

    In-memory and single-process (the gateway is one asyncio worker), which is enough for a
    personal site: it stops a determined bot and a runaway auto-submitter, not a coordinated
    multi-IP swarm.
    """

    def __init__(self, *, cooldown_seconds: int = 60, max_per_window: int = 10,
                 window_seconds: int = 3600) -> None:
        self._cooldown = cooldown_seconds
        self._max_per_window = max_per_window
        self._window = window_seconds
        self._hits: dict = {}

    def allow(self, ip: str, now: float) -> bool:
        """Record a submission for ``ip`` and report whether it is within the limits.

        ``now`` is the gateway's clock (epoch seconds); an empty ``ip`` (no forwarded header)
        falls back to a shared bucket so the cap still applies.
        """
        key = ip or "?"
        recent = [t for t in self._hits.get(key, []) if now - t < self._window]
        last = recent[-1] if recent else None
        if last is not None and now - last < self._cooldown:
            return False
        if len(recent) >= self._max_per_window:
            return False
        recent.append(now)
        self._hits[key] = recent
        return True


class ContactNotifier:
    """Send one visitor's note to the owner over WhatsApp and/or email. Never raises.

    Both channels are optional and independent: an unconfigured channel is skipped (logged), so
    a half-wired deployment still delivers over whichever channel it has. A send failure on one
    never drops the other or crashes the request.
    """

    def __init__(self, *, send_text, contact_chat: str = "", smtp_host: str = "",
                 smtp_port: int = 587, smtp_user: str = "", smtp_password: str = "",
                 email_to: str = "", email_from: str = "") -> None:
        self._send_text = send_text
        self.contact_chat = contact_chat.strip()
        self.smtp_host = smtp_host.strip()
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_to = email_to.strip()
        self.email_from = (email_from or email_to).strip()

    def _send_email(self, name: str, email: str, message: str) -> None:
        """Send the note to the owner over SMTP (STARTTLS). Raises; notify swallows it."""
        msg = EmailMessage()
        msg["From"] = self.email_from
        msg["To"] = self.email_to
        msg["Subject"] = f"Website contact from {name}"
        msg.set_content(f"{email} wrote from the website:\n\n{message}")
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)

    def notify(self, name: str, email: str, message: str) -> dict:
        """Relay a cleaned note; returns which channels it reached. Best-effort, never raises."""
        sent = {"whatsapp": False, "email": False}
        if self.contact_chat:
            try:
                self._send_text(self.contact_chat, _contact_text(name, email, message))
                sent["whatsapp"] = True
            except Exception as exc:  # noqa: BLE001 — a lost WhatsApp must not lose the email
                logger.warning("contact: whatsapp send failed: %s", exc)
        if self.email_to and self.smtp_host:
            try:
                self._send_email(name, email, message)
                sent["email"] = True
            except Exception as exc:  # noqa: BLE001 — a lost email must not fail the request
                logger.warning("contact: email send failed: %s", exc)
        if not sent["whatsapp"] and not sent["email"]:
            logger.error("contact: no channel configured (GATEWAY_CONTACT_CHAT / CONTACT_SMTP_*), "
                         "note was not delivered")
        return sent


def _contact_text(name: str, email: str, message: str) -> str:
    """The WhatsApp body for a visitor's note."""
    return f"\U0001f314 Website contact\n\nName: {name}\nEmail: {email}\n\n{message}"


def render_thanks(lang: str = "en") -> str:
    """A tiny thank-you page the form lands on after a submit (no email, no infra detail)."""
    note = "Merci — ligo responde logo." if lang == "pt" else "Thanks — your note is on its way."
    return (f"<!doctype html><html lang=\"{lang}\"><head><meta charset=\"utf-8\">"
            f"<title>{note}</title>"
            "<style>body{margin:0;background:#141b23;color:#dcf6f0;"
            "font-family:system-ui,sans-serif;padding:3rem 1.5rem;text-align:center}"
            "a{color:#3edbe3}</style></head><body>"
            f"<p>{note}</p><p><a href=\"/\">Back to the site</a></p></body></html>")
