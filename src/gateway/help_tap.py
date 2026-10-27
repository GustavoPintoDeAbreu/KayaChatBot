"""The owner's ``/help`` DM: a fixed command list answered by the Pi itself.

Unlike the idea, homelab and assistant taps, ``/help`` is handed to no inbox and
needs no PC: the reply is built here, so it works while the PC is off (and while
the assistant inbox is down). Like the other taps, a message it claims is never
journaled, never forwarded to the PC and never becomes "something the group said".

It claims, in DMs only: the owner's ``/help`` (any case). The full reference is
``docs/how-to-use-kaya.md``, rendered on the labwatch Help page; the link is
appended when ``GATEWAY_HELP_URL`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Set

from src.chat.whatsapp_adapter import InboundMessage, _phone_from_alt

HELP_TAG = "❓[help]"
HELP_COMMANDS = ("/help",)

# Fixed text, no link and no member names (the repo is public). reply_text()
# appends the Help-page link when GATEWAY_HELP_URL is set.
HELP_TEXT = (
    "Kaya — o que podes pedir\n"
    "\n"
    "No grupo:\n"
    "- /bug <texto> — reporta um problema\n"
    "- /feedback <texto> — dá uma sugestão\n"
    "- /clear — apaga o contexto recente da conversa\n"
    "- /aniversario <dia/mês> — regista o teu aniversário\n"
    '- "responde só em áudio" / "volta a responder por texto" — voz ou texto\n'
    '- "resume o que decidimos sobre X" — ponto de situação de um assunto\n'
    '- "faz uma sondagem com as opções" — cria uma votação\n'
    '- "quem está a ganhar a votação?" — resultado da sondagem\n'
    '- "quantas vezes é que X disse Y?" — conta mensagens\n'
    '- "que documento era aquele?" — pergunta sobre um PDF partilhado\n'
    "- pergunta sobre o mundo (sem nomes do grupo) — pesquisa na web\n"
    "\n"
    "Por DM (só tu):\n"
    "- /idea <texto> — ideia para o pipeline\n"
    "- /homelaboff · /homelabon · /homelabrc [home] — ligar/desligar o PC\n"
    "- /help — esta lista"
)


@dataclass
class HelpTap:
    """Claims the owner's ``/help`` and answers with fixed text, no PC involved."""

    help_url: str
    owner_numbers: Set[str]

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Optional["HelpTap"]:
        """The tap, or None when no owner number is set (the feature is off).

        The owner is the idea pipeline's (``IDEA_OWNER_NUMBERS``) unless
        ``HELP_OWNER_NUMBERS`` says otherwise, the same convention as the
        assistant tap.
        """
        raw = environ.get("HELP_OWNER_NUMBERS") or environ.get("IDEA_OWNER_NUMBERS", "")
        owners = {_phone_from_alt(number.strip().lstrip("+")) for number in raw.split(",") if number.strip()}
        if not owners:
            return None
        return cls(environ.get("GATEWAY_HELP_URL", "").rstrip("/"), owners)

    def _is_owner(self, msg: InboundMessage) -> bool:
        candidates = {_phone_from_alt(msg.sender_phone), _phone_from_alt(msg.sender_id)}
        return bool((candidates - {""}) & self.owner_numbers)

    def classify(self, msg: InboundMessage) -> Optional[str]:
        """``"help"`` to answer, ``"echo"`` to drop, None to leave for Kaya."""
        if msg.is_group:
            return None
        text = (msg.text or "").strip()
        if msg.from_me:
            return "echo" if text.startswith(HELP_TAG) else None
        if not self._is_owner(msg):
            return None
        first = text.split(maxsplit=1)[0].lower() if text else ""
        return "help" if first in HELP_COMMANDS else None

    def reply_text(self) -> str:
        """The fixed command list, plus the Help-page link when one is configured."""
        if self.help_url:
            return f"{HELP_TEXT}\n\n📖 Página de ajuda: {self.help_url}"
        return HELP_TEXT
