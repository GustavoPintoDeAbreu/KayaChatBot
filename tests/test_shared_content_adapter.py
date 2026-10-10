"""Links and stickers as they pass through the WhatsApp adapter.

The reading itself (fetching, extracting, summarising) is `src/chat/links.py`
and is tested there. These tests pin the wiring: what the parser picks out of a
NOWEB payload, what goes into the message the log and the model see, and that a
question about a link reaches the responder with the page attached.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter, parse_waha_message
from src.data.message_log import MessageLog

BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"
URL = "https://x.com/someone/status/2103729888096068029"

_ids = itertools.count(1)


def event(text, extended=None, sticker=None, mention=False, quoted_text=None, media=None):
    message = {}
    if extended is not None:
        message["extendedTextMessage"] = extended
    if sticker is not None:
        message["stickerMessage"] = sticker
    payload = {
        "id": f"sc{next(_ids)}", "from": GROUP, "participant": ALICE,
        "body": text, "notifyName": "Alice",
        "mentionedIds": [BOT_JID] if mention else [],
        "_data": {"message": message},
    }
    if media:
        payload["media"] = media
    if quoted_text is not None:
        payload["replyTo"] = {"participant": ALICE, "body": quoted_text, "id": "orig1"}
    return {"event": "message", "payload": payload}


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, message, speaker, recent_lines, scope=None, exclude_from=None,
                 link_context=""):
        self.calls.append({"message": message, "link_context": link_context})
        return "ok"


def make_adapter(tmp_path, read_links=None, archive_sticker=None):
    config = {"whatsapp": {"bot_jid": BOT_JID, "send_seen": False, "history_turns": 5,
                           "contacts": {ALICE: "Alice"},
                           "shared_chats": [GROUP]}}
    responder = Recorder()
    adapter = WhatsAppAdapter(
        responder, MockWahaClient(echo=False), config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "s"), max_lines=10),
        message_log=MessageLog(str(tmp_path / "log")),
        read_links=read_links, archive_sticker=archive_sticker)
    return adapter, responder


def logged_texts(tmp_path):
    return [line for path in (tmp_path / "log").glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]


def test_link_preview_parsed_from_extended_text():
    msg = parse_waha_message(event(URL, extended={
        "matchedText": URL, "title": "Malaghetto (@Malaghetto74) on X",
        "description": "A 'musa do Pingo Doce' a mostrar a tal matriz cultural"}))
    assert msg.link_preview["title"].startswith("Malaghetto")
    assert "Pingo Doce" in msg.link_preview["description"]


def test_preview_description_equal_to_url_is_dropped():
    msg = parse_waha_message(event(URL, extended={
        "matchedText": URL, "title": "observador.pt", "description": URL}))
    assert msg.link_preview["description"] == ""


def test_no_preview_without_matched_text():
    msg = parse_waha_message(event("olá", extended={"text": "olá"}))
    assert msg.link_preview == {}


def test_sticker_parsed_with_filename_safe_hash():
    msg = parse_waha_message(event("", sticker={
        "fileSha256": "pByTpQNIF/+kC8cfkac6LeApmUMguipO5sbUaBzVieM=", "isAnimated": True}))
    assert msg.is_sticker and msg.sticker_animated
    assert "/" not in msg.sticker_sha and "+" not in msg.sticker_sha
    assert not msg.sticker_sha.endswith("=")


def test_link_line_is_logged_with_the_message(tmp_path):
    seen = []

    def read_links(text, preview):
        seen.append((text, preview))
        return [("[Link: Um post (x.com) — resumo]", "Um post\ntexto do post")]

    adapter, _ = make_adapter(tmp_path, read_links=read_links)
    adapter.handle_event(event(f"olha isto {URL}", extended={
        "matchedText": URL, "title": "Um post", "description": "texto"}))
    assert seen and seen[0][1]["title"] == "Um post"
    assert any("[Link: Um post (x.com)" in line for line in logged_texts(tmp_path))


def test_addressed_link_reaches_the_responder_with_the_page(tmp_path):
    adapter, responder = make_adapter(
        tmp_path, read_links=lambda text, preview: [("[Link: T (x.com)]", "T\nconteúdo")])
    adapter.handle_event(event(f"@bot o que achas? {URL}", mention=True))
    assert responder.calls[-1]["link_context"] == "T\nconteúdo"


def test_reply_quoting_a_link_reads_the_quoted_link(tmp_path):
    calls = []

    def read_links(text, preview):
        calls.append(text)
        return [("[Link: T (x.com)]", "T\nartigo")] if "http" in text else []

    adapter, responder = make_adapter(tmp_path, read_links=read_links)
    adapter.handle_event(event("@bot o que achas?", mention=True, quoted_text=URL))
    assert responder.calls[-1]["link_context"] == "T\nartigo"


def test_failing_link_reader_does_not_lose_the_message(tmp_path):
    def read_links(text, preview):
        raise RuntimeError("boom")

    adapter, _ = make_adapter(tmp_path, read_links=read_links)
    adapter.handle_event(event(f"olha {URL}"))
    assert any(URL in line for line in logged_texts(tmp_path))


def test_sticker_is_archived(tmp_path):
    archived = []
    adapter, _ = make_adapter(tmp_path, archive_sticker=archived.append)
    adapter.handle_event(event("", sticker={"fileSha256": "a" * 44},
                               media={"url": "http://waha/f.webp", "mimetype": "image/webp"}))
    assert archived and archived[0].sticker_sha == "a" * 44


def test_a_sticker_is_labelled_as_a_sticker_not_a_photo(tmp_path):
    """"Essa cara de desespero diz tudo sobre ti, Gil": a meme read as his face."""
    adapter, _ = make_adapter(tmp_path)
    adapter.describe_image = lambda url, mimetype: "um homem com as mãos na cabeça"
    adapter.handle_event(event("", sticker={"fileSha256": "b" * 44},
                               media={"url": "http://waha/s.webp", "mimetype": "image/webp"}))
    adapter.handle_event(event("", media={"url": "http://waha/p.jpg",
                                          "mimetype": "image/jpeg"}))

    logged = logged_texts(tmp_path)
    assert any("[Sticker: um homem com as mãos na cabeça]" in line for line in logged)
    assert any("[Imagem: um homem com as mãos na cabeça]" in line for line in logged)


def test_the_prompts_say_a_sticker_is_not_the_sender():
    from src.config_loader import load_config

    config = load_config("config.yaml")
    modes = config["chat"]["modes"]
    for prompt in (config["data"]["system_prompt"], modes["banter"]["system_prompt"],
                   modes["mixed"]["system_prompt"]):
        assert "nunca é a pessoa que o mandou" in prompt
