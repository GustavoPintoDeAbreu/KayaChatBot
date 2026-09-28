"""A sticker is a reaction, not a photo of whoever sent it.

2026-09-28: a meme of a man with his hands on his head arrived as image/webp,
took the photo path, was logged as "[Imagem: …]", and came back as "Essa cara de
desespero diz tudo sobre ti, Gil". The payload already says it is a sticker.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_whatsapp_adapter import GROUP, group_event, make_report_adapter
from src.chat.whatsapp_adapter import parse_waha_message


def sticker_event(sha="a" * 43 + "=", animated=True):
    event = group_event("")
    event["payload"]["_data"] = {"message": {"stickerMessage": {
        "fileSha256": sha, "isAnimated": animated}}}
    event["payload"]["media"] = {"url": "http://waha/s.webp", "mimetype": "image/webp"}
    return event


def test_a_sticker_is_parsed_as_one():
    msg = parse_waha_message(sticker_event("ab/c+d=="))
    assert msg.is_sticker and msg.sticker_animated
    assert msg.sticker_sha == "ab_c-d"


def test_a_photo_is_not_a_sticker():
    event = group_event("")
    event["payload"]["media"] = {"url": "http://waha/p.jpg", "mimetype": "image/jpeg"}
    assert not parse_waha_message(event).is_sticker


def test_a_sticker_is_logged_as_a_sticker_and_a_photo_as_an_image(tmp_path):
    adapter, _, _ = make_report_adapter(tmp_path)
    adapter.describe_image = lambda url, mimetype: "um homem com as mãos na cabeça"
    photo = group_event("")
    photo["payload"]["media"] = {"url": "http://waha/p.jpg", "mimetype": "image/jpeg"}

    adapter.handle_event(sticker_event())
    adapter.handle_event(photo)

    logged = "\n".join(path.read_text(encoding="utf-8")
                       for path in (tmp_path / "msglog").glob("*.jsonl"))
    assert "[Sticker: um homem com as mãos na cabeça]" in logged
    assert "[Imagem: um homem com as mãos na cabeça]" in logged


def test_the_prompts_say_a_sticker_is_not_the_sender():
    from src.config_loader import load_config

    config = load_config("config.yaml")
    modes = config["chat"]["modes"]
    for prompt in (config["data"]["system_prompt"], modes["banter"]["system_prompt"],
                   modes["mixed"]["system_prompt"]):
        assert "nunca é a pessoa que o mandou" in prompt
