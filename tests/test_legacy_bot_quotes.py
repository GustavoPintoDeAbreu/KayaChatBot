"""Quotes of the bot from before they were marked (2026-09-29) are the bot's, not the group's.

On 2026-11-19 "who is he?" came back with the job the bot had invented in August: the
replies to that answer quoted it, and the quotes were embedded as group memory. The
bot's messages are never logged, so an old unmarked quote that matches nothing a member
wrote in that chat is the bot's. Fictional members here.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import BOT_QUOTES_MARKED_FROM, Ingester, build_chunks
from src.data.message_log import MessageLog

AUG = int(datetime(2026, 8, 17, 12, tzinfo=timezone.utc).timestamp())
OCT = int(datetime(2026, 10, 17, 12, tzinfo=timezone.utc).timestamp())
CHAT = "g1@g.us"


def ingester(tmp_path, rows):
    log = MessageLog(base_dir=str(tmp_path))
    for n, (sender, text, ts, quote) in enumerate(rows):
        log.append(chat_id=CHAT, message_id=f"m{n}", sender=sender, text=text, timestamp=ts,
                   scope="shared", reply_to_id="q" if quote else "", reply_to_text=quote)
    obj = Ingester.__new__(Ingester)
    obj.log = log
    return obj, list(log.read("shared"))


def test_an_old_quote_of_nothing_a_member_wrote_is_the_bots(tmp_path):
    obj, rows = ingester(tmp_path, [
        ("Ana", "vamos jantar amanhã ao sítio do costume", AUG - 60, ""),
        ("Ana", "não sei quem é mais burro", AUG, "O Bruno trabalha na Acme e anda sempre de farda…"),
        ("Bruno", "eu vou", AUG + 60, "vamos jantar amanhã ao sítio do costume"),
        ("Bruno", "lol", OCT, "Uma frase que ninguém escreveu aqui no grupo"),
    ])
    is_bot = obj._legacy_bot_quotes("shared")
    assert [is_bot(r) for r in rows] == [False, True, False, False]
    assert OCT > BOT_QUOTES_MARKED_FROM > AUG


def test_its_text_never_reaches_a_chunk(tmp_path):
    obj, rows = ingester(tmp_path, [
        ("Ana", "não sei quem é mais burro", AUG, "O Bruno trabalha na Acme e anda sempre de farda…"),
    ])
    is_bot = obj._legacy_bot_quotes("shared")
    for row in rows:
        if is_bot(row):
            row["reply_to_bot"] = True
    chunks, _ = build_chunks(rows, "shared", settle_seconds=0, keep_bot_quote=lambda m: False)
    text = "\n".join(c["text"] for c in chunks)
    assert "Acme" not in text and "(a responder ao bot)" in text
