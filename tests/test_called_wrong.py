"""Friends calling an answer wrong in words, and the owner's answer to the 👤 list.

2026-11-19: after the bot gave a member another member's job, a friend wrote
"holy allucinacao". Fictional members here.
Only a 👎 reaction was ever recorded, so it went nowhere.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_whatsapp_adapter import ALICE, BOT_JID, GROUP, dm_event, group_event, make_adapter

BOB = "351922222222@c.us"


def rig(tmp_path):
    adapter, client = make_adapter(tmp_path)
    adapter.config["chat"] = {"feedback": {"log_file": str(tmp_path / "fb.jsonl")}}
    adapter.contacts[BOB] = "Bob"
    adapter._remember_sent("bot-1", {"chat_id": GROUP, "user_text": "quem é este gajo?",
                                     "reply": "O Bruno trabalha na Acme.", "is_group": True,
                                     "speaker": "Bob", "sent_at": time.time()})
    return adapter, client


def logged(tmp_path):
    path = tmp_path / "fb.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_a_strong_call_out_after_an_answer_is_a_thumbs_down(tmp_path):
    adapter, _ = rig(tmp_path)
    adapter.ingest_event(group_event("holy allucinacao", sender=BOB, name="Bob"))
    rows = logged(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["rating"] == "down" and row["kind"] == "called_wrong"
    assert row["assistant_response"] == "O Bruno trabalha na Acme."
    assert row["comment"] == "holy allucinacao" and row["speaker"] == "Bob"


def test_a_weak_word_counts_only_as_a_reply_to_the_bot(tmp_path):
    adapter, _ = rig(tmp_path)
    adapter.ingest_event(group_event("mentira, o jogo acabou 2-1", sender=BOB, name="Bob"))
    assert logged(tmp_path) == []
    event = group_event("mentira", sender=BOB, name="Bob")
    event["payload"]["replyTo"] = {"id": "bot-1", "participant": BOT_JID, "body": "O Bruno trabalha na Acme."}
    adapter.ingest_event(event)
    assert len(logged(tmp_path)) == 1


def test_an_old_answer_or_ordinary_chat_is_not_called_wrong(tmp_path):
    adapter, _ = rig(tmp_path)
    adapter.sent_messages["bot-1"]["sent_at"] = time.time() - 3600
    adapter.ingest_event(group_event("holy allucinacao", sender=BOB, name="Bob"))
    adapter.ingest_event(group_event("amanhã jantar?", sender=BOB, name="Bob"))
    assert logged(tmp_path) == []


def test_his_quoted_answer_to_the_list_is_applied_and_answered(tmp_path):
    adapter, client = rig(tmp_path)
    adapter.maintainer = "Alice"
    seen = []
    adapter.owner_reply_hook = lambda text: seen.append(text) or "👤 Applied 1 fact(s)."
    event = dm_event("1 3")
    event["payload"]["replyTo"] = {"id": "list-1", "body": "👤 Kaya's weekly check\n\n1. …"}
    result = adapter.ingest_event(event)
    assert seen == ["1 3"]
    assert result["command"] == "bios"
    assert client.sent[-1]["text"] == "👤 Applied 1 fact(s)."


def test_only_the_maintainer_answers_the_list(tmp_path):
    adapter, _ = rig(tmp_path)
    adapter.maintainer = "Gustavo"
    seen = []
    adapter.owner_reply_hook = lambda text: seen.append(text) or "x"
    event = dm_event("all")
    event["payload"]["replyTo"] = {"id": "list-1", "body": "👤 Kaya's weekly check"}
    result = adapter.ingest_event(event)
    assert not (isinstance(result, dict) and result.get("command") == "bios")
    assert seen == []


def test_the_owner_is_reached_without_kaya_report_jid(tmp_path):
    adapter, _ = rig(tmp_path)
    adapter.maintainer = "Alice"
    adapter.report_jid = ""
    assert adapter.owner_jid() == ALICE
    adapter.report_jid = "351900000009@c.us"
    assert adapter.owner_jid() == "351900000009@c.us"
    adapter.report_jid, adapter.maintainer = "", "Nobody"
    assert adapter.owner_jid() == ""


def test_his_voice_answer_to_the_list_is_transcribed_first(tmp_path):
    adapter, client = rig(tmp_path)
    adapter.maintainer = "Alice"
    adapter.transcribe = lambda url, mime: "o número dois está errado, o resto está certo"
    seen = []
    adapter.owner_reply_hook = lambda text: seen.append(text) or "👤 Done: applied 1, dropped 1."
    event = dm_event("")
    event["payload"].update(media={"url": "http://waha/f.oga", "mimetype": "audio/ogg"})
    event["payload"]["replyTo"] = {"id": "list-1", "body": "👤 Kaya's weekly check"}
    result = adapter.ingest_event(event)
    assert seen == ["o número dois está errado, o resto está certo"]
    assert result["command"] == "bios" and client.sent[-1]["text"].startswith("👤 Done")
