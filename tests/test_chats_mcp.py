"""Kaya's groups as a read-only MCP server for the lab assistant.

On 2026-10-19 Gustavo told the assistant to "check the kaya bot channel" for the
friends' reactions to a video; it could not read a chat, handed the whole voice note
to Kaya, and Kaya answered with gossip. These pin what the assistant now reads
instead: groups only, by a piece of the name, a bounded window, behind a token.
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import chats_mcp

NOW = 1_792_405_800
BOT = "100000000000000001@g.us"
GENERAL = "100000000000000002@g.us"
TRIP = "100000000000000003@g.us"


def _log(tmp_path, rows, broken=False):
    directory = tmp_path / "live_messages"
    directory.mkdir(exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    if broken:
        lines.insert(1, '{"id": "cut", "chat_id": "')
    (directory / "shared.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "dm_0000000000000000.jsonl").write_text(json.dumps(
        {"chat_id": "200000000000001@lid", "sender": "Gustavo", "text": "private", "timestamp": NOW - 60}) + "\n")
    names = tmp_path / "whatsapp_chat_names.json"
    names.write_text(json.dumps({"names": {BOT: "Kaya bot", GENERAL: "Kaya general",
                                           TRIP: "Viagem"}}), encoding="utf-8")
    return chats_mcp.ChatLog(directory, names, now=lambda: NOW)


def _row(chat_id, minutes_ago, text, sender="Gustavo"):
    return {"chat_id": chat_id, "sender": sender, "text": text, "timestamp": NOW - minutes_ago * 60}


def _chats(tmp_path):
    return _log(tmp_path, [
        _row(GENERAL, 60 * 30, "old news"),
        _row(BOT, 20, "needs better ambient noise"),
        _row(BOT, 10, "Not bad at all", sender="Friend"),
        _row(GENERAL, 5, "jantar hoje?"),
        _row("200000000000001@lid", 4, "a DM that slipped into shared"),
    ], broken=True)


def test_read_chat_by_a_piece_of_the_name_oldest_first(tmp_path):
    result = _chats(tmp_path).read_chat("the kaya bot channel")
    assert result["chat"] == "Kaya bot"
    assert [m["text"] for m in result["messages"]] == ["needs better ambient noise",
                                                       "Not bad at all"]
    assert result["messages"][1]["sender"] == "Friend"


def test_a_window_a_limit_and_a_filter(tmp_path):
    chats = _chats(tmp_path)
    assert chats.read_chat("general")["messages"] == [
        {"time": chats_mcp._when(NOW - 300), "sender": "Gustavo", "text": "jantar hoje?"}]
    assert len(chats.read_chat("general", hours=48)["messages"]) == 2
    limited = chats.read_chat("kaya bot", limit=1)
    assert [m["text"] for m in limited["messages"]] == ["Not bad at all"]
    assert limited["older_in_window"] == 1
    assert [m["sender"] for m in chats.read_chat("bot", contains="NOISE")["messages"]] == ["Gustavo"]


def test_an_unknown_or_ambiguous_name_lists_the_groups(tmp_path):
    chats = _chats(tmp_path)
    assert "known: " in chats.read_chat("football")["error"]
    assert "matches several" in chats.read_chat("kaya")["error"]


def test_groups_only(tmp_path):
    chats = _chats(tmp_path)
    listed = chats.chats()["chats"]
    assert {entry["id"] for entry in listed} == {BOT, GENERAL, TRIP}
    assert listed[0]["chat"] == "Kaya general"
    assert {entry["chat"]: entry["messages_24h"] for entry in listed}["Kaya bot"] == 2
    assert "error" in chats.read_chat("200000000000001@lid")


def test_status(tmp_path):
    status = _chats(tmp_path).status()
    assert status["groups"] == 3 and status["messages_24h"] == 3
    assert status["log_last_message"] == chats_mcp._when(NOW - 300)


def _client(tmp_path, token="secret"):
    app = FastAPI()
    app.include_router(chats_mcp.router(_chats(tmp_path), lambda: token))
    return TestClient(app)


def _rpc(client, method, params=None, token="secret", rid=1):
    return client.post("/mcp", json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}},
                       headers={"X-Kaya-Token": token})


def test_the_mcp_route_lists_and_calls_read_only_tools(tmp_path):
    client = _client(tmp_path)
    init = _rpc(client, "initialize", {"protocolVersion": "2025-06-18"}).json()["result"]
    assert "no names, no quotes" in init["instructions"]
    tools = _rpc(client, "tools/list").json()["result"]["tools"]
    assert {tool["name"] for tool in tools} == {"chats", "read_chat", "status"}
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
    called = _rpc(client, "tools/call", {"name": "read_chat", "arguments": {"chat": "kaya bot"}}).json()["result"]
    assert called["isError"] is False
    assert called["structuredContent"]["result"]["chat"] == "Kaya bot"
    missing = _rpc(client, "tools/call", {"name": "read_chat", "arguments": {"chat": "football"}}).json()["result"]
    assert missing["isError"] is True


def test_a_notification_is_accepted_without_a_body(tmp_path):
    client = _client(tmp_path)
    response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                           headers={"X-Kaya-Token": "secret"})
    assert response.status_code == 202


def test_no_token_no_chats(tmp_path):
    assert _rpc(_client(tmp_path), "tools/list", token="wrong").status_code == 401
    assert _rpc(_client(tmp_path), "tools/list", token="").status_code == 401
    assert _rpc(_client(tmp_path, token=""), "tools/list", token="").status_code == 503
