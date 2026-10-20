"""Kaya's groups for the lab's agents, read-only, at ``POST /mcp``.

Gustavo discusses what he builds with the friends in the groups ("Kaya bot", general…),
and then tells his lab assistant (``~/Desktop/lab-assistant``) "check the kaya bot chat"
for the context. On 2026-10-19 the assistant had no way to read a chat, took the mention
as a reason to hand the whole voice note to Kaya, and Kaya answered with group gossip.
This gives it the chats instead: the group log Kaya already keeps
(``data/live_messages/shared.jsonl``, see ``src/data/message_log.py``) and the group
names (``data/whatsapp_chat_names.json``, see ``channels.py``).

Groups only: DM files are never opened. Nothing here writes.

Not the MCP SDK: the lab's clients speak the stateless JSON form of streamable HTTP
(one JSON-RPC POST per call, no session), which is three methods, while the SDK would
bring httpx2, opentelemetry and pyjwt into this image and clash with gradio's own
``mcp<2`` extra. ``X-Kaya-Token`` must match ``KAYA_MCP_TOKEN``; unset, the route is off
(503), so the friends' messages never leave without it.
"""
from __future__ import annotations

import hmac
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.chat.channels import ChannelNames
from src.data.message_log import MessageLog

PROTOCOL = "2025-06-18"
MAX_HOURS = 24 * 7
MAX_LIMIT = 200
TEXT_CHARS = 1000

INSTRUCTIONS = """\
Kaya's WhatsApp groups (the friends' WhatsApp Community: general, the bot's own group, trips…), \
read-only. Gustavo shares what he builds there and gets the friends' reactions and suggestions; \
when he points at a chat for context, read it with read_chat and use only what bears on his request. \
Anything that leaves this conversation (an idea, a note, a PR) gets a short summary of the friends' \
points: no names, no quotes, nothing off-topic. Voice notes and stickers appear as their transcript \
or description. Kaya's own replies are not in the log."""

try:
    from zoneinfo import ZoneInfo
    _LISBON = ZoneInfo("Europe/Lisbon")
except Exception:  # noqa: BLE001
    _LISBON = timezone.utc

_FILLER = {"the", "chat", "group", "channel", "o", "a", "grupo", "canal", "conversa"}

READ = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}

TOOLS = [
    {"name": "chats",
     "description": "Kaya's groups by name: when each last had a message, and how many in the last 24 h.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": READ},
    {"name": "read_chat",
     "description": "The recent messages of one group, oldest first: time (Lisbon), sender, text. chat: part "
                    "of the group's name ('kaya bot', 'general', 'viagem') or its id. hours: how far back "
                    "(default 24, at most 168). contains: only messages with this text (any case). limit: "
                    "the newest N in the window (default 80, at most 200).",
     "inputSchema": {"type": "object",
                     "properties": {"chat": {"type": "string"},
                                    "hours": {"type": "number", "default": 24},
                                    "contains": {"type": "string", "default": ""},
                                    "limit": {"type": "integer", "default": 80}},
                     "required": ["chat"]},
     "annotations": READ},
    {"name": "status",
     "description": "Whether the group log is being written: its last message and the groups it knows.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": READ},
]


class ChatLog:
    """The group log and the group names, as the tools see them."""

    def __init__(self, log_dir: Path, names_file: Path, now: Callable[[], float] = time.time) -> None:
        self.log = MessageLog(str(log_dir))
        self.names = ChannelNames(Path(names_file))
        self._now = now

    def _rows(self, since: float) -> List[Dict[str, Any]]:
        return [row for row in self.log.read("shared", after_ts=int(since))
                if str(row.get("chat_id", "")).endswith("@g.us")]

    def _groups(self, rows: List[Dict[str, Any]]) -> Dict[str, str]:
        """chat id -> name, for every named group plus any group seen in ``rows``."""
        data = self.names._load()
        ids = set(data.get("names") or {}) | set(data.get("labels") or {})
        ids |= {row["chat_id"] for row in rows}
        return {chat_id: self.names.label(chat_id) or chat_id for chat_id in ids if chat_id.endswith("@g.us")}

    def chats(self) -> Dict[str, Any]:
        rows = self._rows(0)
        last: Dict[str, int] = {}
        day: Dict[str, int] = {}
        cutoff = self._now() - 86400
        for row in rows:
            ts = int(row.get("timestamp") or 0)
            last[row["chat_id"]] = max(last.get(row["chat_id"], 0), ts)
            if ts > cutoff:
                day[row["chat_id"]] = day.get(row["chat_id"], 0) + 1
        out = [{"chat": name, "id": chat_id, "last_message": _when(last.get(chat_id)),
                "messages_24h": day.get(chat_id, 0)}
               for chat_id, name in self._groups(rows).items()]
        out.sort(key=lambda entry: entry["last_message"] or "", reverse=True)
        return {"chats": out}

    def read_chat(self, chat: str, hours: float = 24, contains: str = "", limit: int = 80) -> Dict[str, Any]:
        hours = min(max(float(hours or 24), 0.1), MAX_HOURS)
        limit = min(max(int(limit or 80), 1), MAX_LIMIT)
        since = self._now() - hours * 3600
        rows = self._rows(since)
        chat_id, error = self._pick(chat, self._groups(self._rows(0)))
        if error:
            return {"error": error}
        found = [row for row in rows if row["chat_id"] == chat_id
                 and (not contains or contains.lower() in str(row.get("text") or "").lower())]
        kept = found[-limit:]
        return {"chat": self.names.label(chat_id) or chat_id, "hours": hours,
                "messages": [{"time": _when(int(row.get("timestamp") or 0)), "sender": row.get("sender") or "?",
                              "text": _clip(str(row.get("text") or ""))} for row in kept],
                "older_in_window": len(found) - len(kept)}

    def status(self) -> Dict[str, Any]:
        rows = self._rows(0)
        newest = max((int(row.get("timestamp") or 0) for row in rows), default=0)
        return {"log_last_message": _when(newest), "groups": len(self._groups(rows)),
                "messages_24h": sum(1 for row in rows if int(row.get("timestamp") or 0) > self._now() - 86400)}

    @staticmethod
    def _pick(ref: str, groups: Dict[str, str]):
        """One group by id or by a case-insensitive piece of its name; a name equal to the
        reference wins over names that merely contain it."""
        want = " ".join(word for word in str(ref or "").lower().split() if word not in _FILLER)
        if ref in groups:
            return ref, None
        exact = [chat_id for chat_id, name in groups.items() if name.lower() == want]
        hits = exact or [chat_id for chat_id, name in groups.items() if want and want in name.lower()]
        if len(hits) == 1:
            return hits[0], None
        known = "; ".join(sorted(groups.values()))
        if not hits:
            return None, f"no group matches {ref!r}; known: {known}"
        return None, f"{ref!r} matches several: " + "; ".join(sorted(groups[chat_id] for chat_id in hits))


def handle(chats: ChatLog, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, its answer out (None for a notification)."""
    method, params, rid = message.get("method"), message.get("params") or {}, message.get("id")
    if rid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "kaya", "version": "1"}, "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = _call(chats, params.get("name"), params.get("arguments") or {})
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _call(chats: ChatLog, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    tools = {"chats": chats.chats, "read_chat": chats.read_chat, "status": chats.status}
    if name not in tools:
        return _result({"error": f"unknown tool {name}"}, error=True)
    try:
        value = tools[name](**args)
    except TypeError as exc:
        return _result({"error": f"bad arguments: {exc}"}, error=True)
    return _result(value, error="error" in value)


def _result(value: Dict[str, Any], error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
            "structuredContent": {"result": value}, "isError": error}


def router(chats: ChatLog, token: Callable[[], str]) -> APIRouter:
    """``POST /mcp`` (``GET`` is 405: there is no event stream)."""
    api = APIRouter()

    @api.post("/mcp", include_in_schema=False)
    async def mcp(request: Request, x_kaya_token: str = Header(default="")):
        expected = token()
        if not expected:
            raise HTTPException(status_code=503, detail="chats MCP disabled (KAYA_MCP_TOKEN unset)")
        if not hmac.compare_digest(x_kaya_token.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid token")
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700, "message": "parse error"}}, status_code=400)
        if isinstance(body, list):
            answers = [answer for answer in [await run_in_threadpool(handle, chats, item) for item in body] if answer]
            return JSONResponse(answers) if answers else Response(status_code=202)
        answer = await run_in_threadpool(handle, chats, body)
        return JSONResponse(answer) if answer else Response(status_code=202)

    return api


def _when(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, _LISBON).strftime("%Y-%m-%d %H:%M")


def _clip(text: str) -> str:
    return text if len(text) <= TEXT_CHARS else text[:TEXT_CHARS - 1] + "…"
