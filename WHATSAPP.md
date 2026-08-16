# WhatsApp Bridge

Make the Kaya bot reachable on WhatsApp as a **DM bot** and as a **group
participant** (replies only when @-mentioned or replied to). Self-hosted: a
**WAHA** container links a dedicated number and forwards messages over a webhook
to a Python bridge that runs in the same process as the existing Gradio UI, so
the model is loaded **once**, on the prod GPU.

```
WhatsApp ⇄ WAHA container (Node, Docker)  ──webhook POST──▶  whatsapp_server.py
                  ▲                                            ├─ WhatsAppAdapter (routing)
                  └────────── sendText (reply) ────────────────┤─ KayaEngine  (shared model + RAG)
                                                               └─ Gradio UI (mounted at /app, password-gated;
                                                                  / is the public landing page)
```

## Components

| File | Role |
|---|---|
| `src/chat/engine.py` | Single model load (`get_engine`) + non-streaming `generate_reply` + `build_system_prompt`. Shared by the web UI and the bridge. |
| `src/chat/whatsapp_adapter.py` | Parses WAHA webhooks, decides whether to reply (DM gated by the whitelist; group on mention/reply), resolves the speaker, manages per-chat history. |
| `src/chat/waha_client.py` | `WahaClient` (real REST) and `MockWahaClient` (captures replies — used with no real number). |
| `src/chat/memory.py` → `KeyedSessionMemory` | Per-chat rolling history under `data/whatsapp_sessions/`. |
| `src/chat/whatsapp_server.py` | FastAPI webhook + mounts the Gradio UI; run this instead of `web_app.py` when WhatsApp is on. |
| `scripts/whatsapp_simulator.py` | Interactive REPL that fakes inbound messages — develop/test with no number. |

## Develop & test with NO phone number (mock)

The whole flow runs without a number, a GPU, or WAHA:

```bash
# Routing / mention / reply / history logic — fake responder, instant:
kaya_chatbot_env/bin/python scripts/whatsapp_simulator.py
#   /dm <name>, /group <name>, /mention, /reply, /quit; '@kaya' also addresses the bot

# Same flow with the real fine-tuned model (needs the GPU free):
kaya_chatbot_env/bin/python scripts/whatsapp_simulator.py --real

# Unit tests:
kaya_chatbot_env/bin/python -m pytest tests/test_whatsapp_adapter.py -v
```

Run the **real server** in mock mode (captures outbound instead of sending):

```bash
KAYA_WHATSAPP_MOCK=1 kaya_chatbot_env/bin/python -m src.chat.whatsapp_server
# POST a fake WAHA event, then read what it 'sent':
curl -s localhost:7860/whatsapp/webhook -H 'Content-Type: application/json' \
  -d '{"event":"message","payload":{"id":"1","from":"3519xxx@c.us","body":"olá","notifyName":"Gustavo"}}'
curl -s localhost:7860/whatsapp/outbox
```

## Go live (when the dedicated number is ready)

1. **Get a dedicated number** (real prepaid SIM or an eSIM that receives SMS/voice
   OTP — many free VoIP numbers are rejected by WhatsApp). Register WhatsApp on it
   from a phone first. **Never use your personal number** (ban risk).
2. Set in `.env`: `KAYA_WAHA_API_KEY`, `KAYA_WHATSAPP_WEBHOOK_TOKEN`.
3. In `config.yaml` set `whatsapp.enabled: true`, `whatsapp.mock_mode: false`,
   `whatsapp.bot_jid: "<botnumber>@c.us"`, and fill `whatsapp.contacts` with
   `"<phone>@c.us": "Member name"` so the model knows who is speaking.
4. Start WAHA and link the device:
   ```bash
   docker compose --profile whatsapp up -d waha
   # open http://localhost:3000, start the "default" session, scan the QR with the bot's phone
   ```
5. Run the bridge process (serves UI + webhook). `kaya-prod` and `kaya-dev`
   already ship `command: python -m src.chat.whatsapp_server` in
   `docker-compose.yml`, so there is nothing to change.
6. Test: DM the bot → reply. Add it to a group, `@`-mention it → reply; send
   unrelated chatter → silence; reply to its message → reply.

## Commands

Literal slash commands, matched before the message ever reaches the model. In a
group they still need an @-mention like anything else.

| Command | Aliases | What it does |
|---|---|---|
| `/clear` | `/limpar` | Forget this chat's recent verbatim context |
| `/bug <what happened>` | `/erro` | File a bug report |
| `/feedback <your idea>` | `/sugestao`, `/sugestão` | Leave a suggestion |

Most of what the group actually uses is not a slash command at all — the router
reads it out of ordinary language (`src/chat/router.py`):

| Asked like | Does |
|---|---|
| "responde-me só em áudio" / "volta a escrever" | switches this chat to voice or text, until changed |
| "explica isso num áudio" | answers *this one* by voice, without changing the default |
| "põe o Rafa vestido de rei" / "faz uma imagem de um gato astronauta" | edits the attached (or last) photo, or invents one |
| "esquece o que falámos" | same as `/clear` |
| "quantas vezes é que o Rafa disse isso?" | **counts** it |

That last one does not go through retrieval. Semantic search returns the chunks
nearest a question and cannot answer "how many times", so `CMD_COUNT` scans the
logs directly (`src/chat/tally.py`) and hands the model a finished table to
phrase. Two consequences worth knowing before someone reports them as bugs:

- **Counts are scope-isolated.** A DM tallies only its own log; the group tallies
  the group's. The same question therefore gives different numbers in a DM and in
  the group, on purpose.
- **Only the group's count includes the pre-bot export**
  (`data/all_messages_cleaned.jsonl`) — the group is years older than the bot, and
  a count starting at the install date is the wrong answer.

`/bug` and `/feedback` take the rest of the message as the body; sent bare, they
reply with usage and store nothing (no pending-capture state, so an unrelated
next message can never be swallowed into someone's report). `/bug` lands in
`data/feedback/bug_reports.jsonl`, `/feedback` writes a `type: note` record into
`message_feedback.jsonl`; both show in the web UI's **Feedback** tab. Set
`KAYA_REPORT_JID` in `.env` to have each new report announced by DM.

**Commands are never stored as memory.** The message log — the thing that gets
embedded into ChromaDB — is written *before* the reply gate, so
`whatsapp_adapter._is_command` excludes them explicitly. Without that a week of
bug reports would come back out of retrieval as things the group said.

## `shared_chats` is the setting that breaks quietly

`data/whatsapp_shared_chats.json` (gitignored) lists the chats whose content is
**group-wide** memory. Everything else gets `group:<hash>` or `dm:<hash>` and is
private to itself. Two things hang off it, and both fail *silently* if it names
the wrong chat:

- **Image editing** used to be gated on `chat.imagegen.allowed_scopes:
  ["shared"]`, so a group missing from `whatsapp_shared_chats.json` was told
  *"Só faço imagens no grupo, não por aqui"* — while standing in the group.
  Consent and memory scope are different questions, and `allowed_here` now
  answers them separately, most specific first: `allowed_chats` → `allow_groups`
  (true today: any group, whatever its memory scope) → `allowed_scopes` as the
  fallback. So this no longer fails, but the scope file still drives retrieval:
- **Retrieval asymmetry.** A DM may recall anything the group said; a non-shared
  group's history is invisible from DMs. Nobody notices, because the group can
  still read its own messages *and* the historical export (which is `shared`), so
  answers look right from inside the group.

It happened: the file listed the 3-person *Testing Kaya* group instead of the real
16-person one, so 184 live messages and 25 chunks accumulated under
`group:72fcd308f7383d08`. Fixing it needs a **restart** (the file is read once at
import) plus a migration, because `chunk_uid` hashes the scope into the id — the
old chunks can never be upserted, only deleted and rebuilt:

```bash
# 1. back up data/live_messages, ingest_state.json and data/rag_db
# 2. fix whatsapp_shared_chats.json in BOTH ~/kaya-prod/data and the dev copy
# 3. merge live_messages/group_<hash>.jsonl into shared.jsonl, scope rewritten
# 4. delete the live_ chunks for the old scope (never chunk_/img_/exp_/aud_ —
#    those are the historical export)
# 5. set the shared watermark in ingest_state.json to 0
# 6. restart; ingest.on_boot rebuilds the chunks under the right scope
```

After adding the bot to a **new** group, check `docker logs kaya-prod | grep
shared-memory` and confirm the count matches what you expect.

## Behaviour & limits

- **DM:** answered only for numbers in the anti-spam whitelist when
  `whatsapp.whitelist.enabled` is true (the default) — every other DM is silently
  ignored so a leaked number can't be spammed. Numbers live in the gitignored
  `data/whatsapp_whitelist.json` (`{"allowed": ["351…", …]}`), merged into
  `whatsapp.whitelist.allowed` at startup; edit that file + `docker restart kaya-prod`
  to change who can DM. Set `whitelist.enabled: false` to answer every DM.
  **Group:** answers only on @-mention or reply-to-bot (`whatsapp.group.*`),
  regardless of the DM whitelist; never answers itself.
- **Single loaded model, two surfaces:** `kaya-prod` runs `whatsapp_server`, which
  serves both the WhatsApp webhook and the mounted Gradio UI off one `get_engine()`
  instance — so both honour the active inference backend (prod = gguf; see CLAUDE.md).
- **Ban/ToS risk:** WAHA is unofficial (drives WhatsApp Web). Use the dedicated
  number, keep volume modest, leave `send_seen` on for human-like pacing.
- **Single GPU:** WhatsApp generations share the in-process `gpu_lock` with the web
  UI. If the GPU is busy past the timeout the message is dropped (logged), not queued.
- **Offline:** if the bridge is down, WAHA can't deliver webhooks and messages go
  unanswered (the bridge ignores backlog on restart to avoid replying to stale msgs).
- **Privacy:** in a group, every member's messages pass through the bot and are
  written to `data/live_messages/<scope>.jsonl` (the memory the vector store is
  built from), with per-chat history in `data/whatsapp_sessions/`. Answered turns
  also land in `data/feedback/live_interactions.jsonl`, alongside `/bug` and
  `/feedback` reports. Tell the group.
```


## Troubleshooting

### The bot answers on the web UI but not on WhatsApp

Check these in order — the first two look identical from the outside but have
completely different fixes.

**1. Is the WAHA image stale?** This is the most likely cause and the least
obvious. `docker-compose.yml` pins the floating `devlikeapro/waha:latest` tag,
which only advances when something actually pulls it. WAHA tracks WhatsApp's
protocol, which changes often, so an image left alone for a month or two stops
being able to complete the handshake. The symptom is a session that never leaves
`STARTING` and logs a loop of:

```
session:default - connected to WA
session:default - logging in...
session:default - connection errored  Error: Connection Failure
session:default - Session stuck in STARTING status, force stopping the session.
```

That reads exactly like an expired login and tempts a pointless QR re-scan. It is
not — the stored credentials are fine. Fix:

```bash
docker pull devlikeapro/waha:latest
cd ~/kaya-prod && docker compose --profile prod up -d --force-recreate waha
```

`deploy_prod.sh` now pulls on every deploy so this cannot silently rot again.

**2. Is it still replaying history?** After any reconnect, WAHA re-syncs and can
push *thousands* of old messages through the webhook at ~10/second. New messages
queue behind that flood, so the bot looks dead for minutes. `adapter.ignore_before_ts`
(set to process start in `whatsapp_server.py`) makes the bot skip all of them
rather than spam the group with days-old replies. Watch it drain:

```bash
docker logs kaya-waha 2>&1 | grep -c WebhookSender   # rising fast = still syncing
curl -s -H "X-Api-Key: $KEY" http://127.0.0.1:3000/api/sessions   # want status WORKING
```

**3. Is the sender whitelisted?** DMs are answered only for numbers in
`data/whatsapp_whitelist.json`. Note that prod has its **own** `data/` directory
(not a symlink — it holds the WAHA session), so its whitelist can drift from the
dev copy. Compare them.

**4. Still nothing?** Turn on raw event logging, send one message, then turn it
back off — it logs full message bodies, so do not leave it running:

```bash
echo 'KAYA_WHATSAPP_DEBUG=1' >> ~/kaya-prod/.env
cd ~/kaya-prod && docker compose --profile prod up -d --force-recreate kaya-prod
docker logs kaya-prod 2>&1 | grep wpp-debug | tail -2
```

Useful detail when reading those events: senders now arrive **@lid-addressed**
(`64622145081581@lid`) rather than as a phone number. The whitelist still matches
because NOWEB also sends the real phone in `_data.key.participantAlt` — so grep
for the `@lid`, not the phone, when tracing one person's messages.
