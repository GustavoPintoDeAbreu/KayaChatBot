# The Pi: Kaya's always-on front door

The GPU PC runs on a schedule (`config.yaml` → `power`), so anything that has to be
up all the time runs on the Raspberry Pi 5 (`pi5.local`, 192.168.1.238):

```
WhatsApp ──▶ WAHA (Pi) ──▶ gateway (Pi) ──journal──▶ POST /whatsapp/relay ──▶ kaya-prod (PC)
                               │   ▲                                            │
       internet ──▶ cloudflared (Pi) ──▶ gateway :8080   /  and  /status        │
                                   └──▶ PC :7860   /app  (Gradio, login)        │
                                                                                ▼
                               WAHA (Pi) ◀──────────── replies ──────── kaya-prod
```

| Container | What it does | Port |
|---|---|---|
| `kaya-gateway` | Journals every WhatsApp event in SQLite. Downloads media at once, because WAHA deletes it after 180 s. Forwards events to the PC one at a time, in order. Buffers them while the PC is off and replays them when it is back. Sends the offline reply. Serves `/` and `/status`. | 8080 public (tunnel only), 8088 LAN (PC only) |
| `kaya-waha` | WhatsApp (NOWEB), `noweb-arm-2026.8.2`, the version prod ran | 3000 LAN (PC only) |
| `kaya-cloudflared` | The public tunnel for `sigmakayachat.pt`, its only connector since 2026-09-25 | none |

WAHA and cloudflared sit behind compose profiles (`COMPOSE_PROFILES` in `.env`).
They are switched on during the cutover and never run in two places at once.

## What happens to a message

1. WAHA posts the event to `gateway:8088/waha/webhook`.
2. The gateway journals it. The UNIQUE key is event type plus message id, which
   also absorbs WAHA's habit of delivering every event twice.
3. It downloads any media, and decides whether the message is **addressed** to
   the bot. That uses `is_addressed` from `whatsapp_adapter`, the same function
   the PC uses.
4. **PC up:** the forwarder posts the event to `/whatsapp/relay`. The PC logs it,
   updates the session, and acks once that is on disk; it answers afterwards.
5. **PC off:** the event waits in the journal. If it is addressed to the bot, the
   PC has been unreachable for more than 90 s (or announced its shutdown), and
   this chat has not had one yet this offline period, the gateway replies once:
   *"Estou desligado agora. Volto às 07:00 e respondo-te nessa altura."*
6. **PC back:** the backlog replays in order, and the PC's memory reads as if it
   had been listening all night.
   - The newest addressed message per sender per chat is flagged `deferred_reply`.
     The PC answers it, quoting it, if it is younger than
     `whatsapp.deferred_reply.max_age_hours` (12).
   - The PC holds ingestion until the backlog has drained, because its timestamp
     watermark would skip older messages that arrive after newer ones.

"Host answers but the app does not" means a deploy, not an outage. The gateway
buffers silently, and a mention sent during a restart is still answered
afterwards. That case used to be lost.

Delivered events and their media are deleted after **7 days** (`/status` shows
the journal size). This is the one place outside the PC where group messages
live. It is on the LAN, never sent anywhere else, and the SSD is the Pi's only
disk.

## Deploying

```bash
scripts/deploy_pi.sh --init-env   # first time: writes the Pi's .env from ~/kaya-prod/.env
scripts/deploy_pi.sh              # afterwards: sync code, rebuild, restart
```

The script also installs two units on the Pi:

- `pc-wake.timer`: Wake-on-LAN at `power.wake_time` minus `wol_lead_minutes`.
- `kaya-firewall.service`: ports 3000 and 8088 accept the PC only (`DOCKER-USER`,
  since Docker-published ports skip `INPUT`).

Useful on the Pi (`pi5 ssh`):

```bash
cd ~/kaya-gateway/deploy/pi && docker compose ps
docker logs -f kaya-gateway
curl -s localhost:8088/status | python3 -m json.tool      # journal size, backlog, PC state
sqlite3 data/gateway/journal.sqlite3 "select state, count(*) from events group by state"
sudo systemctl start pc-wake.service                      # wake the PC now
```

## Cutover and rollback

The full order is in `DEPLOYMENT.md` → "The Pi edge". The two rules:

- **Never run two WAHA instances on the same session.** Stop the PC's first, then
  copy `data/waha`, then start the Pi's.
- **Tunnel rules must point at LAN IPs** while both connectors exist. The
  ingress is configured remotely and shared by every connector, so a rule naming
  a compose service would fail on whichever connector cannot resolve it.

Rollback:

1. Set `COMPOSE_PROFILES=` on the Pi and run `docker compose up -d --remove-orphans`.
2. Copy `data/waha` back to the PC.
3. Set `KAYA_EDGE=local` in `~/kaya-prod/.env` and run `scripts/deploy_prod.sh`.
4. Restore the old tunnel rules.
