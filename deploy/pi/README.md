# The Pi: Kaya's always-on front door

The GPU PC runs on a schedule (lab-infra `power.yaml`), so anything that has to be
up all the time runs on the Raspberry Pi 5 (`pi5.local`, 192.168.1.238):

```
WhatsApp ──▶ WAHA (Pi) ──▶ gateway (Pi) ──journal──▶ POST /whatsapp/relay ──▶ kaya-prod (PC)
                               │   ▲                                            │
       internet ──▶ cloudflared (Pi) ──▶ gateway :8080   /  and  /status        │
                                                                                ▼
                               WAHA (Pi) ◀──────────── replies ──────── kaya-prod
```

| Container | What it does | Port |
|---|---|---|
| `kaya-gateway` | Journals every WhatsApp event in SQLite. Downloads media at once, because WAHA deletes it after 180 s. Forwards events to the PC one at a time, in order. Buffers them while the PC is off and replays them when it is back. Sends the offline reply. Serves the showcase page `/`, its images and scripts `/assets/<name>`, `POST /contact` (the page's "Get in touch" form) and `/status`. | 8080 public (tunnel only), 8088 LAN (PC only) |
| `kaya-waha` | WhatsApp (NOWEB), `noweb-arm-2026.10.1` (2026-11-12, from 2026.9.2: the audit listed 28 critical CVEs in the old image) | 3000 LAN (PC only) |
| `kaya-cloudflared` | The public tunnel for `sigmakayachat.pt`, its only connector since 2026-09-25 | none |

WAHA and cloudflared sit behind compose profiles (`COMPOSE_PROFILES` in `.env`).
They are switched on during the cutover and never run in two places at once.

## What happens to a message

1. WAHA posts the event to `gateway:8088/waha/webhook`.
   - **Ideas skip everything below.** The owner's DMs starting with `/idea`, and
     replies quoting a `💡[idea-N]` message, go to idea-pipeline's inbox
     (`src/gateway/idea_tap.py`, on when `IDEA_INBOX_URL` is set). They are not
     journaled, not sent to the PC and never become Kaya's memory. The pipeline's
     own `💡[idea-` messages are dropped when WAHA echoes them back.
   - **The owner's other DMs go to the lab assistant** (after the idea and homelab
     taps): voice notes, and plain text that is no `/command` and no reply to one of
     Kaya's messages, go to lab-assistant's inbox with the audio already downloaded
     (`src/gateway/assistant_tap.py`, on when `ASSISTANT_INBOX_URL` is set). What the
     assistant decides is Kaya's comes back as text through `POST /assistant/handback`
     (relay token) and is journaled from here on like any owner DM. Its own `🤖`
     echoes are dropped. If its inbox is down, the message stays Kaya's.
   - **`/help` is answered on the Pi, not handed anywhere** (`src/gateway/help_tap.py`,
     on when `HELP_OWNER_NUMBERS` or `IDEA_OWNER_NUMBERS` is set): the owner's DM
     gets a fixed command list, so it works while the PC is off. The Help-page link
     is appended when `GATEWAY_HELP_URL` is set in the Pi's `.env`. The reply's own
     echo is dropped like the other taps'. It is never journaled or sent to the PC.
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

## Birthday messages

`src/gateway/birthdays.py` posts a fixed line at 00:00 to
`GATEWAY_BIRTHDAY_CHAT` (the general group), tagging the member, and a nudge at
12:00 only if nobody has said parabéns since midnight. The lines and windows are
`chat.birthdays` in `config.yaml`. The midnight line also carries a fixed ask
(`chat.birthdays.missing`) naming the members whose birthday the bot still needs,
plain names and nobody tagged; the roster carries them with `date: ""`.
The dates come from the PC
(`GET /whatsapp/relay/birthdays`, pulled every 30 min while the PC answers) and
the last copy is kept in `data/gateway/birthdays.json`, so it works with the PC
off. What was sent is in `data/gateway/birthday_state.json`; `/status` on the
internal port shows the roster size and the last send.

## The "Get in touch" form (`POST /contact`)

The showcase page ends with a "Get in touch" section (PT: "Fala comigo"): a plain
HTML `<form method="post" action="/contact">` for the visitor's name, email and what
they want help with. No JS on the page posts it — the CSP's `form-action` is `'self'`
(`static_assets.page_csp`), so the submit lands on this gateway.

`src/gateway/contact.py` relays one note to the owner over two independent channels:

- **WhatsApp** — the same WAHA sender the taps use, into the owner's DM. Nothing new has to be
  set: the DM is derived from `IDEA_OWNER_NUMBERS` (the same convention as `homelab.default_chat`
  and `/help`). `GATEWAY_CONTACT_CHAT` (a JID) or `CONTACT_OWNER_NUMBERS` only override that
  target. Without an owner number at all, the channel is off.
- **Email** — the optional extra: SMTP with STARTTLS (`CONTACT_SMTP_HOST`/`PORT`/`USER`/`PASSWORD`).
  `CONTACT_EMAIL_TO` is the owner's own address and stays a server-side secret: it is
  never on the page, and the email in the form is always the *visitor's*. Empty host
  turns the channel off.

A submit that fills the hidden honeypot field (`name="website"`) or that is missing a
required field never sends. The form is flood-proofed in three layers, none of which needs
a widget on the page (the CSP and the page's own test guards reject one):

- **A proof token minted when the page is served.** `landing()` stamps a fresh
  `<issued-at>.<hmac>` into a hidden field (`name="proof"`), signed with `KAYA_RELAY_TOKEN` —
  no new key. `/contact` delivers only notes that carry one: valid, at least three seconds old
  (nobody types a name, an email and a message faster), and no older than 72 h, so a tab left
  open overnight still sends. A curl loop or a saved page has nothing to echo back.
- **The honeypot** (`name="website"`, hidden by CSS): a bot that fills every field it finds
  gets the thank-you and nothing is sent.
- **Per-IP limits**, in-memory (the gateway is one worker): a two-minute cooldown, five an hour,
  twenty a day, with the table bounded so a distributed flood cannot grow its memory.

Anything dropped answers the same thank-you page, so a bot cannot tell it was dropped. `/status`
on the LAN port carries the counters (`sent`, `dropped_proof`, `dropped_honeypot`,
`dropped_throttle`, `rejected`); the public listener never serves them. In front of the tunnel,
add the edge rule once in the Cloudflare dashboard — Security → WAF → Rate limiting rules:
*Method is POST* and *URI Path equals `/contact`*, block after **20 requests per IP in 60 s**
for 1 h; plus Bot Fight Mode. That is what stops a multi-IP swarm, which an in-process table
cannot see.

The route lives on the public listener only (8080, which only the tunnel reaches), so the way to
watch it is from anywhere at all:

```bash
curl -s -X POST https://sigmakayachat.pt/contact \
  -d 'name=Test&email=test@example.com&message=hello'      # thank-you, but nothing sends: no token
docker logs -f kaya-gateway                                # "contact:" lines only for a real submit
```

To see one end to end, open the page and use the form — fetching it is what hands a browser the
token in the first place.

## Deploying

```bash
scripts/deploy_pi.sh --init-env   # first time: writes the Pi's .env from ~/kaya-prod/.env
scripts/deploy_pi.sh              # afterwards: sync code, rebuild, restart
```

`/homelaboff` and `/homelabon` are on when `HOMELAB_OWNER_NUMBERS` is set in the
Pi's `.env` (the owner's number(s), comma-separated, digits only). `--init-env`
does not copy it, so on an existing Pi add it by hand and restart the gateway.

Three Pi units around the gateway are the lab's since 2026-10-22 (lab-infra `pi/install.sh`):

- `pc-wake.timer`: Wake-on-LAN at power.yaml `wake_time` minus `wol_lead_minutes`.
- `pc-wake-request.path`/`.service`: runs the same `pc-wake.sh` when the gateway
  drops `data/gateway/pc-wake.request` — that is what `/homelabon` does.
- `lab-firewall.service`: ports 3000 and 8088 accept the PC only (`DOCKER-USER`,
  since Docker-published ports skip `INPUT`).

The gateway mounts the Pi's `~/lab-infra/shared` at `/lab-infra` for the power schedule.

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
