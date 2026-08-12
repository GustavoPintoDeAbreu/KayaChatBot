"""Drive the real WhatsApp path with invented people, and assert on what returns.

The unit suite proves the wiring and `preflight_e2e.py` proves each capability in
isolation. Neither reproduces an evening in the group: several people talking
over each other, a photo arriving mid-argument, someone asking for an edit while
the last one is still rendering, and a conversation long enough to push past the
14000-token retrieval budget. That is what this does.

Two ideas carry it:

**Drive the real webhook.** Every message is a synthetic WAHA event POSTed to the
`kaya-sim` instance, which runs the production `whatsapp_server` in mock mode.
Parsing, routing, the GPU lock, scoping, media handling and the async image path
are all the real ones; only the outbound WhatsApp client is a mock. In mock mode
the webhook *awaits* generation and returns the full result dict, so a beat can
assert on the routing decision rather than infer it from the reply text.

**Deterministic spine, improvised filler.** Free-form LLM chatter cannot be
asserted on. A scenario is a list of beats: `say` beats carry exact text, media
and expectations; `improv` beats ask the personas for a few turns of natural
conversation so the context the bot sees is real rather than a list of test
probes. Assertions live on the scripted beats; realism comes from the improv.

Cost is small — personas run on the cheap non-reasoning Grok, and a standard run
is a few hundred thousand tokens (~$0.30). Image edits, not tokens, dominate the
wall clock at ~90s each.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.testing.sim_world import BOT_JID, SIM_GROUP, Persona, personas

BASE_DIR = Path(__file__).parent.parent.parent


# ── talking to the sim instance ──────────────────────────────────────────────

class SimClient:
    """HTTP client for the kaya-sim webhook and its mock outbox."""

    def __init__(self, base_url: str = "http://127.0.0.1:7862", timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._seq = 0
        self._lock = threading.Lock()
        # Unique per run. A plain counter restarts at 1 every run while the sim
        # container outlives runs, so the adapter's replay guard — which is
        # correct, real WhatsApp ids are globally unique — saw the second run's
        # messages as the first run's backlog and ignored every one of them.
        self._run = uuid.uuid4().hex[:8]

    def _next_id(self, prefix: str = "sim") -> str:
        with self._lock:
            self._seq += 1
            return f"{prefix}_{self._run}_{self._seq:05d}"

    def health(self) -> Dict[str, Any]:
        import requests

        return requests.get(f"{self.base_url}/whatsapp/health", timeout=15).json()

    def outbox(self) -> List[Dict[str, Any]]:
        import requests

        return requests.get(f"{self.base_url}/whatsapp/outbox", timeout=30).json().get("sent", [])

    def build_event(self, persona: Persona, text: str, *, chat: str,
                    mention: bool = False, reply_to_bot: bool = False,
                    media: Optional[Dict[str, str]] = None,
                    message_id: Optional[str] = None,
                    timestamp: Optional[int] = None) -> Dict[str, Any]:
        """A WAHA `message` event in the NOWEB shape `parse_waha_message` expects.

        `participantAlt` carries the real phone behind the @lid — that is what
        `resolve_speaker` matches on, and without it every sender falls back to a
        push name and the whitelist gate drops DMs.
        """
        is_group = chat.endswith("@g.us")
        payload: Dict[str, Any] = {
            "id": message_id or self._next_id(),
            "from": chat,
            "body": text,
            "notifyName": persona.name,
            "fromMe": False,
            "timestamp": timestamp or int(time.time()),
            "mentionedIds": [BOT_JID] if mention else [],
            "_data": {"key": {}},
        }
        if is_group:
            payload["participant"] = persona.jid
            payload["_data"]["key"]["participantAlt"] = f"{persona.phone}@s.whatsapp.net"
        else:
            payload["_data"]["key"]["remoteJidAlt"] = f"{persona.phone}@s.whatsapp.net"
        if reply_to_bot:
            payload["replyTo"] = {"participant": BOT_JID}
        if media:
            payload["media"] = media
        return {"event": "message", "me": {"id": BOT_JID}, "payload": payload}

    def send(self, event: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """POST one event. Returns (result, seconds). Never raises."""
        import requests

        started = time.time()
        try:
            response = requests.post(f"{self.base_url}/whatsapp/webhook",
                                     json=event, timeout=self.timeout)
            response.raise_for_status()
            return response.json(), time.time() - started
        except Exception as exc:  # noqa: BLE001 — a dropped turn is data, not a crash
            return {"handled": False, "error": f"{type(exc).__name__}: {exc}"}, \
                   time.time() - started


class MediaServer:
    """Serve a directory over HTTP on an address the sim container can reach.

    The webhook carries media as a URL that the app fetches, so a simulated photo
    or voice note has to be downloadable from inside the container. Binding to the
    docker bridge gateway is the least intrusive way to do that — no extra
    service, no volume, and the files never leave the box.
    """

    def __init__(self, directory: Path, host: str = "172.18.0.1", port: int = 8899):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.host, self.port = host, port
        self._server = None
        self._thread = None

    def start(self) -> None:
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        directory = str(self.directory)

        class QuietHandler(SimpleHTTPRequestHandler):
            """Subclass, not functools.partial: log_message has to be overridden on
            the class, and setting it on a partial silently does nothing — one
            access-log line per media fetch then drowns the run output."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, *args, **kwargs):
                pass

        handler = QuietHandler
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def publish(self, source: Path, name: Optional[str] = None) -> str:
        """Copy a file in and return the URL the container should fetch."""
        import shutil

        name = name or source.name
        shutil.copy2(source, self.directory / name)
        return f"http://{self.host}:{self.port}/{name}"


# ── the invented people ──────────────────────────────────────────────────────

_PERSONA_SYSTEM = """You are {name}, in a Portuguese friend group's WhatsApp chat.

How you write: {trait}. You often bring up {interest}.

Rules:
- European Portuguese, NEVER Brazilian. Laugh as "ahah"/"ahahah", never "kkk". No "você", "a gente", "legal", "cara".
- The way friends actually type: lower case, short, abbreviations, the odd typo.
- ONE message, at most 15 words. No quotation marks, no narration, no emoji spam.
- You are a PERSON, never the bot. Never answer as an assistant.
- Do not mention being an AI or that this is a simulation.

Reply with the message text only."""

_PERSONA_USER = """Recent messages:
{history}

Write {name}'s next message. {nudge}"""


class PersonaDriver:
    """Generates what an invented person says next, via the cloud model.

    Personas are synthetic and their prompts contain no group data — see
    `src/testing/sim_world.py`. Only the bot's *replies* reach the judge, which is
    the documented eval-time exception in CLAUDE.md.
    """

    def __init__(self, provider: Any, model: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0

    def next_message(self, persona: Persona, history: List[str],
                     nudge: str = "") -> str:
        system = _PERSONA_SYSTEM.format(name=persona.name, trait=persona.trait,
                                        interest=persona.interest)
        user = _PERSONA_USER.format(history="\n".join(history[-14:]) or "(a conversa está a começar)",
                                    name=persona.name, nudge=nudge)
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        try:
            text = self.provider.chat_completion([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:  # noqa: BLE001 — a persona outage must not end the run
            return f"(persona error: {type(exc).__name__})"
        text = (text or "").strip().strip('"').split("\n")[0]
        self.completion_chars += len(text)
        return text[:300]


# ── results ──────────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    index: int
    beat: str
    speaker: str
    chat: str
    text: str
    reply: str = ""
    command: Optional[str] = None
    image: Optional[str] = None
    handled: bool = False
    seconds: float = 0.0
    media: str = ""
    failures: List[str] = field(default_factory=list)
    note: str = ""
    scores: Optional[Dict[str, Any]] = None


@dataclass
class SimResult:
    scenario: str
    started: str
    turns: List[TurnRecord] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def failures(self) -> List[TurnRecord]:
        return [t for t in self.turns if t.failures]


# ── assertions ───────────────────────────────────────────────────────────────

def _words(text: str) -> int:
    return len((text or "").split())


def check_expectations(result: Dict[str, Any], expect: Dict[str, Any],
                       outbox_delta: List[Dict[str, Any]]) -> List[str]:
    """Compare one webhook result against a beat's expectations.

    Returns a list of human-readable failures — empty means the beat passed.
    """
    failures: List[str] = []
    reply = result.get("reply") or ""

    if "handled" in expect and bool(result.get("handled")) != bool(expect["handled"]):
        failures.append(f"handled={result.get('handled')}, expected {expect['handled']}")

    if "command" in expect and result.get("command") != expect["command"]:
        failures.append(f"command={result.get('command')!r}, expected {expect['command']!r}")

    if "image" in expect and result.get("image") != expect["image"]:
        failures.append(f"image={result.get('image')!r}, expected {expect['image']!r}")

    if "max_words" in expect and _words(reply) > expect["max_words"]:
        failures.append(f"reply is {_words(reply)} words, expected at most {expect['max_words']}")

    if "min_words" in expect and _words(reply) < expect["min_words"]:
        failures.append(f"reply is {_words(reply)} words, expected at least {expect['min_words']}")

    lowered = reply.lower()
    for needle in expect.get("contains_all", []):
        if needle.lower() not in lowered:
            failures.append(f"reply is missing {needle!r}")

    any_of = expect.get("contains_any", [])
    if any_of and not any(n.lower() in lowered for n in any_of):
        failures.append(f"reply contains none of {any_of}")

    for needle in expect.get("not_contains", []):
        if needle.lower() in lowered:
            failures.append(f"reply should not mention {needle!r}")

    if expect.get("image_sent"):
        if not any("image_bytes" in item for item in outbox_delta):
            failures.append("no image reached the outbox")

    if expect.get("voice_sent"):
        if not any("voice_bytes" in item for item in outbox_delta):
            failures.append("no voice note reached the outbox")

    # What was actually SPOKEN, which is not the same string as the written reply:
    # a web-grounded answer used to end with Piper reading "🌐 Fontes: x.com,
    # play.google.com" out loud. Only assertable because the mock outbox records
    # the spoken text and no longer just its byte count.
    spoken = " ".join(
        item.get("spoken_text", "") for item in outbox_delta if "voice_bytes" in item
    ).lower()
    for needle in expect.get("spoken_not_contains", []):
        if needle.lower() in spoken:
            failures.append(f"the voice note said {needle!r}")
    spoken_any = expect.get("spoken_contains_any", [])
    if spoken_any and not any(n.lower() in spoken for n in spoken_any):
        failures.append(f"the voice note said none of {spoken_any}")

    if expect.get("text_only"):
        if any("voice_bytes" in item or "image_bytes" in item for item in outbox_delta):
            failures.append("expected a text reply, got media")

    return failures


# ── media pool ───────────────────────────────────────────────────────────────

class MediaPool:
    """Real photos and voice notes, addressed by logical name.

    Scenarios say `photo_a` rather than naming a file, so the same scenario keeps
    working when the export changes. Photos come from the curated bake-off set
    (faces already verified as clear); voice notes come straight from the export,
    because a real Portuguese voice note is the only honest test of transcription.
    """

    def __init__(self, server: MediaServer,
                 photo_dir: Path = BASE_DIR / "data" / "bench_photos",
                 audio_dir: Optional[Path] = None):
        self.server = server
        self.photos = sorted(Path(photo_dir).glob("*.jpg"))
        audio_dir = audio_dir or Path.home() / "Downloads" / "WhatsApp kaya media"
        self.voices = sorted(Path(audio_dir).glob("PTT-*.opus"))[:40] if audio_dir.exists() else []
        self._published: Dict[str, str] = {}

    def url(self, name: str) -> Optional[Dict[str, str]]:
        """Resolve `photo_a` / `voice_b` to the media dict a webhook carries."""
        if name in self._published:
            kind = "image/jpeg" if name.startswith("photo") else "audio/ogg; codecs=opus"
            return {"url": self._published[name], "mimetype": kind}

        kind, _, suffix = name.partition("_")
        index = (ord(suffix[0]) - ord("a")) if suffix else 0
        if kind == "photo":
            if not self.photos:
                return None
            source = self.photos[index % len(self.photos)]
            mimetype = "image/jpeg"
        elif kind == "voice":
            if not self.voices:
                return None
            source = self.voices[index % len(self.voices)]
            mimetype = "audio/ogg; codecs=opus"
        else:
            return None

        url = self.server.publish(source, f"{name}{source.suffix}")
        self._published[name] = url
        return {"url": url, "mimetype": mimetype}

    def local_path(self, name: str) -> Optional[Path]:
        kind, _, suffix = name.partition("_")
        index = (ord(suffix[0]) - ord("a")) if suffix else 0
        pool = self.photos if kind == "photo" else self.voices
        return pool[index % len(pool)] if pool else None


# ── the runner ───────────────────────────────────────────────────────────────

class SimRunner:
    """Executes a scenario's beats against the sim instance."""

    def __init__(self, client: SimClient, driver: PersonaDriver, pool: MediaPool,
                 people: List[Persona], log: Callable[[str], None] = print):
        self.client = client
        self.driver = driver
        self.pool = pool
        self.people = people
        self.log = log
        self.history: List[str] = []
        self._by_name = {p.name: p for p in people}
        self._aliases: Dict[str, Persona] = {}
        # Outbox length at the last request that kicks off async work, so a
        # following `wait` beat only counts deliveries caused by THAT request.
        self._pending_mark = 0

    # -- helpers ----------------------------------------------------------
    def _persona(self, who: Optional[str]) -> Persona:
        """Resolve a beat's speaker, deterministically.

        A scenario may name someone outside the current cast — `scope` names Nuno
        but the smoke preset runs three people. Falling back to `random.choice`
        was silently wrong: the secret was planted in one person's DM and the
        follow-up asked in another's, so the test "passed" while proving nothing.
        Unknown names are aliased onto the cast by a stable hash and remembered,
        so one name is always the same person for the whole run.
        """
        if not who:
            return random.choice(self.people)
        if who in self._by_name:
            return self._by_name[who]
        if who not in self._aliases:
            chosen = self.people[sum(map(ord, who)) % len(self.people)]
            self._aliases[who] = chosen
            self.log(f"  · {who!r} is not in the cast — playing them as {chosen.name}")
        return self._aliases[who]

    def _chat_id(self, chat: str) -> str:
        if chat == "group":
            return SIM_GROUP
        if chat.startswith("dm:"):
            return self._persona(chat[3:]).jid
        return chat

    def _outbox_len(self) -> int:
        try:
            return len(self.client.outbox())
        except Exception:  # noqa: BLE001
            return 0

    def _outbox_since(self, mark: int) -> List[Dict[str, Any]]:
        try:
            return self.client.outbox()[mark:]
        except Exception:  # noqa: BLE001
            return []

    # -- beats ------------------------------------------------------------
    def run(self, beats: List[Dict[str, Any]], result: SimResult) -> SimResult:
        for beat in beats:
            kind = beat.get("kind", "say")
            handler = getattr(self, f"_beat_{kind}", None)
            if handler is None:
                self.log(f"  ! unknown beat kind {kind!r}, skipping")
                continue
            handler(beat, result)
        return result

    def _record(self, result: SimResult, beat_kind: str, persona: Persona, chat: str,
                text: str, payload: Dict[str, Any], seconds: float,
                expect: Dict[str, Any], outbox_delta: List[Dict[str, Any]],
                note: str = "", media: str = "") -> TurnRecord:
        record = TurnRecord(
            index=len(result.turns) + 1, beat=beat_kind, speaker=persona.name,
            chat=chat, text=text, reply=payload.get("reply") or "",
            command=payload.get("command"), image=payload.get("image"),
            handled=bool(payload.get("handled")), seconds=round(seconds, 2),
            media=media, note=note,
        )
        if expect:
            record.failures = check_expectations(payload, expect, outbox_delta)
        result.turns.append(record)

        flag = "" if not record.failures else "  ✗ " + "; ".join(record.failures)
        self.log(f"  [{record.index:>3}] {persona.name}: {text[:60]!r} "
                 f"-> {record.reply[:60]!r} ({record.seconds}s){flag}")

        self.history.append(f"{persona.name}: {text}")
        if record.reply:
            self.history.append(f"Kaya Bot: {record.reply}")
        return record

    def _beat_say(self, beat: Dict[str, Any], result: SimResult) -> None:
        persona = self._persona(beat.get("who"))
        chat = beat.get("chat", "group")
        chat_id = self._chat_id(chat)
        media_name = beat.get("media")
        media = self.pool.url(media_name) if media_name else None
        if media_name and media is None:
            self.log(f"  ! media {media_name!r} unavailable, beat sent without it")

        event = self.client.build_event(
            persona, beat.get("text", ""), chat=chat_id,
            mention=beat.get("mention", chat == "group"),
            reply_to_bot=beat.get("reply_to_bot", False), media=media,
            message_id=beat.get("message_id"),
        )
        mark = self._outbox_len()
        self._pending_mark = mark
        payload, seconds = self.client.send(event)
        self._record(result, "say", persona, chat, beat.get("text", ""), payload,
                     seconds, beat.get("expect", {}), self._outbox_since(mark),
                     note=beat.get("note", ""), media=media_name or "")

    def _beat_raw(self, beat: Dict[str, Any], result: SimResult) -> None:
        """A hand-built payload — malformed events, replays, anything odd."""
        persona = self._persona(beat.get("who"))
        mark = self._outbox_len()
        payload, seconds = self.client.send(beat["event"])
        self._record(result, "raw", persona, beat.get("chat", "group"),
                     beat.get("label", "(raw event)"), payload, seconds,
                     beat.get("expect", {}), self._outbox_since(mark),
                     note=beat.get("note", ""))

    def _beat_improv(self, beat: Dict[str, Any], result: SimResult) -> None:
        """Persona chatter, to build context the bot actually has to carry."""
        turns = int(beat.get("turns", 4))
        address_every = int(beat.get("address_bot_every", 0))
        topic = beat.get("topic", "")
        self.log(f"  ~ improv: {turns} turns" + (f" about {topic}" if topic else ""))

        for step in range(turns):
            persona = self._persona(beat.get("who"))
            addressed = bool(address_every) and (step + 1) % address_every == 0
            nudge = beat.get("nudge", "")
            if topic and step == 0:
                nudge = f"Start a conversation about {topic}. {nudge}"
            if addressed:
                nudge += " Ask the group's bot a question about the group."
            text = self.driver.next_message(persona, self.history, nudge)

            chat_id = self._chat_id(beat.get("chat", "group"))
            event = self.client.build_event(persona, text, chat=chat_id,
                                            mention=addressed)
            mark = self._outbox_len()
            payload, seconds = self.client.send(event)
            self._record(result, "improv", persona, beat.get("chat", "group"),
                         text, payload, seconds, {}, self._outbox_since(mark))

    def _beat_burst(self, beat: Dict[str, Any], result: SimResult) -> None:
        """Several people at once — the GPU lock allows ONE job.

        `chat.concurrency.max_concurrent` is 1 and the server drops on
        GpuBusyError rather than queueing, so this measures the real drop rate
        instead of assuming it.
        """
        senders = self.people[: int(beat.get("senders", 4))]
        text = beat.get("text", "quem é o mais alto do grupo?")
        self.log(f"  ~ burst: {len(senders)} senders at once")

        def fire(persona: Persona) -> Tuple[Persona, Dict[str, Any], float]:
            event = self.client.build_event(persona, text, chat=SIM_GROUP, mention=True)
            payload, seconds = self.client.send(event)
            return persona, payload, seconds

        mark = self._outbox_len()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(senders)) as pool:
            outcomes = list(pool.map(fire, senders))

        delta = self._outbox_since(mark)
        answered = sum(1 for _, payload, _ in outcomes if (payload.get("reply") or "").strip())
        for persona, payload, seconds in outcomes:
            self._record(result, "burst", persona, "group", text, payload, seconds,
                         {}, [], note=beat.get("note", ""))
        result.metrics.setdefault("bursts", []).append({
            "senders": len(senders), "answered": answered,
            "dropped": len(senders) - answered,
            "outbox_messages": len(delta),
        })
        self.log(f"    -> {answered}/{len(senders)} answered, "
                 f"{len(senders) - answered} dropped")

    def _beat_wait(self, beat: Dict[str, Any], result: SimResult) -> None:
        """Wait for an async delivery (an image takes ~90s) or just pause.

        The mark is the outbox length recorded when the *request* was made, not 0:
        looking at the whole outbox finds an image from an earlier beat and
        returns immediately, so the wait silently passes and the next beat runs
        while the render is still going. That is what made the "implicit subject"
        beat come back busy instead of editing.
        """
        seconds = float(beat.get("seconds", 5))
        want = beat.get("for")
        deadline = time.time() + seconds
        mark = self._pending_mark if want else 0
        started = time.time()

        while time.time() < deadline:
            delta = self.client.outbox()
            if want == "image" and any("image_bytes" in item for item in delta[mark:]):
                break
            if want == "voice" and any("voice_bytes" in item for item in delta[mark:]):
                break
            if want is None:
                time.sleep(min(1.0, deadline - time.time()))
                continue
            time.sleep(2)

        if want:
            delta = [i for i in self.client.outbox()[mark:] if f"{want}_bytes" in i]
            elapsed = time.time() - started
            ok = bool(delta)
            self.log(f"  ~ waited {elapsed:.0f}s for {want}: "
                     f"{'arrived' if ok else 'NEVER ARRIVED'}")
            if want == "image":
                result.images.append({"seconds": round(elapsed, 1), "arrived": ok,
                                      "bytes": delta[0].get("image_bytes") if delta else 0})
            if not ok and beat.get("required", True):
                result.turns.append(TurnRecord(
                    index=len(result.turns) + 1, beat="wait", speaker="-",
                    chat="-", text=f"waiting for {want}", handled=False,
                    seconds=round(elapsed, 1),
                    failures=[f"{want} never arrived within {seconds:.0f}s"],
                    note=beat.get("note", "")))

    def _beat_ingest(self, beat: Dict[str, Any], result: SimResult) -> None:
        """Fold what has been said into the sim's vector store.

        Mock mode skips the ingest scheduler, so recall of something said earlier
        in this very run only works if it is ingested on purpose.

        Through the app's own endpoint, NOT a separate process: ChromaDB clients
        keep their own view, so ingesting elsewhere leaves the running app still
        answering "não tenho essa informação" about a fact that is now in the
        store. That cost a whole debugging session — the needle was ingested,
        retrievable at rank 1, and the bot still denied it.
        """
        import requests

        self.log("  ~ ingesting the conversation so far …")
        try:
            response = requests.post(f"{self.client.base_url}/whatsapp/ingest", timeout=600)
            response.raise_for_status()
            payload = response.json()
            self.log(f"    {payload.get('messages')} message(s) -> "
                     f"{payload.get('chunks')} chunk(s) across {payload.get('scopes')} scope(s)")
            result.metrics.setdefault("ingests", []).append({"ok": True, **payload})
        except Exception as exc:  # noqa: BLE001
            self.log(f"    ! ingest failed: {exc}")
            result.metrics.setdefault("ingests", []).append({"ok": False, "error": str(exc)})
