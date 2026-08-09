#!/usr/bin/env python3
"""Turn a WhatsApp export's media into searchable group memory.

Photos are described by the vision model; voice notes are transcribed by Whisper.
Either way the raw file becomes text the retriever can search.

Media alone is weak memory: the model can say "um homem num barco com uma cerveja"
but not who, when, or why. The export line carries exactly that —
`31/01/26, 14:02 - Rafa: IMG-20260131-WA0020.jpg (file attached)` — so every
description or transcript is stored with its sender, its date and the messages
around it. That is what makes "aquela foto do barco" findable later.

Stickers are skipped by default. The export references them 2,166 times but they
are only ~640 unique files, reused constantly as reactions; describing them adds
noise, not memory. Video is not handled yet.

    --kinds IMG        photos      (needs a llama.cpp server with --mmproj)
    --kinds PTT,AUD    voice notes (needs faster-whisper; runs on the GPU)

Idempotent and resumable: descriptions are cached by filename, and chunk ids are
derived from the filename, so re-running skips work already done and upserts
rather than duplicating. Safe to interrupt.

    # dry run — parse and report, describe nothing
    kaya_chatbot_env/bin/python scripts/ingest_media.py --export "$HOME/Downloads/chat.txt" \
        --media "$HOME/Downloads/media" --dry-run

    # describe + ingest (needs a llama.cpp server with --mmproj)
    KAYA_LLAMA_URL=http://127.0.0.1:8081 kaya_chatbot_env/bin/python scripts/ingest_media.py \
        --export "$HOME/Downloads/chat.txt" --media "$HOME/Downloads/media" --limit 50
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.scope import SHARED

# "3/26/20, 15:29 - Gil João: IMG-20200604-WA0004.jpg (file attached)"
LINE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})\s*-\s*"
    r"(?P<sender>[^:]{1,60}):\s*(?P<text>.*)$"
)
ATTACHMENT = re.compile(r"(?P<file>(?P<kind>IMG|STK|VID|PTT|AUD)-\d{8}-WA\d+\.\w+)\s*\(.*?\)")

AUDIO_KINDS = {"PTT", "AUD"}
IMAGE_KINDS = {"IMG", "STK"}

DESC_PROMPT = (
    "Descreve esta imagem para servir de memória de um grupo de amigos. "
    "1-2 frases, português europeu. Diz o que se vê (pessoas, local, objetos, texto visível) "
    "e, se for um meme, screenshot ou print de conversa, diz isso explicitamente. "
    "Não inventes nomes de pessoas."
)


def parse_export(path: Path) -> List[Dict[str, Any]]:
    """Flatten the export into messages, tagging any attachment reference."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                # continuation of the previous multi-line message
                if out:
                    out[-1]["text"] += " " + raw.strip()
                continue
            text = m.group("text").strip()
            att = ATTACHMENT.search(text)
            out.append({
                "date": m.group("date"), "time": m.group("time"),
                "sender": m.group("sender").strip(), "text": text,
                "file": att.group("file") if att else None,
                "kind": att.group("kind") if att else None,
            })
    return out


def _iso(date_s: str, time_s: str) -> str:
    for fmt in ("%m/%d/%y %H:%M", "%d/%m/%y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt).isoformat()
        except ValueError:
            continue
    return ""


def context_around(messages: List[Dict], idx: int, window: int = 3) -> str:
    """Nearby non-attachment chatter, which is what gives a photo its meaning."""
    lines = []
    for j in range(max(0, idx - window), min(len(messages), idx + window + 1)):
        if j == idx:
            continue
        m = messages[j]
        if m.get("file") or not m["text"].strip():
            continue
        if "<Media omitted>" in m["text"]:
            continue
        lines.append(f"{m['sender']}: {m['text'][:160]}")
    return "\n".join(lines[-6:])


def describe(url: str, image_path: Path, timeout: float = 180.0) -> Optional[str]:
    """One vision call. Returns None on failure rather than raising."""
    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
    except OSError:
        return None
    try:
        r = requests.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": DESC_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "max_tokens": 110,
                "temperature": 0.2,
                # Gemma-4 otherwise emits its thinking channel and the answer
                # lands in reasoning_content with an empty content field.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content") or ""
        return content.strip() or None
    except (requests.RequestException, KeyError, ValueError):
        return None


_WHISPER = None


def get_whisper(model_name: str, device: str, compute_type: str):
    """Load Whisper once, lazily — only when audio is actually being ingested."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        print(f"loading Whisper {model_name} ({device}/{compute_type}) …")
        _WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER


def transcribe(path: Path, model_name: str, device: str, compute_type: str,
               language: str = "pt") -> Optional[str]:
    """Transcribe one voice note. Returns None on failure rather than raising."""
    try:
        model = get_whisper(model_name, device, compute_type)
        # vad_filter drops silence, which is most of a WhatsApp voice note's tail.
        segments, _info = model.transcribe(str(path), language=language,
                                           vad_filter=True, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text or None
    except Exception:  # noqa: BLE001 — one bad file must not stop the run
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Describe export images into the vector store.")
    ap.add_argument("--export", required=True, help="WhatsApp chat .txt export")
    ap.add_argument("--media", required=True, help="Folder with the exported media files")
    ap.add_argument("--kinds", default="IMG", help="Comma-separated: IMG,STK,VID,PTT,AUD")
    ap.add_argument("--limit", type=int, default=0, help="Max images this run (0 = all)")
    ap.add_argument("--scope", default=SHARED, help="Memory scope to write under")
    ap.add_argument("--cache", default="data/media_descriptions.json")
    ap.add_argument("--dry-run", action="store_true", help="Parse and report only")
    ap.add_argument("--url", default=os.environ.get("KAYA_LLAMA_URL", "http://127.0.0.1:8081"))
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--whisper-device", default="cuda")
    ap.add_argument("--whisper-compute", default="int8_float16")
    ap.add_argument("--language", default="pt")
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent
    config = load_config(str(base_dir / "config.yaml"))
    kinds = {k.strip().upper() for k in args.kinds.split(",") if k.strip()}
    media_dir = Path(args.media).expanduser()

    messages = parse_export(Path(args.export).expanduser())
    attachments = [(i, m) for i, m in enumerate(messages) if m.get("kind")]
    by_kind: Dict[str, int] = {}
    for _, m in attachments:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1

    print(f"parsed {len(messages)} messages, {len(attachments)} attachment references")
    for k, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        mark = "->" if k in kinds else "  (skipped)"
        print(f"   {k:<4} {n:>5} {mark}")

    # Only the kinds asked for, only files that actually exist, deduped by filename
    # (the same sticker/photo is referenced many times).
    wanted, seen = [], set()
    for idx, m in attachments:
        if m["kind"] not in kinds or m["file"] in seen:
            continue
        path = media_dir / m["file"]
        if not path.exists():
            continue
        seen.add(m["file"])
        wanted.append((idx, m, path))

    print(f"\n{len(wanted)} unique files present on disk for kinds={sorted(kinds)}")

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = base_dir / args.cache
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    todo = [w for w in wanted if w[1]["file"] not in cache]
    print(f"{len(cache)} already described, {len(todo)} to do")

    if args.dry_run:
        print("\n--- dry run: sample of what would be ingested ---")
        for idx, m, path in wanted[:3]:
            print(f"  {m['file']}  from {m['sender']} on {m['date']}")
            ctx = context_around(messages, idx)
            if ctx:
                print(f"    context: {ctx.splitlines()[0][:90]}")
        return

    if args.limit:
        todo = todo[: args.limit]

    described, failed, t0 = 0, 0, time.time()
    for n, (idx, m, path) in enumerate(todo, 1):
        if m["kind"] in AUDIO_KINDS:
            desc = transcribe(path, args.whisper_model, args.whisper_device,
                              args.whisper_compute, args.language)
        else:
            desc = describe(args.url, path)
        if desc:
            cache[m["file"]] = desc
            described += 1
        else:
            failed += 1
        if n % 10 == 0 or n == len(todo):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} described ({failed} failed) "
                  f"~{rate:.1f}s each, ETA {rate*(len(todo)-n)/60:.0f} min", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ndescribed {described}, failed {failed}, cache now {len(cache)} entries")

    # ── build + upsert chunks ────────────────────────────────────────────────
    from src.data.ingest import Ingester

    ing = Ingester(config)
    ids, docs, metas = [], [], []
    for idx, m, path in wanted:
        desc = cache.get(m["file"])
        if not desc:
            continue
        ts = _iso(m["date"], m["time"])
        ctx = context_around(messages, idx)
        kind_label = "Áudio" if m["kind"] in AUDIO_KINDS else "Imagem"
        doc = f"[{kind_label} enviado por {m['sender']}" + (f" em {ts[:10]}" if ts else "") + "]\n"
        doc += desc
        if ctx:
            doc += f"\n\nConversa à volta:\n{ctx}"
        ids.append(("aud_" if m["kind"] in AUDIO_KINDS else "img_") + m["file"].replace(".", "_"))
        docs.append(doc)
        metas.append({
            "participants": m["sender"], "mentioned": "",
            "message_count": 1, "token_count": len(doc) // 4,
            "timestamp_start": ts, "timestamp_end": ts,
            "scope": args.scope,
            "source": "audio" if m["kind"] in AUDIO_KINDS else "image",
            "media_file": m["file"],
        })

    if not ids:
        print("nothing to ingest")
        return

    print(f"embedding + upserting {len(ids)} media chunks …")
    embeddings = ing.encoder.encode(docs, show_progress_bar=False,
                                    normalize_embeddings=True).tolist()
    for i in range(0, len(ids), 200):
        sl = slice(i, i + 200)
        ing.collection.upsert(ids=ids[sl], documents=docs[sl],
                              metadatas=metas[sl], embeddings=embeddings[sl])
    print(f"✓ collection now holds {ing.collection.count()} chunks")


if __name__ == "__main__":
    main()
