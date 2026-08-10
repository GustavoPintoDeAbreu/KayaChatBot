# Phase 5 — image generation, image understanding, and the bake-off

Written 2026-08-10. Companion to `reports/image_bakeoff/<stamp>/index.html`,
which holds the measurements and the pictures themselves.

## What the bot can do that it could not before

| capability | how it works | where |
|---|---|---|
| **read a photo** | the serving model is multimodal; `--mmproj` was simply never loaded | `src/chat/vision.py` |
| **remember a photo** | the description becomes ordinary message text, so memory/ingestion/RAG need no changes at all | `whatsapp_adapter.py` |
| **edit a photo** | out-of-process worker on the free card, async reply | `src/chat/imagegen.py` |
| **invent a picture** | Z-Image Turbo, 6B text-to-image | `scripts/imagegen_worker.py` |
| **speak two languages** | one Piper voice per language, chosen per sentence | `src/chat/tts.py` |

## The bake-off

Four candidates on 8 real group photos × 5 strong edits. Photos chosen by face
geometry (`scripts/pick_bench_photos.py`) — 221 of 658 export images passed a
bar of face size, detector confidence and frontality; 8 were kept, half solo and
half group shots, each with its ArcFace reference embedding.

Scored on **likeness** (ArcFace cosine against the source face) first,
**adherence** and **realism** (1–5, judged by the local gemma-4-12b with
`--mmproj`) second, speed last. No cloud judge: the images are members' faces
and the privacy invariant is absolute.

Method note: instruction-following editors get the instruction verbatim; SDXL
and Z-Image are img2img models that cannot parse one, so each edit also carries a
descriptive `scene`. Handing them an instruction they structurally cannot follow
would have rigged the comparison rather than measured it.

## What the hardware forced

Everything below was measured, not assumed.

**4-bit destroys a 20B diffusion transformer.** Qwen-Image-Edit at NF4 composed
every edit correctly and rendered it as a crystalline mosaic. Isolated by
elimination: the VAE round-trips a photo cleanly in bf16, and the artefact
survived both a bf16 text encoder and no CPU offload. 8-bit is clean.

**8-bit does not fit beside its own text encoder.** 19.4GB transformer + 7B
encoder > 24GB, and Ampere has no FP8. Hence two passes: the encoder runs alone
and embeds every prompt, then it is freed and the transformer loads into the
whole card. Pass A must be a *separate process* — dropping the pipeline in
Python left 20.7GB allocated with no `nn.Module` alive.

**Two cards do not make one image faster.** bf16 needs 40.9GB against 47.1GB of
VRAM, and reserving activation headroom spills to disk. With 8-bit and
`device_map="balanced"`, accelerate put 23.5GB on GPU0 and left GPU1 idle at 0%
— the layers run in sequence and, with no P2P on this box, every boundary
crossing stages through host RAM. What two cards *do* buy is parallelism across
photos: one process per card halved the bake-off's wall time.

**Small traps that cost real time**
- `hf download --exclude "a" "b"` reads the second pattern as a *filename* and
  fails the whole repo. One pattern per flag.
- `encode_prompt` leaves autograd on, so each cached embedding kept a whole
  VL forward graph alive. One photo fit; eight filled the card.
- `encode_prompt` returns a `None` mask when it is all ones, and `__call__`
  reads a `None` negative mask as "no negative prompt" — silently disabling CFG.
- `hf auth whoami` exits 0 while printing "Not logged in".
- Python block-buffers a redirected log, which briefly looked like a stall.

## Design decisions worth keeping

**The worker is a subprocess, permanently.** Not tidiness: VRAM that does not
come back, a card held hostage between requests, and an OOM that would otherwise
take the bot down with it.

**A photo is treated exactly like a voice note.** Both arrive as media with no
text; both are turned into text at the edge. Everything downstream — the message
log, the ingester, the router, retrieval — needed no changes, which is why
"aquela foto do barco" is findable a week later.

**Editing is gated to the shared group** (`chat.imagegen.allowed_scopes`). It
puts a real member's face into an invented scene; the group chose to have a bot
in the room, an arbitrary DM did not.

**Every failure path speaks.** Busy, not allowed, generation failed — each has
its own reply. Silence after a promise is worse than a refusal, which is the
same principle as the audio capability gate.

## Deployment shape

- **GPU1** — llama.cpp serving gemma-4-12b Q6_K + `--mmproj`, plus Whisper and
  the embedder in the app container
- **GPU0** — the image worker, only while a job runs
- prod sees both cards but stays pinned to GPU1 by `CUDA_VISIBLE_DEVICES`
- the ~120GB of diffusion weights are mounted read-only from the host HF cache
  rather than copied into the image

`scripts/preflight_e2e.py` checks all of it against the running stack and exits
non-zero, so it can gate the deploy.

## Known limits

- An edit takes minutes. The reply is async by design; the acknowledgement sets
  the expectation.
- One image at a time. A second request is told to wait rather than queued.
- The implicit subject is the *last* photo in the chat. Quoting a specific older
  photo is not yet wired.
- Video is still not ingested (535 files in the export).
