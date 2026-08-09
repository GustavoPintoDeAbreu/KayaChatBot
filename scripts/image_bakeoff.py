#!/usr/bin/env python3
"""Bake off image-editing models on real group photos.

Same method that overturned the LLM assumptions: build every candidate, run them
on the group's own photos with the prompts they would actually get, and decide on
measurements rather than on model cards.

Two numbers decide it, in this order:

* **likeness** — ArcFace cosine between the face in the source photo and the face
  in the edit. An editor that produces a beautiful picture of a stranger has
  failed the only job that matters here: these are edits *of specific friends*.
  A missing face scores 0, which is the correct verdict, not an error.
* **adherence** — did it do what it was told, judged 1-5 by the local vision
  model. Cloud judges are not an option: the images are group members' faces and
  the privacy invariant is absolute. The judge is the same gemma-4-12b already
  serving prod, run with --mmproj on the bench port.

VRAM and wall time are recorded but rank last — the plan explicitly drops latency
as a constraint for generation, since replies are sent asynchronously.

Instruction-following editors get the instruction verbatim. SDXL and Z-Image are
img2img models that cannot parse one, so each edit also carries a descriptive
`scene`; handing them an instruction they structurally cannot follow would rig
the comparison rather than measure it.

Stages run separately — generation holds a GPU for hours, judging needs a llama
server that would otherwise be competing for the same card:

    # 1. generate (GPU0; prod keeps GPU1)
    CUDA_VISIBLE_DEVICES=0 kaya_chatbot_env/bin/python scripts/image_bakeoff.py \
        --stage generate --arms sdxl-ipadapter,z-image-turbo,qwen-image-edit

    # 2. score — needs the vision server up:
    #    KAYA_BENCH_GGUF=gemma-4-12b-it-Q6_K.gguf KAYA_BENCH_SM=none \
    #    KAYA_BENCH_EXTRA="-ngl 999 --mmproj /models/mmproj-F16.gguf" \
    #    docker compose --profile bench up -d llama-bench
    kaya_chatbot_env/bin/python scripts/image_bakeoff.py --stage score \
        --run reports/image_bakeoff/<stamp>
    #    tear it down with `docker rm -f kaya-llama-bench`, NOT with
    #    `docker compose --profile bench down`, which also tries to take the
    #    compose network out from under the live prod containers.

    # 3. contact sheet + ranking
    kaya_chatbot_env/bin/python scripts/image_bakeoff.py --stage report \
        --run reports/image_bakeoff/<stamp>

Everything it writes stays on the box: reports/image_bakeoff/ is gitignored
because it is full of edited photographs of real people.
"""

import argparse
import base64
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# The 8-bit Qwen transformer leaves under 2GB of headroom on a 24GB card, and a
# few hundred MB of that was going to allocator fragmentation. Set before torch
# is imported anywhere (every import in this file is deferred into a function).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = Path(__file__).parent.parent

# The edits the group would actually ask for: strong, funny transformations that
# put real pressure on identity preservation. A weak edit ("make it brighter")
# would let every candidate score full marks and rank nothing.
EDITS: List[Dict[str, str]] = [
    {
        "id": "mugshot",
        "instruction": "Turn this into a police mugshot: the person is holding a "
                       "booking board with numbers, standing against a height chart "
                       "wall under harsh light. Keep the face exactly the same.",
        "scene": "a police mugshot photo, person holding a booking board with numbers, "
                 "height chart on the wall behind, harsh overhead lighting, police station",
    },
    {
        "id": "medieval-king",
        "instruction": "Dress the person as a medieval king on a stone throne, with a "
                       "golden crown and a fur cloak. Keep the face exactly the same.",
        "scene": "a medieval king seated on a stone throne, golden crown, fur cloak, "
                 "castle hall, torchlight, oil painting realism",
    },
    {
        "id": "astronaut",
        "instruction": "Put the person in an astronaut suit floating inside a space "
                       "station, helmet visor up. Keep the face exactly the same.",
        "scene": "an astronaut in a white spacesuit floating inside a space station, "
                 "helmet visor up, earth visible through the window",
    },
    {
        "id": "gym-bodybuilder",
        "instruction": "Make the person an enormous oiled bodybuilder posing on a "
                       "competition stage. Keep the face exactly the same.",
        "scene": "an enormous oiled bodybuilder flexing on a competition stage, "
                 "spotlights, cheering crowd behind",
    },
    {
        "id": "beach-cocktail",
        "instruction": "Put the person on a tropical beach at sunset holding a giant "
                       "cocktail, wearing a loud floral shirt. Keep the face exactly "
                       "the same.",
        "scene": "a person on a tropical beach at sunset holding a giant cocktail, "
                 "loud floral shirt, palm trees, warm golden light",
    },
]

NEGATIVE = ("deformed, distorted face, disfigured, extra fingers, blurry, low quality, "
            "watermark, text, cartoon, doll, plastic skin")

JUDGE_PROMPT = """You are scoring an AI photo edit.

The instruction given to the editor was:
"{instruction}"

Look at the image and answer with STRICT JSON only, no other text:
{{"adherence": <1-5>, "realism": <1-5>, "note": "<max 12 words>"}}

adherence: 5 = every part of the instruction is clearly visible; 1 = the
instruction was ignored.
realism: 5 = looks like a real photograph; 1 = mangled anatomy or obvious AI
artefacts."""


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_photos(photo_dir: Path) -> List[Dict[str, Any]]:
    manifest = photo_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} — run scripts/pick_bench_photos.py first")
    return json.loads(manifest.read_text(encoding="utf-8"))


def fit_source(path: str, longest: int = 1024):
    """Load a photo scaled so the long side is `longest`, on a multiple of 16.

    Every candidate here is happier on a bounded, aligned canvas, and the phone
    photos in the export run to 4000px — generating at native resolution would
    trade hours of GPU time for detail the judge never sees.
    """
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = longest / max(image.size)
    if scale < 1.0:
        image = image.resize((max(int(image.width * scale), 16),
                              max(int(image.height * scale), 16)), Image.LANCZOS)
    width = max(image.width // 16 * 16, 16)
    height = max(image.height // 16 * 16, 16)
    return image.crop((0, 0, width, height))


# Quantizing a diffusion transformer's input/output projections is what turns a
# 4-bit edit into a crystalline mosaic: they carry the latent in and out of the
# residual stream, so their error lands directly in pixel space rather than being
# averaged over many layers. Keeping them in bf16 costs a few hundred MB.
_KEEP_BF16 = ["img_in", "txt_in", "proj_out", "norm_out", "time_text_embed"]


def _bnb_diffusers_config(bits: int):
    """bitsandbytes config for a diffusers component (transformer, VAE)."""
    import torch
    from diffusers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True,
                                  llm_int8_skip_modules=_KEEP_BF16)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=_KEEP_BF16)


def _bnb_transformers_config(bits: int):
    """The same, for a transformers component — the two libraries each ship
    their own BitsAndBytesConfig and a pipeline mixes components from both."""
    import torch
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)


# ── candidate arms ───────────────────────────────────────────────────────────
# Each loader returns (generate_fn, label). generate_fn(source_image, edit, seed)
# returns a PIL image. Loading is deferred into the function so an arm that is
# not being run costs nothing and a missing download fails only that arm.

def build_sdxl_ipadapter(photos, edits, seed):
    """SDXL img2img with the plus-face IP-Adapter.

    The mature baseline. It cannot follow an instruction, so it gets the
    descriptive `scene`; the IP-Adapter is what carries the face through, and
    without it SDXL would simply paint a different person in the same pose.
    """
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
    from transformers import CLIPVisionModelWithProjection

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16)
    # The *_vit-h adapters need the ViT-H encoder (1280-dim) that lives in
    # models/image_encoder, not the ViT-bigG one (1664-dim) sitting next to the
    # weights in sdxl_models/, which is what load_ip_adapter picks by default. The
    # mismatch surfaces as a matmul shape error at the first generate, not at
    # load, so it is loaded explicitly here and handed to the pipeline.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    pipe.set_ip_adapter_scale(0.7)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def generate(source, edit, seed, photo_id=None):
        generator = torch.Generator("cuda").manual_seed(seed)
        return pipe(
            prompt=edit["scene"] + ", photorealistic, sharp focus",
            negative_prompt=NEGATIVE,
            image=source, ip_adapter_image=source,
            strength=0.65, guidance_scale=6.0, num_inference_steps=40,
            generator=generator,
        ).images[0]

    return generate, pipe


def build_z_image_turbo(photos, edits, seed):
    """Z-Image Turbo as img2img.

    Turbo is a text-to-image model — the instruction-editing sibling (Z-Image
    Omni / Edit) is not publicly downloadable, so this arm measures what is
    actually available: a 6B model re-noising the photo towards a description. It
    is in the bench as the fast end of the range, and is expected to trade
    likeness for speed.
    """
    import torch
    from diffusers import ZImageImg2ImgPipeline

    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def generate(source, edit, seed, photo_id=None):
        generator = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            prompt=edit["scene"] + ", photorealistic, keep the same face",
            negative_prompt=NEGATIVE,
            image=source, strength=0.6,
            guidance_scale=1.0, num_inference_steps=12,
            generator=generator,
        ).images[0]

    return generate, pipe


def _qwen_embed_all(repo: str, photos, edits, out_path: Path) -> None:
    """Pass A: embed every (photo, instruction) pair and write them to disk.

    Runs as its own process (see `_qwen_embeddings`). Dropping the pipeline in
    process was not enough — 20.7GB stayed allocated with no nn.Module left
    alive, so the parameter tensors are held by something the collector cannot
    see. Process exit is the one deallocation that is guaranteed.
    """
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE, calculate_dimensions)

    encoder = QwenImageEditPlusPipeline.from_pretrained(
        repo, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    encoder.to("cuda")

    cache: Dict[Any, Any] = {}
    for photo in photos:
        source = fit_source(photo["path"])
        width, height = calculate_dimensions(CONDITION_IMAGE_SIZE,
                                             source.width / source.height)
        condition = [encoder.image_processor.resize(source, height, width)]
        for edit in list(edits) + [{"id": "__negative__", "instruction": NEGATIVE}]:
            # no_grad is load-bearing: encode_prompt does not disable autograd, so
            # a cached embedding keeps the whole VL forward graph alive behind it.
            # One photo fits; eight fill the card.
            with torch.no_grad():
                embeds, mask = encoder.encode_prompt(
                    image=condition, prompt=edit["instruction"],
                    device=torch.device("cuda"))
            if mask is None:
                # encode_prompt drops an all-ones mask, but __call__ reads a None
                # negative mask as "no negative prompt" and silently turns true
                # CFG off — so it is rebuilt rather than passed through.
                mask = torch.ones(embeds.shape[:2], dtype=torch.long)
            cache[(photo["id"], edit["id"])] = (embeds.detach().cpu(),
                                                mask.detach().cpu())
        print(f"  embedded {photo['id']}")

    torch.save({"|".join(key): value for key, value in cache.items()}, out_path)
    print(f"  wrote {len(cache)} embeddings to {out_path}")


def _qwen_embeddings(repo: str, photos, edits) -> Dict[str, Any]:
    """Run pass A in a subprocess and load what it wrote."""
    import subprocess

    import torch

    out_path = BASE_DIR / "reports" / "image_bakeoff" / ".qwen_embeds.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"repo": repo, "photos": photos, "edits": list(edits),
                          "out": str(out_path)})
    print("  embedding prompts in a separate process …")
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--stage", "embed", "--payload", payload],
                   check=True, cwd=str(BASE_DIR))
    return torch.load(out_path, weights_only=False)


def build_qwen_image_edit(photos, edits, seed):
    """Qwen-Image-Edit-2509 — a true instruction editor, 20B, in two passes.

    bf16 is ~57GB of weights against 24GB of card and ~39GB of free RAM, so it has
    to be quantized to run here at all. At NF4 it composes the edit correctly and
    then renders it as a crystalline mosaic that destroys the face — measured, not
    assumed: the VAE round-trips a photo cleanly in bf16, and the artefact
    survives both a bf16 text encoder and no CPU offload, so it is the 4-bit
    transformer. 8-bit does not corrupt, but at 19.4GB it leaves no room for the
    7B text encoder beside it, and this card is Ampere — there is no FP8.

    Hence two passes. The encoder runs alone first and every prompt is embedded up
    front; then it is freed and the transformer loads 8-bit into the whole card,
    generating from the cached embeddings. Nothing but latents and text embeddings
    ever needs both halves resident at once.
    """
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE, calculate_dimensions)

    repo = "Qwen/Qwen-Image-Edit-2509"
    cache = _qwen_embeddings(repo, photos, edits)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        repo, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=_bnb_diffusers_config(8))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        repo, transformer=transformer, text_encoder=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # Tiling the VAE and slicing attention buy back the ~1GB the 8-bit
    # transformer does not leave for activations. Both are exact, not lossy.
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    def generate(source, edit, seed, photo_id=None):
        embeds, mask = cache[f"{photo_id}|{edit['id']}"]
        negative, negative_mask = cache[f"{photo_id}|__negative__"]
        generator = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            image=[source],
            prompt_embeds=embeds.to("cuda"), prompt_embeds_mask=mask.to("cuda"),
            negative_prompt_embeds=negative.to("cuda"),
            negative_prompt_embeds_mask=negative_mask.to("cuda"),
            true_cfg_scale=4.0, num_inference_steps=40,
            generator=generator,
        ).images[0]

    return generate, pipe


def build_flux_kontext(photos, edits, seed):
    """FLUX.1 Kontext dev — built for instruction editing, gated on HF.

    4-bit for the same reason as Qwen: 12B bf16 is 24GB of transformer before the
    T5 encoder is loaded at all.
    """
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.quantizers import PipelineQuantizationConfig

    quant = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True,
                      "bnb_4bit_quant_type": "nf4",
                      "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16,
        quantization_config=quant)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def generate(source, edit, seed, photo_id=None):
        generator = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            image=source, prompt=edit["instruction"],
            guidance_scale=2.5, num_inference_steps=28,
            generator=generator,
        ).images[0]

    return generate, pipe


ARMS: Dict[str, Callable] = {
    "sdxl-ipadapter": build_sdxl_ipadapter,
    "z-image-turbo": build_z_image_turbo,
    "qwen-image-edit": build_qwen_image_edit,
    "flux-kontext": build_flux_kontext,
}


# ── stages ───────────────────────────────────────────────────────────────────

def stage_generate(run_dir: Path, arms: List[str], photos: List[Dict[str, Any]],
                   edits: List[Dict[str, str]], seed: int) -> None:
    import torch

    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {"cells": []}
    done = {(cell["arm"], cell["photo"], cell["edit"]) for cell in results["cells"]}

    for arm in arms:
        if arm not in ARMS:
            print(f"!! unknown arm {arm}, skipping")
            continue
        print(f"\n=== {arm}")
        free_gpu()
        try:
            generate, pipe = ARMS[arm](photos, edits, seed)
        except Exception as exc:  # noqa: BLE001 — one broken arm must not end the run
            print(f"!! {arm} failed to load: {type(exc).__name__}: {exc}")
            results["cells"].append({"arm": arm, "photo": None, "edit": None,
                                     "error": f"load: {exc}"})
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        out_dir = run_dir / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()

        for photo in photos:
            source = fit_source(photo["path"])
            for edit in edits:
                if (arm, photo["id"], edit["id"]) in done:
                    continue
                started = time.time()
                try:
                    image = generate(source, edit, seed, photo_id=photo['id'])
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {photo['id']}/{edit['id']}: {type(exc).__name__}: {exc}")
                    results["cells"].append({"arm": arm, "photo": photo["id"],
                                             "edit": edit["id"], "error": str(exc)[:300]})
                    continue
                elapsed = time.time() - started
                name = f"{photo['id']}__{edit['id']}.png"
                image.save(out_dir / name)
                results["cells"].append({
                    "arm": arm, "photo": photo["id"], "edit": edit["id"],
                    "file": f"{arm}/{name}", "seconds": round(elapsed, 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                })
                print(f"  {photo['id']}/{edit['id']}  {elapsed:.0f}s")
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        del generate, pipe
        free_gpu()

    print(f"\n✓ generation done → {results_path}")


def face_embedding(app, path: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)


def judge_image(url: str, path: Path, instruction: str,
                timeout: float = 240.0) -> Optional[Dict[str, Any]]:
    """Score one edit with the local vision model. None on failure."""
    import requests

    try:
        encoded = base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None
    try:
        response = requests.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": JUDGE_PROMPT.format(instruction=instruction)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ]}],
                "max_tokens": 120, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except (requests.RequestException, KeyError, ValueError):
        return None

    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
    except ValueError:
        return None
    return {
        "adherence": float(parsed.get("adherence", 0)),
        "realism": float(parsed.get("realism", 0)),
        "note": str(parsed.get("note", ""))[:80],
    }


def stage_score(run_dir: Path, photo_dir: Path, judge_url: str) -> None:
    import numpy as np

    from scripts.pick_bench_photos import load_analyser

    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    edits = {edit["id"]: edit for edit in EDITS}

    references = {}
    for npy in photo_dir.glob("*.npy"):
        references[npy.stem] = np.load(npy)

    app = load_analyser()
    judged = 0
    for cell in results["cells"]:
        if cell.get("error") or not cell.get("file"):
            continue
        path = run_dir / cell["file"]
        if not path.exists():
            continue

        if "likeness" not in cell:
            reference = references.get(cell["photo"])
            embedding = face_embedding(app, path)
            if reference is None or embedding is None:
                # No face in the output is not a missing measurement — it is the
                # worst possible result for an edit meant to keep someone's face.
                cell["likeness"] = 0.0
                cell["face_found"] = embedding is not None
            else:
                cell["likeness"] = round(float(np.dot(reference, embedding)), 4)
                cell["face_found"] = True

        if "adherence" not in cell:
            verdict = judge_image(judge_url, path, edits[cell["edit"]]["instruction"])
            if verdict:
                cell.update(verdict)
                judged += 1
            else:
                cell["adherence"] = None
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  {cell['arm']}/{cell['photo']}/{cell['edit']}  "
              f"likeness={cell['likeness']}  adherence={cell.get('adherence')}")

    print(f"\n✓ scored ({judged} judged) → {results_path}")


def stage_report(run_dir: Path) -> None:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cells = [c for c in results["cells"] if not c.get("error")]

    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], []).append(cell)

    def mean(values: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in values if isinstance(v, (int, float))]
        return round(sum(clean) / len(clean), 3) if clean else None

    summary = []
    for arm, arm_cells in by_arm.items():
        summary.append({
            "arm": arm,
            "n": len(arm_cells),
            "likeness": mean([c.get("likeness") for c in arm_cells]),
            "adherence": mean([c.get("adherence") for c in arm_cells]),
            "realism": mean([c.get("realism") for c in arm_cells]),
            "face_kept_pct": round(100 * sum(bool(c.get("face_found")) for c in arm_cells)
                                   / max(len(arm_cells), 1)),
            "median_seconds": mean([c.get("seconds") for c in arm_cells]),
            "peak_vram_gb": max([c.get("peak_vram_gb") or 0 for c in arm_cells], default=0),
        })
    summary.sort(key=lambda row: (row["likeness"] or 0, row["adherence"] or 0), reverse=True)
    results["summary"] = summary
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'arm':<20}{'likeness':>10}{'adherence':>11}{'realism':>9}"
          f"{'face%':>7}{'secs':>8}{'VRAM':>7}")
    for row in summary:
        print(f"{row['arm']:<20}{str(row['likeness']):>10}{str(row['adherence']):>11}"
              f"{str(row['realism']):>9}{row['face_kept_pct']:>6}%"
              f"{str(row['median_seconds']):>8}{row['peak_vram_gb']:>6}G")

    write_contact_sheet(run_dir, results, summary)


def write_contact_sheet(run_dir: Path, results: Dict[str, Any],
                        summary: List[Dict[str, Any]]) -> None:
    """One HTML page: every edit, every arm, side by side with its scores.

    Numbers rank the candidates; the eye decides whether the winner is actually
    usable, and that needs the images next to each other.
    """
    rows = []
    arms = [row["arm"] for row in summary]
    cells = {(c["arm"], c["photo"], c["edit"]): c
             for c in results["cells"] if c.get("file")}
    photos = sorted({c["photo"] for c in results["cells"] if c.get("photo")})

    for photo in photos:
        for edit in EDITS:
            tiles = []
            for arm in arms:
                cell = cells.get((arm, photo, edit["id"]))
                if not cell:
                    tiles.append('<td class="miss">—</td>')
                    continue
                tiles.append(
                    f'<td><img src="{cell["file"]}" loading="lazy">'
                    f'<div class="s">likeness {cell.get("likeness")} · '
                    f'adherence {cell.get("adherence")} · {cell.get("seconds")}s</div></td>')
            rows.append(f'<tr><th>{photo}<br><span>{edit["id"]}</span></th>'
                        + "".join(tiles) + "</tr>")

    head = "".join(f"<th>{arm}</th>" for arm in arms)
    table = "".join(
        f"<tr><td>{row['arm']}</td><td>{row['likeness']}</td><td>{row['adherence']}</td>"
        f"<td>{row['realism']}</td><td>{row['face_kept_pct']}%</td>"
        f"<td>{row['median_seconds']}s</td><td>{row['peak_vram_gb']}G</td></tr>"
        for row in summary)

    html = f"""<!doctype html><meta charset="utf-8">
<title>Image bake-off {run_dir.name}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;background:#111;color:#eee}}
 h1{{font-size:1.3rem}} img{{width:260px;border-radius:6px;display:block}}
 table{{border-collapse:collapse}} td,th{{padding:.5rem;vertical-align:top;text-align:left}}
 th span{{color:#8ab;font-weight:400}} .s{{font-size:11px;color:#9aa;margin-top:.25rem}}
 .miss{{color:#666}} .sum td,.sum th{{border-bottom:1px solid #333}}
</style>
<h1>Image bake-off — {run_dir.name}</h1>
<table class="sum"><tr><th>arm</th><th>likeness</th><th>adherence</th><th>realism</th>
<th>face kept</th><th>time</th><th>VRAM</th></tr>{table}</table>
<p>Likeness is ArcFace cosine against the source face (higher is better; ~0.4+ is
the usual "same person" threshold). Adherence and realism are 1-5, judged locally.</p>
<table><tr><th></th>{head}</tr>{"".join(rows)}</table>
"""
    path = run_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"\ncontact sheet: file://{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-editing model bake-off.")
    parser.add_argument("--stage", default="generate",
                        choices=["generate", "score", "report", "embed"])
    parser.add_argument("--payload", default=None, help="internal: embed-stage arguments")
    parser.add_argument("--arms", default="sdxl-ipadapter,z-image-turbo,qwen-image-edit")
    parser.add_argument("--photos", default="data/bench_photos")
    parser.add_argument("--run", default=None, help="existing run dir (resume / score / report)")
    parser.add_argument("--limit-photos", type=int, default=0)
    parser.add_argument("--limit-edits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--judge-url",
                        default=os.environ.get("KAYA_LLAMA_URL", "http://127.0.0.1:8081"))
    args = parser.parse_args()

    photo_dir = BASE_DIR / args.photos if not Path(args.photos).is_absolute() else Path(args.photos)
    run_dir = Path(args.run) if args.run else BASE_DIR / "reports" / "image_bakeoff" / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "embed":
        spec = json.loads(args.payload)
        _qwen_embed_all(spec["repo"], spec["photos"], spec["edits"], Path(spec["out"]))
        return

    if args.stage == "generate":
        photos = load_photos(photo_dir)
        edits = EDITS
        if args.limit_photos:
            photos = photos[: args.limit_photos]
        if args.limit_edits:
            edits = edits[: args.limit_edits]
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        print(f"run: {run_dir}\narms: {arms}\n{len(photos)} photos x {len(edits)} edits "
              f"= {len(photos) * len(edits)} images per arm")
        stage_generate(run_dir, arms, photos, edits, args.seed)
    elif args.stage == "score":
        stage_score(run_dir, photo_dir, args.judge_url)
    else:
        stage_report(run_dir)


if __name__ == "__main__":
    main()
