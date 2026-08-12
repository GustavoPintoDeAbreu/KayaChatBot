#!/usr/bin/env python3
"""Generate or edit one image, in a process of its own, then exit.

A subprocess rather than a module the bot imports, for three reasons that all
come from the same fact — these models are 20GB and the bot is long-lived:

* **Memory actually comes back.** Dropping a diffusion pipeline in-process left
  20.7GB allocated with no `nn.Module` still alive (measured during the
  bake-off). Process exit is the only deallocation that is guaranteed.
* **The card stays free between requests.** Image edits are occasional; holding
  20GB permanently would cost the box a whole GPU for a feature used a few times
  a day.
* **A crash is contained.** An OOM here kills a worker, not the bot.

Editing runs the bake-off winner. Generation from scratch is a different job and
runs Z-Image Turbo: it lost the *editing* bake-off badly (it returns a stranger)
but it is a competent 6B text-to-image model, which is exactly what "faz uma
imagem de um gato astronauta" needs.

    kaya_chatbot_env/bin/python scripts/imagegen_worker.py --mode edit \
        --image in.jpg --prompt "põe-me como rei medieval" --out out.png

    kaya_chatbot_env/bin/python scripts/imagegen_worker.py --mode generate \
        --prompt "um gato astronauta" --out out.png
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = Path(__file__).parent.parent

NEGATIVE = ("deformed, distorted face, disfigured, extra fingers, blurry, low quality, "
            "watermark, text, cartoon, doll, plastic skin")

# Kontext has to be told what to KEEP, not only what to change: "make him a
# medieval king" is read as licence to return a different medieval king. Every
# instruction in the bake-off ended with a clause like this — the 0.409 likeness
# was measured with it, and production was shipping without it.
DEFAULT_IDENTITY_CLAUSE = (
    "Keep the person's face, facial features, hairline, skin tone and identity "
    "exactly the same as in the original photo. Do not change who they are."
)

KEEP_BF16 = {
    "qwen": ["img_in", "txt_in", "proj_out", "norm_out", "time_text_embed"],
    "flux": ["x_embedder", "context_embedder", "proj_out", "norm_out", "time_text_embed"],
}


def open_photo(path: str):
    """The photo at full resolution, upright. No scaling: that comes later, once."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_image(path: str, longest: int = 1024):
    """Legacy loader, kept for the Qwen path which does its own bucketing."""
    from PIL import Image

    image = open_photo(path)
    scale = longest / max(image.size)
    if scale < 1.0:
        image = image.resize((max(int(image.width * scale), 16),
                              max(int(image.height * scale), 16)), Image.LANCZOS)
    width = max(image.width // 16 * 16, 16)
    height = max(image.height // 16 * 16, 16)
    return image.crop((0, 0, width, height))


def bnb_8bit(keep):
    from diffusers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=keep)


def edit_with_flux(image_path: str, prompt: str, out_path: str,
                   steps: int, seed: int, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """FLUX.1 Kontext — single pass, 8-bit, ~15GB resident, ~90s an image."""
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TfBnb

    from src.chat import face_utils

    options = options or {}
    candidates_wanted = max(1, int(options.get("candidates", 1)))
    source, reference = face_utils.prepare_source(
        image_path,
        face_crop=bool(options.get("face_crop", True)),
        min_ratio=float(options.get("face_min_ratio", 0.22)),
        target_ratio=float(options.get("face_target_ratio", 0.32)),
    )

    quant = PipelineQuantizationConfig(quant_mapping={
        "transformer": bnb_8bit([]),
        "text_encoder_2": TfBnb(load_in_8bit=True),
    })
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16,
        quantization_config=quant)
    # Fully resident, NOT enable_model_cpu_offload(): the offload hooks duplicate
    # bitsandbytes weights instead of moving them, which pushed a 14.5GB pipeline
    # to 22.4GB and OOMed — identically at 704px, which is what identified it as
    # weights rather than activations.
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    # One model load, N takes. Loading the 15GB pipeline is most of the cost, so a
    # second candidate is far cheaper than a second request would be.
    candidates = []
    for index in range(candidates_wanted):
        take_seed = seed + index
        image = pipe(
            image=source, prompt=prompt,
            height=source.height, width=source.width, _auto_resize=False,
            guidance_scale=float(options.get("guidance_scale", 2.5)),
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(take_seed),
        ).images[0]
        candidates.append((image, take_seed))

    best, best_seed, score = face_utils.pick_best(reference, candidates)
    best.save(out_path)
    return {
        "seed": best_seed,
        "likeness": None if score is None else round(score, 4),
        "candidates": len(candidates),
        "source_size": [source.width, source.height],
    }


def _qwen_embed(repo: str, image_path: str, prompt: str, out_path: str) -> None:
    """Pass A, run as its own process — see edit_with_qwen."""
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE, calculate_dimensions)

    encoder = QwenImageEditPlusPipeline.from_pretrained(
        repo, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("cuda")
    source = load_image(image_path)
    width, height = calculate_dimensions(CONDITION_IMAGE_SIZE, source.width / source.height)
    condition = [encoder.image_processor.resize(source, height, width)]

    payload = {}
    for key, text in (("positive", prompt), ("negative", NEGATIVE)):
        # no_grad is load-bearing: encode_prompt leaves autograd on, so the
        # returned embedding keeps the whole VL forward graph alive behind it.
        with torch.no_grad():
            embeds, mask = encoder.encode_prompt(
                image=condition, prompt=text, device=torch.device("cuda"))
        if mask is None:
            # An all-ones mask is returned as None, and __call__ reads a None
            # negative mask as "no negative prompt", silently disabling true CFG.
            mask = torch.ones(embeds.shape[:2], dtype=torch.long)
        payload[key] = (embeds.detach().cpu(), mask.detach().cpu())
    torch.save(payload, out_path)


def edit_with_qwen(image_path: str, prompt: str, out_path: str,
                   steps: int, seed: int, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Qwen-Image-Edit-2509 in two passes.

    The 20B transformer needs 8-bit to render cleanly (NF4 turns it into a
    mosaic) and 8-bit occupies 19.4GB, which leaves no room for the 7B text
    encoder beside it. So the encoder runs first in a process of its own, and the
    transformer then loads into the whole card and works from what it wrote.
    """
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    repo = "Qwen/Qwen-Image-Edit-2509"
    embeds_path = str(Path(out_path).with_suffix(".embeds.pt"))
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--stage", "embed", "--image", image_path, "--prompt", prompt,
                    "--out", embeds_path], check=True, cwd=str(BASE_DIR))
    cached = torch.load(embeds_path, weights_only=False)
    Path(embeds_path).unlink(missing_ok=True)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        repo, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=bnb_8bit(KEEP_BF16["qwen"]))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        repo, transformer=transformer, text_encoder=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    positive, positive_mask = cached["positive"]
    negative, negative_mask = cached["negative"]
    result = pipe(
        image=[load_image(image_path)],
        prompt_embeds=positive.to("cuda"), prompt_embeds_mask=positive_mask.to("cuda"),
        negative_prompt_embeds=negative.to("cuda"),
        negative_prompt_embeds_mask=negative_mask.to("cuda"),
        true_cfg_scale=4.0, num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    result.save(out_path)
    return {"seed": seed, "likeness": None, "candidates": 1}


def generate_from_text(prompt: str, out_path: str, steps: int, seed: int) -> None:
    """Z-Image Turbo — 6B text-to-image, a couple of seconds per image."""
    import torch
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo",
                                          torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    result = pipe(prompt=prompt + ", photorealistic, sharp focus",
                  negative_prompt=NEGATIVE,
                  width=1024, height=1024,
                  guidance_scale=1.0, num_inference_steps=steps,
                  generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    result.save(out_path)


EDITORS = {"flux-kontext": edit_with_flux, "qwen-image-edit": edit_with_qwen}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or edit one image.")
    parser.add_argument("--mode", default="edit", choices=["edit", "generate"])
    parser.add_argument("--stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--editor", default=None)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=0)
    parser.add_argument("--no-face-crop", action="store_true")
    parser.add_argument("--no-identity-clause", action="store_true")
    args = parser.parse_args()

    if args.stage == "embed":
        _qwen_embed("Qwen/Qwen-Image-Edit-2509", args.image, args.prompt, args.out)
        return

    from src.config_loader import load_config

    config: Dict[str, Any] = load_config(str(BASE_DIR / "config.yaml"))
    icfg = (config.get("chat", {}) or {}).get("imagegen", {}) or {}
    # A fixed seed makes a bad face reproducible: asking again returns the same
    # stranger. Random unless one is pinned, for the bench or to reproduce a bug.
    seed = args.seed or int(icfg.get("seed") or 0) or random.randint(1, 2**31 - 1)

    if args.mode == "generate":
        generate_from_text(args.prompt, args.out,
                           args.steps or int(icfg.get("generate_steps", 12)), seed)
        print(json.dumps({"ok": True, "out": args.out, "seed": seed}))
        return

    if not args.image:
        raise SystemExit("--image is required for --mode edit")
    editor = args.editor or icfg.get("editor", "flux-kontext")
    if editor not in EDITORS:
        raise SystemExit(f"unknown editor {editor!r}; pick one of {sorted(EDITORS)}")

    prompt = args.prompt
    clause = str(icfg.get("identity_clause", DEFAULT_IDENTITY_CLAUSE) or "")
    if clause and not args.no_identity_clause:
        prompt = f"{prompt.rstrip().rstrip('.')}. {clause}"

    options = {
        "candidates": args.candidates or int(icfg.get("candidates", 1)),
        "face_crop": not args.no_face_crop and bool(icfg.get("face_crop", True)),
        "face_min_ratio": float(icfg.get("face_min_ratio", 0.22)),
        "face_target_ratio": float(icfg.get("face_target_ratio", 0.32)),
        "guidance_scale": float(icfg.get("guidance_scale", 2.5)),
    }
    info = EDITORS[editor](args.image, prompt, args.out,
                           args.steps or int(icfg.get("edit_steps", 28)),
                           seed, options) or {}

    print(json.dumps({"ok": True, "out": args.out, "prompt": prompt, **info}))


if __name__ == "__main__":
    main()
