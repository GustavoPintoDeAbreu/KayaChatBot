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
from typing import Any, Dict, Optional

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

# The same clause for a photo with more than one person in it. The singular one
# was being sent for group shots too — "the person's face" in a picture of three
# people names nobody, and both failed edits from the live group ("make these
# guys kissing", "put a gun in the left guy's hand") were two-subject photos.
DEFAULT_IDENTITY_CLAUSE_PLURAL = (
    "Keep every person's face, facial features, hairline, skin tone and identity "
    "exactly the same as in the original photo. Do not change who any of them are."
)

# The face-restore pass. A LoRA trained for FLUX.2 Klein 9B that takes a
# body/scene image plus a face image and transfers the face — which is exactly
# the second half of a two-stage edit. rank-64 (316MB) by default; the repo also
# ships a rank-128 build if the quality is not there.
DEFAULT_RESTORE_LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"
DEFAULT_RESTORE_LORA = "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"

# Modules whose 8-bit error lands directly in pixel space, so they stay bf16.
# Only the Qwen path uses this; FLUX quantizes its transformer whole, having been
# measured to survive it. A "flux" key lived here for a while looking load-bearing
# and was never read by anything.
KEEP_BF16 = {
    "qwen": ["img_in", "txt_in", "proj_out", "norm_out", "time_text_embed"],
}


def load_image(path: str, longest: int = 1024):
    """Bounded, 16-aligned load for the Qwen path, which does its own bucketing.

    The FLUX path does NOT use this: it goes through face_utils.prepare_source,
    which frames around the faces and resizes once, straight to a Kontext bucket.
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


def _face_count(image_path: str) -> Optional[int]:
    """How many faces are in the source photo. ``None`` if it cannot be told.

    "Cannot be told" and "there is nobody in the picture" used to be the same
    answer, 0, and the caller only asked whether it was above 1 — so a photo of
    four Monster cans on a shop counter was sent to Kontext with "Keep the
    person's face ... Do not change who they are" appended, an instruction whose
    documented effect is that the model satisfies it by changing nothing. The two
    have to be distinguishable: a real 0 drops the clause, an unknown keeps the
    singular one, which is the behaviour that shipped — a worse prompt, never a
    missing picture.
    """
    try:
        from PIL import Image, ImageOps

        from src.chat import face_utils

        with Image.open(image_path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        return len(face_utils.detect(image))
    except Exception as exc:  # noqa: BLE001
        print(f"could not count faces ({exc}); assuming one", file=sys.stderr)
        return None


def select_identity_clause(faces: Optional[int], icfg: Dict[str, Any]) -> str:
    """Which "keep who they are" clause belongs on this edit, if any.

    A group shot needs the plural clause: "the person's face" in a picture of
    three people names nobody, and both edits that failed in the live group were
    two-subject photos. A photo with NOBODY in it needs neither — asked to
    "transform the monster cans into dildos" on a shot of a shop counter, the
    worker appended "Keep the person's face ... Do not change who they are",
    which per config.yaml's own note is the one instruction a model best
    satisfies by not editing at all.

    ``None`` means insightface could not tell, and keeps the singular clause —
    the behaviour that shipped, and a worse prompt rather than a missing picture.
    """
    if faces == 0:
        return ""
    if faces is not None and faces > 1:
        return str(icfg.get("identity_clause_plural",
                            DEFAULT_IDENTITY_CLAUSE_PLURAL) or "")
    return str(icfg.get("identity_clause", DEFAULT_IDENTITY_CLAUSE) or "")


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
    # Best-of-N is decided by ArcFace similarity to the source face. With no face
    # to compare against, pick_best has nothing to rank and falls through to the
    # first candidate — so every extra take is a render thrown away unlooked at.
    if reference is None:
        candidates_wanted = 1

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

    guidance = float(options.get("guidance_scale", 2.5))
    noop_threshold = float(options.get("noop_threshold", 0.0))

    def take(count: int, scale: float, base_seed: int, text: str = prompt):
        """``count`` renders from the loaded pipeline at one guidance setting."""
        return [
            (pipe(
                image=source, prompt=text,
                height=source.height, width=source.width, _auto_resize=False,
                guidance_scale=scale, num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(base_seed + index),
            ).images[0], base_seed + index)
            for index in range(count)
        ]

    # One model load, N takes. Loading the 15GB pipeline is most of the cost, so a
    # second candidate is far cheaper than a second request would be.
    candidates = take(candidates_wanted, guidance, seed)
    best, best_seed, score = face_utils.pick_best(
        reference, candidates, source=source, noop_threshold=noop_threshold)
    changed = face_utils.change_ratio(source, best)
    retried = False

    # An edit that changed nothing is a failure, and it used to be delivered as
    # a success: "Was the generation rejected? The image looks exactly the same".
    # The pipeline is already resident, so a harder retry costs one render rather
    # than another 15GB load. The identity clause is what the caller drops for
    # this pass — see imagegen.run.
    if noop_threshold > 0 and changed < noop_threshold:
        retry_scale = float(options.get("retry_guidance_scale", guidance + 1.0))
        print(json.dumps({"event": "noop_retry", "changed": round(changed, 4),
                          "guidance_scale": retry_scale}), flush=True)
        # Without the identity clause: "Keep the face exactly the same. Do not
        # change who they are" is precisely the instruction a model satisfies
        # best by changing nothing at all.
        retries = take(max(1, candidates_wanted), retry_scale, seed + 1000,
                       text=str(options.get("retry_prompt") or prompt))
        best, best_seed, score = face_utils.pick_best(
            reference, retries, source=source, noop_threshold=noop_threshold)
        changed = face_utils.change_ratio(source, best)
        retried = True

    info = {
        "seed": best_seed,
        "likeness": None if score is None else round(score, 4),
        "candidates": len(candidates),
        "source_size": [source.width, source.height],
        # How much of the picture moved, and whether it moved enough to count as
        # an edit at all. The caller turns a False here into an honest "I could
        # not make that edit" rather than sending the original back.
        "changed": round(changed, 4),
        "edited": bool(noop_threshold <= 0 or changed >= noop_threshold),
        "retried": retried,
    }

    if options.get("restore_face"):
        # Free Kontext before Klein loads: 15GB and 17GB do not both fit on a
        # 24GB card, and the restore pass is a different model entirely.
        del pipe, candidates
        free_cuda()
        best, restore_info = restore_with_bfs(best, image_path, reference, options)
        info.update(restore_info)

    best.save(out_path)
    return info


def free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# The card's own template, and the argument order is the counter-intuitive part:
# Picture 1 is the BODY being kept, Picture 2 is the FACE being donated.
_BFS_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, "
    "environment, and background. remove the head from Picture 1 completely and "
    "replace it with the head from Picture 2, strictly preserving the hair, eye "
    "color, nose structure of Picture 2. copy the direction of the eye, head "
    "rotation, micro expressions from Picture 1, high quality, sharp details, 4k."
)


def load_klein():
    """FLUX.2 Klein 9B, 8-bit and resident. Shared by the editor and the restore.

    Klein arrived here as the second stage of a face restore and is now also an
    editor in its own right — it follows an instruction far better than Kontext
    (adherence 4.98 vs 4.05 on the 40-cell grid) and only lost the bake-off on
    face likeness, which is not a criterion when the subject is a shop counter.
    """
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TfBnb

    quant = PipelineQuantizationConfig(quant_mapping={
        "transformer": bnb_8bit([]),
        "text_encoder": TfBnb(load_in_8bit=True),
    })
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16,
        quantization_config=quant)
    # Resident, not offloaded — see edit_with_flux for what the offload hooks do
    # to bitsandbytes weights.
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _run_bfs(edited, original, options: Dict[str, Any]):
    """The model call: Klein 9B plus the BFS LoRA, two images in, one out.

    Kept separate from ``restore_with_bfs`` so the decision of whether to KEEP a
    restore can be tested without a GPU — that guard is the part worth testing.
    """
    import torch

    pipe = load_klein()
    pipe.load_lora_weights(
        DEFAULT_RESTORE_LORA_REPO,
        weight_name=str(options.get("restore_lora") or DEFAULT_RESTORE_LORA))

    # Picture 1 is the BODY kept, Picture 2 the FACE donated. That order is the
    # card's and it is the opposite of what reading the prompt suggests.
    return pipe(
        image=[edited, original], prompt=_BFS_PROMPT,
        height=edited.height, width=edited.width,
        num_inference_steps=int(options.get("restore_steps", 28)),
        generator=torch.Generator("cpu").manual_seed(
            int(options.get("seed", 0)) + 977),
    ).images[0]


def restore_with_bfs(edited, source_path: str, reference, options: Dict[str, Any]):
    """Put the original face back onto an edit that changed everything else.

    The bake-off showed both editors trading identity against instruction-following
    because one prompt was being asked to do both jobs: Kontext scored 0.918
    likeness on a prison-yard edit by declining to add the requested injuries,
    while Klein followed every instruction and lost the face. Splitting the work
    removes the trade — the editor commits to the scene, and identity is restored
    afterwards by a model trained for exactly that.

    Returns ``(image, info)``. The image is the RESTORED one only when it actually
    scores better against the source face than the editor's own output did;
    otherwise the editor's output is returned untouched. That guard is what makes
    this stage safe to enable by default — a failed restore costs time, never
    quality.
    """
    from src.chat import face_utils

    before = face_utils.likeness(reference, edited)
    try:
        original = open_source_photo(source_path)
        restored = _run_bfs(edited, original, options)
    except Exception as exc:  # noqa: BLE001 — a failed restore must not lose the edit
        print(f"restore failed ({type(exc).__name__}: {exc}); keeping the edit",
              file=sys.stderr)
        return edited, {"restored": False, "restore_error": str(exc)[:160]}

    after = face_utils.likeness(reference, restored)
    gain_needed = float(options.get("restore_min_gain", 0.0))
    info = {
        "restored": False,
        "likeness_before_restore": None if before is None else round(before, 4),
        "likeness_after_restore": None if after is None else round(after, 4),
    }
    if before is None or after is None:
        # Nothing to compare — no face found in one of them. Keep the edit rather
        # than gamble on an unmeasurable swap.
        return edited, info
    if after - before <= gain_needed:
        return edited, info
    info["restored"] = True
    info["likeness"] = round(after, 4)
    return restored, info


def open_source_photo(path: str):
    """The original photo, upright, for use as the face donor."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


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
                   steps: int, seed: int, options: Dict[str, Any] = None,
                   repo: str = "Qwen/Qwen-Image-Edit-2509") -> Dict[str, Any]:
    """Qwen-Image-Edit in two passes.

    The 20B transformer needs 8-bit to render cleanly (NF4 turns it into a
    mosaic) and 8-bit occupies 19.4GB, which leaves no room for the 7B text
    encoder beside it. So the encoder runs first in a process of its own, and the
    transformer then loads into the whole card and works from what it wrote.
    """
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    embeds_path = str(Path(out_path).with_suffix(".embeds.pt"))
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--stage", "embed", "--repo", repo,
                    "--image", image_path, "--prompt", prompt,
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


def edit_with_qwen_2511(image_path: str, prompt: str, out_path: str,
                        steps: int, seed: int,
                        options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Qwen-Image-Edit-2511 — same shape as 2509, a later checkpoint.

    2509 adhered to the instruction better than anything else tested and scored
    0.138 on likeness: it understood the request and returned a stranger. 2511's
    stated change is consistency, which is the half it was losing.
    """
    return edit_with_qwen(image_path, prompt, out_path, steps, seed, options,
                          repo="Qwen/Qwen-Image-Edit-2511")


def edit_with_klein(image_path: str, prompt: str, out_path: str,
                    steps: int, seed: int,
                    options: Dict[str, Any] = None) -> Dict[str, Any]:
    """FLUX.2 Klein 9B as the editor. One pass, 8-bit, ~20GB, ~145s.

    No face machinery: this is the arm for edits where there is no identity to
    protect, so there is no reference embedding, no crop and no best-of-N. The
    no-op check still applies — an edit that returns the source unchanged is a
    failure whatever the subject was.
    """
    import torch

    from src.chat import face_utils

    options = options or {}
    source = face_utils.fit_to_kontext(open_source_photo(image_path))
    pipe = load_klein()
    result = pipe(
        image=[source], prompt=prompt,
        height=source.height, width=source.width,
        guidance_scale=float(options.get("guidance_scale", 2.5)),
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    result.save(out_path)

    changed = face_utils.change_ratio(source, result)
    noop_threshold = float(options.get("noop_threshold", 0.0))
    return {
        "seed": seed,
        "likeness": None,
        "candidates": 1,
        "source_size": list(source.size),
        "changed": round(changed, 4),
        "edited": noop_threshold <= 0 or changed >= noop_threshold,
        "retried": False,
    }


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


EDITORS = {
    "flux-kontext": edit_with_flux,
    "qwen-image-edit": edit_with_qwen,
    "qwen-image-edit-2511": edit_with_qwen_2511,
    "flux2-klein": edit_with_klein,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or edit one image.")
    parser.add_argument("--mode", default="edit", choices=["edit", "generate"])
    parser.add_argument("--stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo", default="Qwen/Qwen-Image-Edit-2509",
                        help=argparse.SUPPRESS)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--editor", default=None)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                        help="override the configured guidance; the caller raises "
                             "it for edits that change what people are DOING")
    parser.add_argument("--no-face-crop", action="store_true")
    parser.add_argument("--no-identity-clause", action="store_true")
    parser.add_argument("--restore-face", action="store_true",
                        help="after the edit, put the original face back "
                             "(kept only if it scores better)")
    args = parser.parse_args()

    if args.stage == "embed":
        _qwen_embed(args.repo, args.image, args.prompt, args.out)
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

    prompt = bare_prompt = args.prompt
    clause = select_identity_clause(_face_count(args.image), icfg)
    if clause and not args.no_identity_clause:
        prompt = f"{prompt.rstrip().rstrip('.')}. {clause}"

    options = {
        "candidates": args.candidates or int(icfg.get("candidates", 1)),
        "face_crop": not args.no_face_crop and bool(icfg.get("face_crop", True)),
        "face_min_ratio": float(icfg.get("face_min_ratio", 0.22)),
        "face_target_ratio": float(icfg.get("face_target_ratio", 0.32)),
        "guidance_scale": args.guidance_scale or float(icfg.get("guidance_scale", 2.5)),
        # An edit that comes back identical to the source is a failure. Below
        # this much change it is retried harder, and if that fails too the
        # caller says so instead of sending the original back as the "edit".
        "noop_threshold": float(icfg.get("noop_threshold", 0.0)),
        "retry_guidance_scale": float(icfg.get("retry_guidance_scale", 3.5)),
        "retry_prompt": bare_prompt,
        # The caller decides per request whether this edit is meant to change
        # the face; restoring it on a zombie would undo the whole point.
        "restore_face": args.restore_face and bool(icfg.get("restore_face", True)),
        "restore_lora": icfg.get("restore_lora") or DEFAULT_RESTORE_LORA,
        "restore_steps": int(icfg.get("restore_steps", 28)),
        "restore_min_gain": float(icfg.get("restore_min_gain", 0.0)),
        "seed": seed,
    }
    info = EDITORS[editor](args.image, prompt, args.out,
                           args.steps or int(icfg.get("edit_steps", 28)),
                           seed, options) or {}

    print(json.dumps({"ok": True, "out": args.out, "prompt": prompt, **info}))


if __name__ == "__main__":
    main()
