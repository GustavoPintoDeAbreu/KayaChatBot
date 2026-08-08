#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and export a quantized GGUF for serving.

Three steps, matching the path used for the current prod model:

  1. Load exactly as prod does (Unsloth FastModel, 4-bit base resolved from
     adapter_config.json + the LoRA), then dequantize and merge to 16-bit via
     save_pretrained_merged.
  2. convert_hf_to_gguf.py -> an f16 GGUF.
  3. llama-quantize -> the target quant, written into models/gguf/.

Steps 2 and 3 run inside the llama.cpp container that already serves the models,
so no local llama.cpp build is needed; models/ is bind-mounted.

    kaya_chatbot_env/bin/python scripts/export_gguf.py --profile gemma4-31b-wpp --quant Q4_K_M
    kaya_chatbot_env/bin/python scripts/export_gguf.py --model-dir models/kaya_x --quant Q5_K_M --keep

Intermediates are large (a merged 31B is ~62GB, its f16 GGUF another ~62GB) and
are deleted unless --keep is passed. Pick the quant to match what the winning
bake-off arm actually served: a 31B at Q6_K is 25GB and will not fit one card.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config

BASE_DIR = Path(__file__).parent.parent
GGUF_DIR = BASE_DIR / "models" / "gguf"
WORK_DIR = BASE_DIR / "models" / "_export_work"
CONVERT = Path.home() / "kaya-gguf-poc" / "llamacpp-convert" / "convert_hf_to_gguf.py"
LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda"


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def free_gb(path: Path) -> float:
    st = shutil.disk_usage(path)
    return st.free / 1e9


def merge_16bit(model_dir: Path, max_seq_length: int, out_dir: Path) -> None:
    """Dequantize + merge the adapter into a 16-bit HF checkpoint."""
    from unsloth import FastModel

    log(f"loading {model_dir} (4-bit base + LoRA) …")
    t0 = time.perf_counter()
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    log(f"loaded in {time.perf_counter() - t0:.0f}s; merging to 16-bit → {out_dir}")
    t0 = time.perf_counter()
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    log(f"merged in {time.perf_counter() - t0:.0f}s")


def convert_to_f16(merged_dir: Path, f16_path: Path) -> None:
    if not CONVERT.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {CONVERT}")
    log(f"converting → {f16_path.name}")
    subprocess.run(
        [sys.executable, str(CONVERT), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        check=True, cwd=str(BASE_DIR),
    )


def quantize(f16_path: Path, out_path: Path, quant: str) -> None:
    """Quantize inside the llama.cpp image so no local build is required."""
    log(f"quantizing → {out_path.name} ({quant})")
    subprocess.run(
        ["docker", "run", "--rm",
         "-v", f"{f16_path.parent}:/work",
         "-v", f"{GGUF_DIR}:/out",
         "--entrypoint", "/app/llama-quantize", LLAMA_IMAGE,
         f"/work/{f16_path.name}", f"/out/{out_path.name}", quant],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA and export a quantized GGUF.")
    ap.add_argument("--profile", default=None, help="Model profile (sets model dir + seq length).")
    ap.add_argument("--model-dir", default=None, help="Adapter dir (overrides the profile's).")
    ap.add_argument("--quant", default="Q4_K_M", help="Target quant, e.g. Q4_K_M / Q5_K_M / Q6_K.")
    ap.add_argument("--out-name", default=None, help="Output filename in models/gguf/.")
    ap.add_argument("--keep", action="store_true", help="Keep the merged + f16 intermediates.")
    args = ap.parse_args()

    config = load_config(str(BASE_DIR / "config.yaml"), profile_override=args.profile)
    model_dir = Path(args.model_dir or config["training"]["output_dir"])
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir
    if not (model_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter_config.json in {model_dir}")

    stem = args.out_name or f"{model_dir.name}-{args.quant}.gguf"
    if not stem.endswith(".gguf"):
        stem += ".gguf"
    out_path = GGUF_DIR / stem
    if out_path.exists():
        log(f"{out_path} already exists — nothing to do.")
        return

    # A merged 31B is ~62GB and its f16 GGUF another ~62GB; fail early rather
    # than halfway through a multi-hour export.
    need = 150.0
    if free_gb(BASE_DIR) < need:
        raise SystemExit(f"need ~{need:.0f}GB free for intermediates, "
                         f"have {free_gb(BASE_DIR):.0f}GB")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    merged = WORK_DIR / f"{model_dir.name}_merged16"
    f16 = WORK_DIR / f"{model_dir.name}-f16.gguf"

    try:
        if not merged.exists():
            merge_16bit(model_dir, config["model"]["max_seq_length"], merged)
        else:
            log(f"reusing existing merge at {merged}")
        if not f16.exists():
            convert_to_f16(merged, f16)
        else:
            log(f"reusing existing f16 at {f16}")
        quantize(f16, out_path, args.quant)
    finally:
        if not args.keep:
            shutil.rmtree(merged, ignore_errors=True)
            f16.unlink(missing_ok=True)
            log("removed intermediates (pass --keep to retain them)")

    size_gb = out_path.stat().st_size / 1e9 if out_path.exists() else 0.0
    log(f"done → {out_path} ({size_gb:.1f}GB)")
    log(f"score it:  kaya_chatbot_env/bin/python scripts/model_bakeoff.py --only <new-arm>")


if __name__ == "__main__":
    main()
