#!/usr/bin/env bash
# Download the image-editing candidates for the Phase 5 bake-off.
#
# Each --include/--exclude carries ONE pattern: the CLI reads a bare second
# pattern as a filename to fetch, and fails the whole repo with "File not found".
#
# Patterns matter here: several of these repos ship the same weights three or
# four times over (flax, onnx, openvino, a single-file dupe). Pulling a repo
# whole costs 77GB for SDXL alone, against ~7GB for the fp16 diffusers layout we
# actually load.
#
# FLUX.1-Kontext-dev is gated: accept the licence on huggingface.co while logged
# in, then `hf auth login`. Without a token it is skipped rather than failing the
# run, so the other candidates still download.
set -uo pipefail

cd "$(dirname "$0")/.."
HF="kaya_chatbot_env/bin/hf"
LOG_DIR="${1:-reports/image_bakeoff}"
mkdir -p "$LOG_DIR"

# Unauthenticated Hub requests are rate-limited, and these are tens of GB — a
# mid-download 429 is routine, not a reason to lose the repo. Retry with backoff;
# already-downloaded files are cached, so a retry resumes rather than restarts.
fetch() {
    local repo="$1"; shift
    echo "=== $repo"
    for attempt in 1 2 3 4 5; do
        if "$HF" download "$repo" "$@" --quiet; then
            echo "--- done: $repo"
            return 0
        fi
        echo "    attempt $attempt failed; retrying in $((attempt * 60))s"
        sleep $((attempt * 60))
    done
    echo "!!! FAILED: $repo (gated, or the Hub kept refusing)"
    return 1
}

# SDXL + IP-Adapter — the fast, mature baseline. fp16 diffusers layout only.
fetch stabilityai/stable-diffusion-xl-base-1.0 \
    --include "*.json" --include "*.txt" --include "*.fp16.safetensors" \
    --include "tokenizer*/*" --include "scheduler/*"
fetch madebyollin/sdxl-vae-fp16-fix
# models/image_encoder is the ViT-H encoder the *-plus-face* adapters need — the
# face-preserving ones, i.e. the only reason SDXL is in a likeness bench at all.
# It lives outside sdxl_models/ despite being used by the SDXL adapters.
fetch h94/IP-Adapter --include "sdxl_models/*" --include "models/image_encoder/*"

# Z-Image Turbo — 6B, the fast end of the instruction-editing candidates.
fetch Tongyi-MAI/Z-Image-Turbo --exclude "*.onnx*" --exclude "*.msgpack"

# Qwen-Image-Edit-2509 — 20B instruction editor, ungated. Loaded 4-bit.
fetch Qwen/Qwen-Image-Edit-2509 --exclude "*.onnx*" --exclude "*.msgpack"

# FLUX.1 Kontext — the favourite on paper, but gated. Skip the 23.8GB
# single-file checkpoint; diffusers loads the sharded transformer/ directory.
# `hf auth whoami` exits 0 while printing "Not logged in", so the exit code says
# nothing — the output is what has to be checked.
if ! "$HF" auth whoami 2>&1 | grep -qi "not logged in"; then
    fetch black-forest-labs/FLUX.1-Kontext-dev --exclude "flux1-kontext-dev.safetensors"
else
    echo "=== black-forest-labs/FLUX.1-Kontext-dev SKIPPED — no HF token."
    echo "    Accept the licence at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev"
    echo "    then: kaya_chatbot_env/bin/hf auth login"
fi

echo
echo "cache now: $(du -sh ~/.cache/huggingface/hub 2>/dev/null | cut -f1)"
df -h / | tail -1
