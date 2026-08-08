#!/usr/bin/env bash
#
# Download the GGUF candidates scored by scripts/model_bakeoff.py into models/gguf/.
#
#   scripts/fetch_bakeoff_models.sh            # all arms
#   scripts/fetch_bakeoff_models.sh 31b-q8     # a subset, by arm tag
#
# Files land flat in models/gguf/ (the dir both llama services mount as /models).
# Already-present files are skipped, so this is safe to re-run after an
# interruption. ~262GB for the full set.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/models/gguf"
HF="$REPO_ROOT/kaya_chatbot_env/bin/hf"
mkdir -p "$DEST"

# tag|repo|filename
CANDIDATES=(
  "12b-q6|unsloth/gemma-4-12b-it-GGUF|gemma-4-12b-it-Q6_K.gguf"
  "26b-a4b-q4|unsloth/gemma-4-26B-A4B-it-qat-GGUF|gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
  "26b-a4b-q6|unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf"
  "31b-q4|unsloth/gemma-4-31B-it-qat-GGUF|gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
  "31b-q5|unsloth/gemma-4-31B-it-GGUF|gemma-4-31B-it-Q5_K_M.gguf"
  "31b-q8|unsloth/gemma-4-31B-it-GGUF|gemma-4-31B-it-Q8_0.gguf"
  "70b-abl-iq4|bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF|Llama-3.3-70B-Instruct-abliterated-IQ4_XS.gguf"
  "70b-abl-q4km|bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF|Llama-3.3-70B-Instruct-abliterated-Q4_K_M.gguf"
  # Sharded: fetch both parts; llama.cpp follows -00002- automatically from -00001-.
  "gptoss-120b|unsloth/gpt-oss-120b-GGUF|Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
  "gptoss-120b|unsloth/gpt-oss-120b-GGUF|Q4_K_M/gpt-oss-120b-Q4_K_M-00002-of-00002.gguf"
)

WANTED=("$@")
want() {
  [[ ${#WANTED[@]} -eq 0 ]] && return 0
  local t
  for t in "${WANTED[@]}"; do [[ "$t" == "$1" ]] && return 0; done
  return 1
}

failed=()
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r tag repo file <<< "$entry"
  want "$tag" || continue
  base="$(basename "$file")"

  if [[ -f "$DEST/$base" ]]; then
    echo "✓ $tag: $base already present ($(du -h "$DEST/$base" | cut -f1))"
    continue
  fi

  avail_gb=$(df --output=avail -BG "$DEST" | tail -1 | tr -dc '0-9')
  if (( avail_gb < 60 )); then
    echo "❌ only ${avail_gb}GB free — refusing to start another download." >&2
    failed+=("$tag:diskfull")
    break
  fi

  echo "⬇️  $tag: $repo :: $file"
  if "$HF" download "$repo" "$file" --local-dir "$DEST" >/dev/null; then
    # hf preserves the repo subdir (e.g. Q4_K_M/); flatten so /models/<name> works.
    if [[ "$file" == */* && -f "$DEST/$file" ]]; then
      mv -f "$DEST/$file" "$DEST/$base"
      rmdir -p "$DEST/$(dirname "$file")" 2>/dev/null || true
    fi
    echo "   ✓ $base ($(du -h "$DEST/$base" | cut -f1))"
  else
    echo "   ✗ FAILED $tag" >&2
    failed+=("$tag")
  fi
done

echo
echo "=== models/gguf ==="
ls -lh "$DEST"/*.gguf 2>/dev/null | awk '{print "  "$5"\t"$9}'
df -h --output=avail "$DEST" | tail -1 | xargs echo "free:"
if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "All requested candidates present."
