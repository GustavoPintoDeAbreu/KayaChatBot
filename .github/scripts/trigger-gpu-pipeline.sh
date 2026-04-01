#!/usr/bin/env bash
# Dispatch a GPU pipeline job to the self-hosted runner via GitHub Actions workflow_dispatch.
#
# Usage:
#   bash .github/scripts/trigger-gpu-pipeline.sh <mode> [--wait]
#
# Modes (maps directly to gpu-pipeline.yml workflow_dispatch input):
#   finetune          Run LoRA fine-tuning (up to 240 min — long, do NOT use --wait)
#   full-pipeline     Run full data + training pipeline (up to 240 min — long)
#   evaluate          Run full pytest suite in Docker (~10 min — use --wait)
#   inference-test    Test model inference (~10 min — use --wait)
#   benchmark         Run conversation benchmark (~60 min)
#   generate-knowledge  Regenerate group_knowledge.json via xAI Grok (~30 min — use --wait)
#   build-vectordb    Rebuild ChromaDB vector database (~15 min — use --wait)
#
# Options:
#   --wait   Poll until the run completes and print a summary. Only use for short modes.
#            Long modes (finetune, full-pipeline) should be dispatched without --wait.
#
# Requirements:
#   - gh CLI authenticated (GITHUB_TOKEN in env, or logged in via `gh auth login`)
#   - Repository remote named "origin" must point to GitHub
#
# Examples:
#   bash .github/scripts/trigger-gpu-pipeline.sh evaluate --wait
#   bash .github/scripts/trigger-gpu-pipeline.sh finetune
#   bash .github/scripts/trigger-gpu-pipeline.sh generate-knowledge --wait
#   bash .github/scripts/trigger-gpu-pipeline.sh build-vectordb --wait

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
VALID_MODES="finetune full-pipeline evaluate inference-test benchmark generate-knowledge build-vectordb"
LONG_MODES="finetune full-pipeline benchmark"

MODE="${1:-}"
WAIT=false

if [[ -z "$MODE" ]]; then
  echo "Error: <mode> is required." >&2
  echo "Usage: bash .github/scripts/trigger-gpu-pipeline.sh <mode> [--wait]" >&2
  echo "Valid modes: $VALID_MODES" >&2
  exit 1
fi

# Validate mode
if ! echo "$VALID_MODES" | tr ' ' '\n' | grep -qx "$MODE"; then
  echo "Error: unknown mode '$MODE'." >&2
  echo "Valid modes: $VALID_MODES" >&2
  exit 1
fi

shift
for arg in "$@"; do
  case "$arg" in
    --wait) WAIT=true ;;
    *) echo "Error: unknown option '$arg'" >&2; exit 1 ;;
  esac
done

# Warn if --wait is used with a long mode
if [[ "$WAIT" == "true" ]] && echo "$LONG_MODES" | tr ' ' '\n' | grep -qx "$MODE"; then
  echo "Warning: mode '$MODE' can take up to 240 minutes. --wait will block until it finishes." >&2
fi

# ---------------------------------------------------------------------------
# Derive repository slug from the git remote
# ---------------------------------------------------------------------------
REPO=$(git remote get-url origin 2>/dev/null | \
  sed -E 's|.*github\.com[:/]||; s|\.git$||')

if [[ -z "$REPO" ]]; then
  echo "Error: could not determine GitHub repository from git remote 'origin'." >&2
  exit 1
fi

echo "Repository : $REPO"
echo "Mode       : $MODE"
echo "Wait       : $WAIT"
echo ""

# ---------------------------------------------------------------------------
# Dispatch the workflow
# ---------------------------------------------------------------------------
echo "Dispatching gpu-pipeline.yml with mode='$MODE' ..."
gh workflow run gpu-pipeline.yml \
  --repo "$REPO" \
  -f "mode=$MODE"

# Give GitHub a moment to register the run
sleep 6

# Get the most-recently triggered run of this workflow
RUN_JSON=$(gh run list \
  --repo "$REPO" \
  --workflow gpu-pipeline.yml \
  --limit 1 \
  --json databaseId,url,status,conclusion)

RUN_ID=$(echo "$RUN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['databaseId'])")
RUN_URL=$(echo "$RUN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['url'])")

echo "Run started: $RUN_URL"
echo "Run ID     : $RUN_ID"
echo ""

# ---------------------------------------------------------------------------
# Optionally wait for completion
# ---------------------------------------------------------------------------
if [[ "$WAIT" == "true" ]]; then
  echo "Waiting for run $RUN_ID to complete (this may take several minutes) ..."
  # gh run watch exits 0 on success, non-zero on failure; both are valid outcomes here
  gh run watch "$RUN_ID" --repo "$REPO" --exit-status || true

  echo ""
  echo "=== Run Summary ==="
  gh run view "$RUN_ID" --repo "$REPO"
else
  echo "Run dispatched. Results will be posted to the PR once the self-hosted runner completes."
  echo "To check progress manually: gh run watch $RUN_ID --repo $REPO"
  echo "Or visit: $RUN_URL"
fi
