#!/usr/bin/env bash
#
# Deploy a given commit/branch/tag as the live prod web app.
#
#   scripts/deploy_prod.sh [ref]      # ref defaults to "main"
#
# Operates on a dedicated prod checkout (KAYA_PROD_DIR, default ~/kaya-prod) that
# is separate from your development copy, so the live site always runs the
# deployed commit and you can keep editing elsewhere without affecting it.
#
# One-time setup (run once):
#   git clone git@github.com:GustavoPintoDeAbreu/KayaChatBot.git ~/kaya-prod
#   ln -s ~/Desktop/KayaChatBot/models ~/kaya-prod/models   # share the 42GB models
#   ln -s ~/Desktop/KayaChatBot/data   ~/kaya-prod/data     # share data/rag_db
#   cp ~/Desktop/KayaChatBot/.env ~/kaya-prod/.env          # or let CI write it
#
# Requires ~/kaya-prod/.env with KAYA_WEB_USER/PASS and CLOUDFLARE_TUNNEL_TOKEN.
#
# Two settings in that .env decide what else this script manages:
#   KAYA_EDGE=local|pi       local (default): WAHA runs here, as before.
#                            pi: WAHA runs on the Raspberry Pi (deploy/pi), so this
#                            script removes any local copy instead: a second WAHA on
#                            the same session fights the Pi's for the login.
#   KAYA_TUNNEL=local|pi     where cloudflared runs; defaults to KAYA_EDGE. They are
#                            separate because the tunnel's move needs the Cloudflare
#                            dashboard and WAHA's does not.
#   KAYA_INFERENCE_BACKEND   ollama (via KAYA_PROD_OLLAMA_URL) or gguf (via
#                            KAYA_PROD_LLAMA_URL; empty = the local `llama` service).
#                            A …/upstream/kaya URL means the shared GPU broker
#                            (~/llm-broker) serves it, and `kaya-llama` is removed.
set -euo pipefail

REF="${1:-main}"
PROD_DIR="${KAYA_PROD_DIR:-$HOME/kaya-prod}"

if [[ ! -d "$PROD_DIR/.git" ]]; then
  echo "❌ $PROD_DIR is not a git checkout. Run the one-time setup in this script's header." >&2
  exit 1
fi
cd "$PROD_DIR"

if [[ ! -f .env ]]; then
  echo "❌ $PROD_DIR/.env missing (needs KAYA_WEB_USER/PASS + CLOUDFLARE_TUNNEL_TOKEN)." >&2
  exit 1
fi

echo "📥 Fetching and checking out '$REF' in $PROD_DIR ..."
git fetch origin --prune --tags
if git rev-parse --verify --quiet "origin/$REF" >/dev/null; then
  git checkout -B "$REF" "origin/$REF"      # remote branch
else
  git checkout -f "$REF"                      # tag or commit SHA
fi

# Map the pinned GPU UUIDs to current indices (the runtime rejects UUIDs, and
# indices are not stable across reboots). Operates on $PROD_DIR/.env.
if [[ -f scripts/gpu_env.sh ]]; then
  source scripts/gpu_env.sh
  echo "🎯 GPU pin: prod=index:${KAYA_GPU_PROD:-all} dev=index:${KAYA_GPU_DEV:-all}"
fi

env_value() { sed -n "s/^$1=//p" .env | tail -1 | tr -d '"'"'"; }
KAYA_EDGE="$(env_value KAYA_EDGE)"; KAYA_EDGE="${KAYA_EDGE:-local}"
KAYA_TUNNEL="$(env_value KAYA_TUNNEL)"; KAYA_TUNNEL="${KAYA_TUNNEL:-$KAYA_EDGE}"
PROD_LLAMA_URL="$(env_value KAYA_PROD_LLAMA_URL)"
PROD_OLLAMA_URL="$(env_value KAYA_PROD_OLLAMA_URL)"
PROD_BACKEND="$(env_value KAYA_INFERENCE_BACKEND)"
if [[ "$PROD_BACKEND" == "ollama" ]]; then MODEL_URL="$PROD_OLLAMA_URL"; else MODEL_URL="$PROD_LLAMA_URL"; fi
echo "🧭 edge=$KAYA_EDGE, tunnel=$KAYA_TUNNEL, backend=${PROD_BACKEND:-gguf}, model=${MODEL_URL:-local llama service}"

docker network inspect llm >/dev/null 2>&1 || docker network create llm >/dev/null

export KAYA_VERSION="$(git rev-parse --short HEAD)"
echo "🔖 Deploying commit $KAYA_VERSION"

# Release container names/ports that would collide with prod. kaya-dev is NOT
# stopped: it owns the other GPU (KAYA_GPU_DEV) and binds 7861, so it survives a
# prod deploy untouched. kaya-whatsapp still binds 7860 and may belong to a
# DIFFERENT compose project, so stop it by name.
echo "🛑 Releasing conflicting container names/ports ..."
docker rm -f kaya-whatsapp kaya-waha 2>/dev/null || true

if [[ "$KAYA_EDGE" == "local" ]]; then
  # WAHA tracks WhatsApp's protocol, which moves fast, and the compose file pins the
  # floating `:latest` tag — which only advances when something actually pulls it.
  # A stale image cannot complete the handshake and fails as an endless
  # "Connection Failure" login loop that looks exactly like an expired login,
  # tempting a needless QR re-scan. Refresh it on every deploy.
  echo "🔄 Refreshing the WAHA image (stale WAHA looks like a broken login) ..."
  docker pull devlikeapro/waha:latest >/dev/null 2>&1 \
    && echo "   ✓ WAHA image up to date" \
    || echo "   ⚠️  pull failed — continuing with the cached image" >&2
fi
if [[ "$KAYA_TUNNEL" == "pi" ]]; then
  # A second connector here would take half the tunnel's traffic.
  docker rm -f kaya-cloudflared 2>/dev/null || true
fi

echo "🔨 Building image ..."
docker compose build kaya-prod

services=(kaya-prod); profiles=(--profile prod)
[[ "$KAYA_EDGE" == "local" ]] && { services+=(waha); profiles+=(--profile waha-local); }
echo "🚀 (Re)starting ${services[*]} ..."
docker compose "${profiles[@]}" up -d --force-recreate "${services[@]}"
if [[ "$KAYA_TUNNEL" == "local" ]]; then
  # Not force-recreated: restarting the connector drops every open page.
  docker compose --profile tunnel up -d cloudflared
fi

if [[ "$MODEL_URL" == */upstream/* ]]; then
  # The broker loads the model on demand; a resident kaya-llama would hold 13 GB
  # of GPU1 that the broker thinks is free.
  docker rm -f kaya-llama 2>/dev/null || true
  if curl -fsS --max-time 3 http://127.0.0.1:8200/health >/dev/null 2>&1; then
    echo "🦙 model served by the GPU broker ($MODEL_URL)"
  else
    echo "⚠️  the GPU broker is not answering on :8200 — start it: cd ~/llm-broker && docker compose up -d" >&2
  fi
elif [[ "${PROD_BACKEND:-gguf}" == "gguf" ]]; then
  # Prod generates via the llama.cpp gguf server (KAYA_INFERENCE_BACKEND=gguf, set
  # on the kaya-prod service). Start it too. Not force-recreated, so a redeploy
  # leaves the model loaded. Export KAYA_INFERENCE_BACKEND=hf to skip + roll back.
  echo "🦙 backend=gguf → ensuring the llama.cpp server is up ..."
  docker compose --profile gguf up -d llama
fi

echo
echo "✅ Prod is now serving commit $KAYA_VERSION (ref: $REF)."
echo "   Local:  http://localhost:7860"
echo "   Public: your prod Cloudflare hostname (see DEPLOYMENT.md)"
echo "   Logs:   docker compose logs -f kaya-prod"
