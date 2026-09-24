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
#   KAYA_EDGE=local|pi       local (default): WAHA + the tunnel run here, as before.
#                            pi: they run on the Raspberry Pi (deploy/pi), so this
#                            script removes any local copy instead: a second WAHA on
#                            the same session fights the Pi's for the login.
#   KAYA_PROD_LLAMA_URL      empty: the local `llama` service serves the model.
#                            …/upstream/kaya: the shared GPU broker (~/llm-broker)
#                            does, and `kaya-llama` is removed to free GPU1.
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
PROD_LLAMA_URL="$(env_value KAYA_PROD_LLAMA_URL)"
echo "🧭 edge=$KAYA_EDGE, model=${PROD_LLAMA_URL:-local llama service}"

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
else
  # The Pi owns WhatsApp and the public tunnel. Leave nothing here that could
  # log in to the same session or register as a second tunnel connector.
  docker rm -f kaya-cloudflared 2>/dev/null || true
fi

echo "🔨 Building image ..."
docker compose build kaya-prod

if [[ "$KAYA_EDGE" == "local" ]]; then
  # kaya-prod runs the WhatsApp bridge (UI + webhook); waha is its inbound gateway.
  echo "🚀 (Re)starting prod + WAHA + tunnel ..."
  docker compose --profile prod --profile waha-local --profile tunnel \
    up -d --force-recreate kaya-prod waha cloudflared
else
  echo "🚀 (Re)starting prod (WAHA and the tunnel run on the Pi) ..."
  docker compose --profile prod up -d --force-recreate kaya-prod
fi

if [[ "$PROD_LLAMA_URL" == */upstream/* ]]; then
  # The broker loads the model on demand; a resident kaya-llama would hold 13 GB
  # of GPU1 that the broker thinks is free.
  docker rm -f kaya-llama 2>/dev/null || true
  if curl -fsS --max-time 3 http://127.0.0.1:8200/health >/dev/null 2>&1; then
    echo "🦙 model served by the GPU broker ($PROD_LLAMA_URL)"
  else
    echo "⚠️  the GPU broker is not answering on :8200 — start it: cd ~/llm-broker && docker compose up -d" >&2
  fi
elif [[ "${KAYA_INFERENCE_BACKEND:-gguf}" == "gguf" ]]; then
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
