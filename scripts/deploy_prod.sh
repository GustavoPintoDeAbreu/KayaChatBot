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

export KAYA_VERSION="$(git rev-parse --short HEAD)"
echo "🔖 Deploying commit $KAYA_VERSION"

# Free the GPU and release container names/ports from any other env. dev and the
# on-demand WhatsApp dev container (kaya-whatsapp) may belong to a DIFFERENT
# compose project, so stop them by name. WAHA is recreated below in the prod
# project (its linked-device session persists in the ./data/waha volume).
echo "🛑 Stopping other envs that share the single GPU / names ..."
docker rm -f kaya-dev kaya-whatsapp kaya-waha 2>/dev/null || true

echo "🔨 Building image ..."
docker compose build kaya-prod

# kaya-prod runs the WhatsApp bridge (UI + webhook); waha is its inbound gateway.
echo "🚀 (Re)starting prod + WAHA + tunnel ..."
docker compose --profile prod --profile tunnel up -d --force-recreate kaya-prod waha cloudflared

echo
echo "✅ Prod is now serving commit $KAYA_VERSION (ref: $REF)."
echo "   Local:  http://localhost:7860"
echo "   Public: your prod Cloudflare hostname (see DEPLOYMENT.md)"
echo "   Logs:   docker compose logs -f kaya-prod"
