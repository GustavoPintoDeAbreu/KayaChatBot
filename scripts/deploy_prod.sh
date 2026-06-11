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

# Free the GPU: prod is the always-on env, so stop dev if it's up (one GPU).
if docker ps --format '{{.Names}}' | grep -qx "kaya-dev"; then
  echo "🛑 Stopping kaya-dev (shares the single GPU) ..."
  docker compose --profile dev stop kaya-dev || true
fi

echo "🔨 Building image ..."
docker compose build kaya-prod

echo "🚀 (Re)starting prod + tunnel ..."
docker compose --profile prod --profile tunnel up -d --force-recreate kaya-prod cloudflared

echo
echo "✅ Prod is now serving commit $KAYA_VERSION (ref: $REF)."
echo "   Local:  http://localhost:7860"
echo "   Public: your prod Cloudflare hostname (see DEPLOYMENT.md)"
echo "   Logs:   docker compose logs -f kaya-prod"
