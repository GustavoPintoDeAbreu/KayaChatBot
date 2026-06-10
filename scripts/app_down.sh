#!/usr/bin/env bash
#
# Power down the Kaya web app, freeing the GPU.
#
#   scripts/app_down.sh dev    # stop the dev container
#   scripts/app_down.sh prod   # stop the prod container
#   scripts/app_down.sh all    # stop both apps and the tunnel
#
set -euo pipefail

ENV_NAME="${1:-}"
if [[ "$ENV_NAME" != "dev" && "$ENV_NAME" != "prod" && "$ENV_NAME" != "all" ]]; then
  echo "Usage: $0 <dev|prod|all>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$ENV_NAME" == "all" ]]; then
  echo "🛑 Stopping kaya-dev, kaya-prod and cloudflared ..."
  docker compose --profile dev --profile prod --profile tunnel stop kaya-dev kaya-prod cloudflared || true
  docker compose --profile dev --profile prod --profile tunnel rm -f kaya-dev kaya-prod cloudflared || true
else
  echo "🛑 Stopping kaya-${ENV_NAME} ..."
  docker compose --profile "$ENV_NAME" stop "kaya-${ENV_NAME}" || true
  docker compose --profile "$ENV_NAME" rm -f "kaya-${ENV_NAME}" || true
  # Leave the tunnel running only if the other app is still up; otherwise stop it.
  OTHER_ENV=$([[ "$ENV_NAME" == "dev" ]] && echo "prod" || echo "dev")
  if ! docker ps --format '{{.Names}}' | grep -qx "kaya-${OTHER_ENV}"; then
    echo "   No app left running — stopping cloudflared too."
    docker compose --profile tunnel stop cloudflared || true
  fi
fi

echo "✅ Done. GPU freed."
