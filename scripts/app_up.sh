#!/usr/bin/env bash
#
# Power up the Kaya web app on demand.
#
#   scripts/app_up.sh dev    # start the dev container  (port 7861)
#   scripts/app_up.sh prod   # start the prod container (port 7860)
#
# Starts the requested app container plus the Cloudflare Tunnel sidecar so the
# UI is reachable from another computer. dev and prod share the single GPU, so
# this refuses to start one while the other is already running.
#
# Requires .env with CLOUDFLARE_TUNNEL_TOKEN, KAYA_WEB_USER and KAYA_WEB_PASS.
set -euo pipefail

ENV_NAME="${1:-}"
if [[ "$ENV_NAME" != "dev" && "$ENV_NAME" != "prod" ]]; then
  echo "Usage: $0 <dev|prod>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
  echo "❌ .env not found. Create it from .env.example with the required secrets." >&2
  exit 1
fi

OTHER_ENV=$([[ "$ENV_NAME" == "dev" ]] && echo "prod" || echo "dev")
if docker ps --format '{{.Names}}' | grep -qx "kaya-${OTHER_ENV}"; then
  echo "❌ kaya-${OTHER_ENV} is already running and shares the single GPU." >&2
  echo "   Stop it first: scripts/app_down.sh ${OTHER_ENV}" >&2
  exit 1
fi

PORT=$([[ "$ENV_NAME" == "dev" ]] && echo 7861 || echo 7860)

# Expose the running commit to the app (shown in the UI header). Falls back to
# "unknown" inside the container if unset.
export KAYA_VERSION="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "🚀 Powering up kaya-${ENV_NAME} + cloudflared (commit ${KAYA_VERSION}) ..."
docker compose --profile "$ENV_NAME" --profile tunnel up -d "kaya-${ENV_NAME}" cloudflared

echo
echo "✅ kaya-${ENV_NAME} is starting (model load takes ~1 min)."
echo "   Local:  http://localhost:${PORT}"
echo "   Remote: via your Cloudflare hostname (see DEPLOYMENT.md)."
echo "   Logs:   docker compose logs -f kaya-${ENV_NAME}"
echo "   Stop:   scripts/app_down.sh ${ENV_NAME}"
