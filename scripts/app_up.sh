#!/usr/bin/env bash
#
# Power up the Kaya web app on demand.
#
#   scripts/app_up.sh dev    # start the dev container  (port 7861)
#   scripts/app_up.sh prod   # start the prod container (port 7860)
#
# Starts the requested app container plus the Cloudflare Tunnel sidecar so the
# UI is reachable from another computer. dev and prod own separate GPUs
# (KAYA_GPU_DEV / KAYA_GPU_PROD in .env), so they can run at the same time; this
# only refuses when they would actually collide on the same card.
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

# Map the pinned GPU UUIDs to their current indices (this runtime cannot take a
# UUID in NVIDIA_VISIBLE_DEVICES, and indices move across reboots).
source "$REPO_ROOT/scripts/gpu_env.sh"

OTHER_ENV=$([[ "$ENV_NAME" == "dev" ]] && echo "prod" || echo "dev")

# dev and prod own different cards (KAYA_GPU_DEV / KAYA_GPU_PROD), so they can run
# at the same time. Only refuse when they would actually land on the same GPU.
GPU_DEV="$(grep -E '^KAYA_GPU_DEV=' .env | tail -1 | cut -d= -f2-)"
GPU_PROD="$(grep -E '^KAYA_GPU_PROD=' .env | tail -1 | cut -d= -f2-)"

if docker ps --format '{{.Names}}' | grep -qx "kaya-${OTHER_ENV}"; then
  if [[ -z "$GPU_DEV" || -z "$GPU_PROD" || "$GPU_DEV" == "$GPU_PROD" ]]; then
    echo "❌ kaya-${OTHER_ENV} is running and both envs resolve to the same GPU." >&2
    echo "   Set KAYA_GPU_DEV and KAYA_GPU_PROD to different UUIDs in .env," >&2
    echo "   or stop it first (scripts/app_down.sh ${OTHER_ENV})." >&2
    echo "   List UUIDs: nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv" >&2
    exit 1
  fi
  echo "ℹ️  kaya-${OTHER_ENV} is running on the other card — starting alongside it."
fi

# kaya-whatsapp is the on-demand dev bridge: it runs on the DEV card and binds
# 7860, so it collides with prod on the port and with dev on the card.
if docker ps --format '{{.Names}}' | grep -qx "kaya-whatsapp"; then
  if [[ "$ENV_NAME" == "prod" ]]; then
    echo "❌ kaya-whatsapp is running and binds port 7860, the same port as prod." >&2
  else
    echo "❌ kaya-whatsapp is running on the dev card — two models would not fit." >&2
  fi
  echo "   Stop it first: docker rm -f kaya-whatsapp" >&2
  exit 1
fi

PORT=$([[ "$ENV_NAME" == "dev" ]] && echo 7861 || echo 7860)

# Expose the running commit to the app (shown in the UI header). Falls back to
# "unknown" inside the container if unset.
export KAYA_VERSION="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# One Cloudflare Tunnel serves BOTH hostnames (ingress is configured remotely), so
# only ever start it once. Now that dev and prod run side by side they belong to
# different compose projects, and both trying to create /kaya-cloudflared fails
# with a container-name conflict.
if docker ps --format '{{.Names}}' | grep -qx "kaya-cloudflared"; then
  echo "🚀 Powering up kaya-${ENV_NAME} (commit ${KAYA_VERSION}); tunnel already running ..."
  docker compose --profile "$ENV_NAME" up -d "kaya-${ENV_NAME}"
else
  echo "🚀 Powering up kaya-${ENV_NAME} + cloudflared (commit ${KAYA_VERSION}) ..."
  docker compose --profile "$ENV_NAME" --profile tunnel up -d "kaya-${ENV_NAME}" cloudflared
fi

echo
echo "✅ kaya-${ENV_NAME} is starting (model load takes ~1 min)."
echo "   Local:  http://localhost:${PORT}"
echo "   Remote: via your Cloudflare hostname (see DEPLOYMENT.md)."
echo "   Logs:   docker compose logs -f kaya-${ENV_NAME}"
echo "   Stop:   scripts/app_down.sh ${ENV_NAME}"
