#!/usr/bin/env bash
#
# Deploy the always-on front door to the Raspberry Pi (deploy/pi/README.md).
#
#   scripts/deploy_pi.sh [--init-env]
#
# Copies the gateway's code and config.yaml to pi5:~/kaya-gateway, the runtime
# whitelist from ~/kaya-prod/data, then rebuilds and restarts the containers (the
# Wake-on-LAN timer and the LAN firewall are lab-infra's pi/install.sh since 2026-10-22). Which of WAHA and the
# tunnel run is decided by COMPOSE_PROFILES in the Pi's deploy/pi/.env.
#
# --init-env writes that .env from ~/kaya-prod/.env when it does not exist yet
# (secrets never pass through git). An existing .env is never overwritten.
set -euo pipefail

PI="${PI_HOST:-gustavo@pi5.local}"
REMOTE_DIR="${PI_DIR:-kaya-gateway}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROD_DIR="${KAYA_PROD_DIR:-$HOME/kaya-prod}"
cd "$REPO_ROOT"

ssh -o BatchMode=yes -o ConnectTimeout=5 "$PI" true || { echo "❌ cannot reach $PI" >&2; exit 1; }

echo "📦 Syncing code to $PI:~/$REMOTE_DIR ..."
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'deploy/pi/.env' --exclude 'deploy/pi/data/' \
  --include 'src/***' --include 'config.yaml' \
  --include 'deploy/' --include 'deploy/pi/***' --exclude '*' \
  ./ "$PI:$REMOTE_DIR/"

if [[ "${1:-}" == "--init-env" ]]; then
  if ssh "$PI" "test -f $REMOTE_DIR/deploy/pi/.env"; then
    echo "ℹ️  $REMOTE_DIR/deploy/pi/.env already exists; not touching it"
  else
    value() { sed -n "s/^$1=//p" "$PROD_DIR/.env" | tail -1; }
    relay="$(value KAYA_RELAY_TOKEN)"
    [[ -n "$relay" ]] || { echo "❌ set KAYA_RELAY_TOKEN in $PROD_DIR/.env first" >&2; exit 1; }
    {
      echo "COMPOSE_PROFILES="
      echo "PI_LAN_IP=${PI_LAN_IP:-192.168.1.238}"
      echo "KAYA_PC_URL=${KAYA_PC_URL:-http://192.168.1.149:7860}"
      for key in KAYA_RELAY_TOKEN KAYA_WAHA_API_KEY KAYA_WHATSAPP_WEBHOOK_TOKEN \
                 KAYA_WEB_USER KAYA_WEB_PASS CLOUDFLARE_TUNNEL_TOKEN; do
        echo "$key=$(value "$key")"
      done
    } | ssh "$PI" "umask 077 && cat > $REMOTE_DIR/deploy/pi/.env"
    echo "🔐 wrote $REMOTE_DIR/deploy/pi/.env (COMPOSE_PROFILES empty: gateway only)"
  fi
fi

echo "📋 Copying the runtime whitelist ..."
ssh "$PI" "mkdir -p $REMOTE_DIR/deploy/pi/data/runtime $REMOTE_DIR/deploy/pi/data/gateway $REMOTE_DIR/deploy/pi/data/waha"
if [[ -f "$PROD_DIR/data/whatsapp_whitelist.json" ]]; then
  scp -q "$PROD_DIR/data/whatsapp_whitelist.json" "$PI:$REMOTE_DIR/deploy/pi/data/runtime/whatsapp_whitelist.json"
fi

# The Wake-on-LAN timer and the LAN firewall are the lab's since 2026-10-22: lab-infra pi/install.sh.
ssh "$PI" "test -f lab-infra/shared/power.yaml" \
  || echo "⚠️  lab-infra/shared/power.yaml missing on the Pi: run lab-infra pi/install.sh (offline reply falls back)"

ssh "$PI" "test -f $REMOTE_DIR/deploy/pi/.env" \
  || { echo "❌ $REMOTE_DIR/deploy/pi/.env missing: run with --init-env" >&2; exit 1; }

echo "🔨 Building and (re)starting on the Pi ..."
ssh "$PI" "cd $REMOTE_DIR/deploy/pi && docker compose up -d --build --remove-orphans && docker image prune -f >/dev/null"
ssh "$PI" "cd $REMOTE_DIR/deploy/pi && docker compose ps --format '{{.Name}}  {{.Status}}'"
echo "✅ Pi deployed. Status: curl -s http://${PI_LAN_IP:-192.168.1.238}:8088/status"
