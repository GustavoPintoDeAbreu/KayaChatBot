#!/usr/bin/env bash
# Run at every poweroff of the GPU PC, however it was started (the desktop menu,
# the power button, `systemctl poweroff`). Run by kaya-going-down.service as its
# ExecStop, so the network and the containers are still up. See README.md here.
#
#   kaya-going-down.sh [--dry-run]
#
# 1. Waits (bounded) for WhatsApp replies already accepted by the bot.
# 2. Tells the Pi gateway the PC is going down, so its offline reply says "volto
#    às 07:00" rather than "estou em baixo".
# 3. Sets the RTC alarm for the next wake, as a backup to the Pi's Wake-on-LAN.
#
# A reboot is skipped: the PC is back within a minute. Never `docker stop` here:
# an explicitly stopped container stays stopped after the next boot.
set -uo pipefail

DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1
ENV_FILE="${KAYA_POWER_ENV:-/etc/kaya-power.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
USER_HOME="$(getent passwd "${POWER_USER:-gustavo}" | cut -d: -f6)"
PC_APP_URL="${POWER_PC_APP_URL:-http://127.0.0.1:7860}"
GATEWAY_URL="${POWER_GATEWAY_URL:-}"
RELAY_TOKEN="${KAYA_RELAY_TOKEN:-}"
SCHEDULE_PY="${POWER_PYTHON:-python3}"
SCHEDULE_REPO="${POWER_REPO:-$USER_HOME/kaya-prod}"
REPLY_WAIT_SECONDS="${POWER_REPLY_WAIT_SECONDS:-60}"

say() { echo "[kaya-going-down] $*"; }

if [ "$DRY_RUN" = 0 ] && systemctl list-jobs --no-legend 2>/dev/null | grep -q 'reboot\.target'; then
  say "rebooting: nothing to announce"; exit 0
fi

wait_for_replies() {
  [ -n "$RELAY_TOKEN" ] || return 0
  local waited=0 status
  while [ "$waited" -lt "$REPLY_WAIT_SECONDS" ]; do
    status="$(curl -fsS --max-time 5 -H "X-Relay-Token: $RELAY_TOKEN" "$PC_APP_URL/whatsapp/relay/status" 2>/dev/null)" || return 0
    echo "$status" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("pending_replies",0)==0 else 1)' && return 0
    say "waiting for WhatsApp replies in flight ..."
    sleep 5; waited=$((waited + 5))
  done
  say "gave up waiting for replies after ${REPLY_WAIT_SECONDS} s"
}

wake_epoch="$(cd "$SCHEDULE_REPO" && "$SCHEDULE_PY" -m src.gateway.schedule --config config.yaml next-wake-epoch 2>/dev/null)"
[[ "$wake_epoch" =~ ^[0-9]+$ ]] || wake_epoch=""

if [ "$DRY_RUN" = 1 ]; then
  say "dry run: would wait up to ${REPLY_WAIT_SECONDS} s for replies, tell ${GATEWAY_URL:-no gateway} we are going down, and set the RTC alarm for ${wake_epoch:+$(date -d "@$wake_epoch")}"
  exit 0
fi

wait_for_replies

if [ -n "$GATEWAY_URL" ]; then
  curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" "$GATEWAY_URL/pc/going-down" >/dev/null 2>&1 \
    && say "told the gateway we are going down" \
    || say "could not reach the gateway (it will notice within 90 s)"
fi

if [ -n "$wake_epoch" ]; then
  rtcwake -m no -t "$wake_epoch" >/dev/null 2>&1 \
    && say "RTC alarm set for $(date -d "@$wake_epoch")" \
    || say "could not set the RTC alarm"
fi
