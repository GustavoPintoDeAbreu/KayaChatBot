#!/usr/bin/env bash
# Scheduled shutdown of the GPU PC. Run by kaya-shutdown.timer (as root) at the
# times in config.yaml `power.shutdown`. See README.md in this directory.
#
#   kaya-shutdown.sh [--dry-run]
#
# The night is SKIPPED, not postponed, when anything says the PC is in use:
# that was the decision, because a PC someone is using at 23:00 is usually still
# in use at 23:30. Transient work (a WhatsApp reply being generated) is waited out.
#
# The shutdown is a plain `systemctl poweroff`. Never `docker stop` first: an
# explicitly stopped container stays stopped after the reboot, which is how the
# live stack would silently fail to come back at 07:00.
set -uo pipefail

DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1
ENV_FILE="${KAYA_POWER_ENV:-/etc/kaya-power.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
USER_NAME="${POWER_USER:-gustavo}"
USER_UID="$(id -u "$USER_NAME")"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
IDLE_MINUTES="${POWER_IDLE_MINUTES:-15}"
WARN_SECONDS="${POWER_WARN_SECONDS:-120}"
GPU_BUSY_PERCENT="${POWER_GPU_BUSY_PERCENT:-30}"
PC_APP_URL="${POWER_PC_APP_URL:-http://127.0.0.1:7860}"
GATEWAY_URL="${POWER_GATEWAY_URL:-}"
RELAY_TOKEN="${KAYA_RELAY_TOKEN:-}"
SCHEDULE_PY="${POWER_PYTHON:-python3}"
SCHEDULE_REPO="${POWER_REPO:-$USER_HOME/kaya-prod}"

say() { echo "[kaya-shutdown] $*"; }

as_user() {
  sudo -u "$USER_NAME" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$USER_UID/bus" \
    DISPLAY=:0 "$@"
}

idle_ms() {
  as_user gdbus call --session --dest org.gnome.Mutter.IdleMonitor \
    --object-path /org/gnome/Mutter/IdleMonitor/Core \
    --method org.gnome.Mutter.IdleMonitor.GetIdletime 2>/dev/null \
    | sed -n 's/^(uint64 \([0-9]*\),)$/\1/p'
}

# Echoes the first reason the PC counts as in use; nothing when it is idle.
busy_reason() {
  [ -e "$USER_HOME/.stay-on" ] && { echo "$USER_HOME/.stay-on exists"; return; }
  local idle; idle="$(idle_ms)"
  if [ -n "$idle" ] && [ "$idle" -lt $((IDLE_MINUTES * 60000)) ]; then
    echo "keyboard/mouse used $((idle / 60000)) min ago"; return
  fi
  pgrep -f 'Runner.Worker' >/dev/null && { echo "a CI job is running"; return; }
  pgrep -f 'qwen-code/bin/qwen-(impl|bench)' >/dev/null && { echo "a Qwen implementation run is active"; return; }
  local jobs
  jobs="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(kaya-train|kaya-data|kaya-infer|kaya-llama-bench|kaya-sim)$' | tr '\n' ' ')"
  [ -n "$jobs" ] && { echo "job containers running: $jobs"; return; }
  local util
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
  if [ -n "$util" ] && [ "$util" -ge "$GPU_BUSY_PERCENT" ]; then
    sleep 5
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [ -n "$util" ] && [ "$util" -ge "$GPU_BUSY_PERCENT" ] && { echo "a GPU is ${util}% busy"; return; }
  fi
}

# Replies already accepted by the bot are finished rather than dropped.
wait_for_replies() {
  [ -n "$RELAY_TOKEN" ] || return 0
  local waited=0 status
  while [ "$waited" -lt 600 ]; do
    status="$(curl -fsS --max-time 5 -H "X-Relay-Token: $RELAY_TOKEN" "$PC_APP_URL/whatsapp/relay/status" 2>/dev/null)" || return 0
    echo "$status" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("pending_replies",0)==0 else 1)' && return 0
    say "waiting for WhatsApp replies in flight ..."
    sleep 15; waited=$((waited + 15))
  done
}

reason="$(busy_reason)"
if [ -n "$reason" ]; then
  say "skipping tonight's shutdown: $reason"
  exit 0
fi

if [ "$DRY_RUN" = 1 ]; then
  say "dry run: the PC is idle and would shut down now"
  exit 0
fi

as_user notify-send -u critical "O PC vai desligar-se" \
  "Desliga-se dentro de $((WARN_SECONDS / 60)) minutos (horário). Mexe no rato para cancelar esta noite." \
  2>/dev/null || true
sleep "$WARN_SECONDS"
reason="$(busy_reason)"
if [ -n "$reason" ]; then
  say "cancelled during the warning: $reason"
  exit 0
fi

wait_for_replies

if [ -n "$GATEWAY_URL" ]; then
  curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" "$GATEWAY_URL/pc/going-down" >/dev/null 2>&1 \
    && say "told the gateway we are going down" \
    || say "could not reach the gateway (it will notice within 90 s)"
fi

# Backup wake: the RTC alarm, in case the Pi's Wake-on-LAN packet is missed.
wake_epoch="$(cd "$SCHEDULE_REPO" && "$SCHEDULE_PY" -m src.gateway.schedule --config config.yaml next-wake-epoch 2>/dev/null)"
if [ -n "$wake_epoch" ]; then
  rtcwake -m no -t "$wake_epoch" >/dev/null 2>&1 \
    && say "RTC alarm set for $(date -d "@$wake_epoch")" \
    || say "could not set the RTC alarm"
fi

say "powering off"
systemctl poweroff
