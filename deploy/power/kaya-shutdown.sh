#!/usr/bin/env bash
# Scheduled shutdown of the GPU PC. Run by kaya-shutdown.timer (as root) at the
# times in config.yaml `power.shutdown`. See README.md in this directory.
#
#   kaya-shutdown.sh [--dry-run]
#
# While anything says the PC is in use, the shutdown is POSTPONED: checked again
# every POWER_RETRY_MINUTES until the PC is idle, then carried out. Retries stop
# POWER_STOP_BEFORE_WAKE_MINUTES before the next wake, so a PC busy all night
# stays on rather than going down in the morning. Transient work (a WhatsApp
# reply being generated) is waited out.
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
RETRY_MINUTES="${POWER_RETRY_MINUTES:-15}"
STOP_BEFORE_WAKE_MINUTES="${POWER_STOP_BEFORE_WAKE_MINUTES:-60}"

say() { echo "[kaya-shutdown] $*"; }
hhmm() { date -d "@$1" +%H:%M; }

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

# Returns once the PC is idle. Exits when it is still busy at the last retry, or
# at once when there is no deadline (the next wake time could not be read).
wait_until_idle() {
  local reason next
  while reason="$(busy_reason)"; [ -n "$reason" ]; do
    next=$(( $(date +%s) + RETRY_MINUTES * 60 ))
    if [ -z "$deadline" ]; then
      say "skipping tonight's shutdown: $reason"; exit 0
    fi
    if [ "$next" -gt "$deadline" ]; then
      say "giving up tonight: $reason (retries stop at $(hhmm "$deadline"))"; exit 0
    fi
    say "postponing: $reason; next check at $(hhmm "$next")"
    sleep $((RETRY_MINUTES * 60))
  done
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

# The kernel GRUB boots next (the newest installed) must have an nvidia module,
# or tomorrow's boot comes up without a GPU driver and the bot is gone. A kernel
# installed without its module is what took the bot down on 2026-09-25.
next_kernel="$(ls -1 /boot/vmlinuz-* 2>/dev/null | sed 's#/boot/vmlinuz-##' | sort -V | tail -1)"
if [ -n "$next_kernel" ] && ! find "/lib/modules/$next_kernel" -name 'nvidia.ko*' 2>/dev/null | grep -q .; then
  say "skipping tonight's shutdown: kernel $next_kernel has no nvidia module (sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04)"
  as_user notify-send -u critical "Kaya: o PC não se desligou" \
    "O kernel $next_kernel não tem o driver NVIDIA. Instala linux-modules-nvidia-595-open-generic-hwe-24.04 antes de reiniciar." \
    2>/dev/null || true
  exit 0
fi

# The wake this shutdown precedes: it sets the retry deadline and the RTC alarm.
wake_epoch="$(cd "$SCHEDULE_REPO" && "$SCHEDULE_PY" -m src.gateway.schedule --config config.yaml next-wake-epoch 2>/dev/null)"
deadline=""
if [[ "$wake_epoch" =~ ^[0-9]+$ ]]; then
  deadline=$((wake_epoch - STOP_BEFORE_WAKE_MINUTES * 60))
else
  wake_epoch=""
  say "could not read the next wake time: one check, no retries"
fi

while true; do
  wait_until_idle
  if [ "$DRY_RUN" = 1 ]; then
    say "dry run: the PC is idle and would shut down now"
    exit 0
  fi
  as_user notify-send -u critical "O PC vai desligar-se" \
    "Desliga-se dentro de $((WARN_SECONDS / 60)) minutos (horário). Mexe no rato para adiar." \
    2>/dev/null || true
  sleep "$WARN_SECONDS"
  reason="$(busy_reason)"
  [ -z "$reason" ] && break
  say "postponed during the warning: $reason"
done

wait_for_replies

if [ -n "$GATEWAY_URL" ]; then
  curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" "$GATEWAY_URL/pc/going-down" >/dev/null 2>&1 \
    && say "told the gateway we are going down" \
    || say "could not reach the gateway (it will notice within 90 s)"
fi

# Backup wake: the RTC alarm, in case the Pi's Wake-on-LAN packet is missed.
if [ -n "$wake_epoch" ]; then
  rtcwake -m no -t "$wake_epoch" >/dev/null 2>&1 \
    && say "RTC alarm set for $(date -d "@$wake_epoch")" \
    || say "could not set the RTC alarm"
fi

say "powering off"
systemctl poweroff
