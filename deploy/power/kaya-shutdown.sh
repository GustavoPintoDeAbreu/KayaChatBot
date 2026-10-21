#!/usr/bin/env bash
# Shutdown of the GPU PC. The scheduled mode runs from kaya-shutdown.timer (as
# root) at the times in config.yaml `power.shutdown`; the manual mode runs from
# kaya-shutdown-manual.service, started by the PC's power listener when the
# owner confirms a /homelaboff. See README.md in this directory.
#
#   kaya-shutdown.sh [--dry-run | --manual | --report | --after-early-wake]
#
# Scheduled mode: while anything says the PC is in use, the shutdown is
# POSTPONED: checked again every POWER_RETRY_MINUTES until the PC is idle, then
# carried out. Retries stop POWER_STOP_BEFORE_WAKE_MINUTES before the next wake,
# so a PC busy all night stays on rather than going down in the morning.
#
# Manual mode: no deadline and no retries-stop — it waits until the PC is idle
# and reports the wait to the owner's WhatsApp every POWER_MANUAL_UPDATE_MINUTES
# (capped at POWER_MANUAL_MAX_WAIT_MINUTES). `.stay-on` does not block it: he
# asked for it. GPU use while Kaya is finishing a reply counts as hers. A
# `cancel` file (written by the power listener) stops the wait.
#
# --after-early-wake: ten minutes after a boot that was an early wake for the
# idea pipeline (its ~/.idea-wake-at, honoured by the previous night's shutdown),
# run the same postponing loop: the PC powers off again once idle, and honours
# a new request when it does. The day has started by then and the PC stays on.
#
# --report prints what would hold the PC up, one tab-separated line per reason,
# and exits: the power listener serves it to the gateway.
#
# The shutdown is a plain `systemctl poweroff`. Never `docker stop` first: an
# explicitly stopped container stays stopped after the reboot, which is how the
# live stack would silently fail to come back at 07:00.
set -uo pipefail

DRY_RUN=0; MODE=scheduled
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --manual) MODE=manual ;;
  --report) MODE=report ;;
  --after-early-wake) MODE=early ;;
esac
ENV_FILE="${KAYA_POWER_ENV:-/etc/kaya-power.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
USER_NAME="${POWER_USER:-gustavo}"
USER_UID="$(id -u "$USER_NAME")"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
IDLE_MINUTES="${POWER_IDLE_MINUTES:-30}"
CLAUDE_IDLE_MINUTES="${POWER_CLAUDE_IDLE_MINUTES:-30}"
CLAUDE_PROJECTS_DIR="${POWER_CLAUDE_PROJECTS_DIR:-$USER_HOME/.claude/projects}"
WARN_SECONDS="${POWER_WARN_SECONDS:-120}"
GPU_BUSY_PERCENT="${POWER_GPU_BUSY_PERCENT:-30}"
# The dev card (GPU0), by UUID because indices change across boots. GPU1 is Kaya's.
DEV_GPU_UUID="${POWER_DEV_GPU_UUID:-GPU-ab32b3d2-3bab-2b24-9749-1caa6400f82d}"
PC_APP_URL="${POWER_PC_APP_URL:-http://127.0.0.1:7860}"
GATEWAY_URL="${POWER_GATEWAY_URL:-}"
RELAY_TOKEN="${KAYA_RELAY_TOKEN:-}"
SCHEDULE_PY="${POWER_PYTHON:-python3}"
SCHEDULE_REPO="${POWER_REPO:-$USER_HOME/kaya-prod}"
RETRY_MINUTES="${POWER_RETRY_MINUTES:-10}"
STOP_BEFORE_WAKE_MINUTES="${POWER_STOP_BEFORE_WAKE_MINUTES:-60}"
STAY_ON_FILE="${POWER_STAY_ON_FILE:-$USER_HOME/.stay-on}"
# A .stay-on untouched for this long no longer holds the PC: `touch ~/.stay-on` keeps it
# up for one night, and the idea pipeline refreshes its own file while it holds it. An
# orphaned file kept the PC on two nights (2026-10-20/21). Never deleted from here.
STAY_ON_MAX_HOURS="${POWER_STAY_ON_MAX_HOURS:-20}"
DOCKER="${DOCKER_BIN:-/snap/bin/docker}"
GPU_RECHECK_SECONDS="${POWER_GPU_RECHECK_SECONDS:-5}"
BOOT_DIR="${POWER_BOOT_DIR:-/boot}"
MODULES_DIR="${POWER_MODULES_DIR:-/lib/modules}"
STATE_DIR="${POWER_STATE_DIR:-/run/kaya-power}"
MANUAL_CHECK_SECONDS="${POWER_MANUAL_CHECK_SECONDS:-60}"
MANUAL_UPDATE_MINUTES="${POWER_MANUAL_UPDATE_MINUTES:-30}"
MANUAL_MAX_WAIT_MINUTES="${POWER_MANUAL_MAX_WAIT_MINUTES:-720}"
# The idea pipeline's wake request (epoch seconds on the first line) and the
# marker of the early wake this boot followed, written at poweroff.
WAKE_REQUEST_FILE="${POWER_WAKE_REQUEST_FILE:-$USER_HOME/.idea-wake-at}"
EARLY_WAKE_FILE="${POWER_EARLY_WAKE_FILE:-/var/lib/kaya-power/early-wake}"

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

# The highest GPU utilization (an integer), or nothing. With a UUID, only that GPU's.
gpu_util() {
  nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v only="${1:-}" '(only == "" || $1 == only) && $2 ~ /^[0-9]+$/ { print $2 }' \
    | sort -n | tail -1
}

# The replies the bot has accepted and is still generating. The integer, or
# nothing when there is no token or the app is unreachable.
pending_replies() {
  [ -n "$RELAY_TOKEN" ] || return 0
  curl -fsS --max-time 5 -H "X-Relay-Token: $RELAY_TOKEN" "$PC_APP_URL/whatsapp/relay/status" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("pending_replies",0))' 2>/dev/null
}

# Minutes since a Claude Code session last wrote its transcript, or nothing when
# none did within CLAUDE_IDLE_MINUTES. A Remote Control session driven from the
# phone touches no keyboard, so the idle monitor cannot see it.
claude_active_minutes() {
  local newest
  newest="$(find "$CLAUDE_PROJECTS_DIR" -name '*.jsonl' -mmin "-$CLAUDE_IDLE_MINUTES" \
              -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
  [ -n "$newest" ] && echo $(( ($(date +%s) - ${newest%.*}) / 60 ))
}

# Whether .stay-on exists and was touched within STAY_ON_MAX_HOURS.
stay_on_fresh() {
  [ -n "$(find "$STAY_ON_FILE" -maxdepth 0 -mmin "-$((STAY_ON_MAX_HOURS * 60))" 2>/dev/null)" ]
}

stay_on_since() { date -r "$STAY_ON_FILE" '+%Y-%m-%d %H:%M' 2>/dev/null; }

# Echoes every reason the PC counts as in use, one per line. Scheduled mode
# blocks on .stay-on and on Claude Code sessions; manual mode does not — the
# owner asked for it.
busy_reasons() {
  local mode="${1:-scheduled}"
  if [ "$mode" = scheduled ] && [ -e "$STAY_ON_FILE" ]; then
    if stay_on_fresh; then
      echo "$STAY_ON_FILE exists"
    else
      say "ignoring stale $STAY_ON_FILE from $(stay_on_since) (older than ${STAY_ON_MAX_HOURS} h)" >&2
    fi
  fi
  if [ "$mode" = scheduled ]; then
    local claude; claude="$(claude_active_minutes)"
    [ -n "$claude" ] && echo "a Claude Code session was active $claude min ago"
  fi
  local idle; idle="$(idle_ms)"
  if [ -n "$idle" ] && [ "$idle" -lt $((IDLE_MINUTES * 60000)) ]; then
    echo "keyboard/mouse used $((idle / 60000)) min ago"
  fi
  pgrep -f 'Runner.Worker' >/dev/null && echo "a CI job is running"
  pgrep -f 'qwen-code/bin/qwen-(impl|bench)' >/dev/null && echo "a Qwen implementation run is active"
  local jobs
  jobs="$("$DOCKER" ps --format '{{.Names}}' 2>/dev/null | grep -E '^(kaya-train|kaya-data|kaya-infer|kaya-llama-bench|kaya-sim)$' | tr '\n' ' ')"
  [ -n "$jobs" ] && echo "job containers running: $jobs"
  if [ -n "$("$DOCKER" ps -q --filter label=idea-pipeline.idea 2>/dev/null | head -1)" ]; then
    echo "an idea-pipeline build or test container is running"
  fi
  # Manual mode: while Kaya is finishing replies, GPU load counts as hers, except on the dev
  # card, where a bench or a qcode session is never hers.
  local only=""
  if [ "$mode" = manual ]; then
    local pending; pending="$(pending_replies)"
    if [ -n "$pending" ] && [ "$pending" -gt 0 ]; then
      only="$DEV_GPU_UUID"
    fi
  fi
  local util
  util="$(gpu_util "$only")"
  if [ -n "$util" ] && [ "$util" -ge "$GPU_BUSY_PERCENT" ]; then
    sleep "$GPU_RECHECK_SECONDS"
    util="$(gpu_util "$only")"
    [ -n "$util" ] && [ "$util" -ge "$GPU_BUSY_PERCENT" ] && echo "a GPU is ${util}% busy"
  fi
}

# The first reason only: the scheduled loop's interface.
# Collected first: piping into head -n1 made the rest of the echoes hit a closed pipe.
busy_reason() { local all; all="$(busy_reasons scheduled)"; printf '%s\n' "${all%%$'\n'*}"; }

# The newest installed kernel that has no nvidia module, or nothing.
kernel_problem() {
  local kernel
  for f in "$BOOT_DIR"/vmlinuz-*; do [ -e "$f" ] && echo "${f##*/vmlinuz-}"; done | sort -V | tail -1 | {
    read -r kernel
    [ -n "$kernel" ] || return 0
    find "$MODULES_DIR/$kernel" -name 'nvidia.ko*' 2>/dev/null | grep -q . || echo "$kernel"
  }
}

# The schedule CLI. POWER_NOW_EPOCH pins its clock (tests only).
schedule_cli() {
  (cd "$SCHEDULE_REPO" && "$SCHEDULE_PY" -m src.gateway.schedule --config config.yaml \
    ${POWER_NOW_EPOCH:+--now "$POWER_NOW_EPOCH"} "$@" 2>/dev/null)
}

# The digits of the next wake time, or nothing.
next_wake_epoch() {
  local epoch
  epoch="$(schedule_cli next-wake-epoch)"
  [[ "$epoch" =~ ^[0-9]+$ ]] && echo "$epoch"
}

# The digits of the early wake the idea pipeline asked for, when the schedule
# honours it, or nothing (no request, or the request is ignored).
early_wake_epoch() {
  [ -f "$WAKE_REQUEST_FILE" ] || return 0
  local epoch
  epoch="$(schedule_cli --wake-request "$WAKE_REQUEST_FILE" early-wake-epoch)"
  [[ "$epoch" =~ ^[0-9]+$ ]] && echo "$epoch"
}

# One WhatsApp update to the owner, through the gateway. FINAL=1 marks the last.
notify_owner() {
  local text="$1" final="${2:-0}"
  say "update: $text"
  if [ -n "$GATEWAY_URL" ] && [ -n "$RELAY_TOKEN" ]; then
    local body
    body="$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "final": sys.argv[2] == "1"}))' "$text" "$final")"
    curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" \
      -H 'Content-Type: application/json' --data "$body" "$GATEWAY_URL/pc/power/update" >/dev/null 2>&1 \
      || say "could not send the WhatsApp update"
  fi
}

# The last steps, shared by both modes: Kaya's replies, the going-down notice,
# the RTC alarm, the poweroff. FINAL_TEXT is one last WhatsApp update.
power_off_now() {
  local waited=0 pending
  local alarm="$wake_epoch" early="" body=()
  # Only the scheduled and early modes honour the request: a manual shutdown is
  # the owner asking for the PC off now, and it clears a stale marker.
  if [ "$MODE" = scheduled ] || [ "$MODE" = early ]; then
    early="$(early_wake_epoch)"
  fi
  if [ -n "$early" ]; then
    alarm="$early"
    mkdir -p "$(dirname "$EARLY_WAKE_FILE")" && echo "$early" > "$EARLY_WAKE_FILE"
    body=(-H 'Content-Type: application/json' --data "{\"wake_at\": $early}")
    say "the idea pipeline asked to wake at $(hhmm "$early"): waking then, not at the scheduled time"
  else
    rm -f "$EARLY_WAKE_FILE"
  fi
  while [ "$waited" -lt 600 ]; do
    pending="$(pending_replies)"
    [ -z "$pending" ] || [ "$pending" = 0 ] && break
    say "waiting for WhatsApp replies in flight ..."
    sleep 15; waited=$((waited + 15))
  done
  if [ -n "$GATEWAY_URL" ]; then
    curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" "${body[@]}" "$GATEWAY_URL/pc/going-down" >/dev/null 2>&1 \
      && say "told the gateway we are going down" \
      || say "could not reach the gateway (it will notice within 90 s)"
  fi
  if [ -n "$alarm" ]; then
    rtcwake -m no -t "$alarm" >/dev/null 2>&1 \
      && say "RTC alarm set for $(date -d "@$alarm")" \
      || say "could not set the RTC alarm"
  fi
  [ -n "${1:-}" ] && notify_owner "$1" 1
  say "powering off"
  systemctl poweroff
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

cancel_requested() { [ -e "$STATE_DIR/cancel" ]; }

sleep_unless_cancelled() {
  local n="$1" i
  for ((i = 0; i < n; i++)); do
    cancel_requested && return 1
    sleep 1
  done
  return 0
}

# Switch to "stopping" under the same lock the power listener's cancel takes,
# so a cancel is either honoured (return 1) or reported as too late.
enter_stopping() {
  exec 9>"$STATE_DIR/lock"; flock 9
  if cancel_requested; then flock -u 9; return 1; fi
  echo stopping > "$STATE_DIR/phase"; flock -u 9
}

report() {
  local problem
  problem="$(kernel_problem)"
  if [ -n "$problem" ]; then
    printf 'note\tkernel %s has no nvidia module: a shutdown would be refused\n' "$problem"
  fi
  if [ -e "$STAY_ON_FILE" ]; then
    if stay_on_fresh; then
      printf 'note\t%s exists (ignored when you ask)\n' "$STAY_ON_FILE"
    else
      printf 'note\t%s is stale (from %s, older than %s h): it holds nothing\n' \
        "$STAY_ON_FILE" "$(stay_on_since)" "$STAY_ON_MAX_HOURS"
    fi
  fi
  local pending; pending="$(pending_replies)"
  if [ -n "$pending" ] && [ "$pending" -gt 0 ]; then
    local word="replies"; [ "$pending" = 1 ] && word="reply"
    printf 'note\tKaya is finishing %s %s\n' "$pending" "$word"
  fi
  busy_reasons manual | while IFS= read -r reason; do
    [ -n "$reason" ] && printf 'busy\t%s\n' "$reason"
  done
  exit 0
}

manual_main() {
  mkdir -p "$STATE_DIR"
  trap 'rm -f "$STATE_DIR/phase" "$STATE_DIR/cancel"' EXIT
  echo waiting > "$STATE_DIR/phase"
  local problem
  problem="$(kernel_problem)"
  if [ -n "$problem" ]; then
    notify_owner "Not shutting down: kernel $problem has no nvidia module, so the next boot would have no GPU driver. Install linux-modules-nvidia-595-open-generic-hwe-24.04 at the PC first." 1
    return 0
  fi
  wake_epoch="$(next_wake_epoch)"
  local started last_sent last_key reasons listed now key
  started="$(date +%s)"; last_sent=0; last_key=""
  while true; do
    if cancel_requested; then
      notify_owner "Cancelled. The PC stays on." 1
      return 0
    fi
    reasons="$(busy_reasons manual)"
    if [ -z "$reasons" ]; then
      as_user notify-send -u critical "O PC vai desligar-se" \
        "Pedido por WhatsApp. Desliga-se dentro de $((WARN_SECONDS / 60)) minutos. Mexe no rato para adiar." \
        2>/dev/null || true
      sleep_unless_cancelled "$WARN_SECONDS" || continue
      reasons="$(busy_reasons manual)"
      [ -z "$reasons" ] && break
    fi
    listed="$(printf '%s\n' "$reasons" | sed 's/^/- /')"
    now="$(date +%s)"
    if [ $((now - started)) -ge $((MANUAL_MAX_WAIT_MINUTES * 60)) ]; then
      notify_owner "Gave up after $MANUAL_MAX_WAIT_MINUTES min. Still running:"$'\n'"$listed"$'\n'"The PC stays on." 1
      return 0
    fi
    key="$(printf '%s' "$reasons" | tr -d '0-9')"
    if [ "$key" != "$last_key" ] || [ $((now - last_sent)) -ge $((MANUAL_UPDATE_MINUTES * 60)) ]; then
      notify_owner "Waiting for:"$'\n'"$listed" 0
      last_key="$key"; last_sent="$now"
    fi
    sleep_unless_cancelled "$MANUAL_CHECK_SECONDS"
  done
  if ! enter_stopping; then
    notify_owner "Cancelled. The PC stays on." 1
    return 0
  fi
  notify_owner "Only Kaya is left. Finishing her replies, then powering off." 0
  power_off_now "Powering off now. /homelabon wakes it."
}

# Ten minutes after an early wake: consume the marker, and run the same
# postponing loop below — the PC powers off again once idle, and honours a new
# request when it does.
early_main() {
  local requested
  requested="$(head -n1 "$EARLY_WAKE_FILE" 2>/dev/null)"
  rm -f "$EARLY_WAKE_FILE"
  if ! [[ "$requested" =~ ^[0-9]+$ ]]; then
    say "not an early wake: nothing to do"; exit 0
  fi
  if [ "$(schedule_cli is-on)" != false ]; then
    say "the day has started: the PC stays on"; exit 0
  fi
  say "woke early for the idea pipeline (asked for $(hhmm "$requested")): powering off again once idle"
}

[ "$MODE" = report ] && { report; exit 0; }
[ "$MODE" = manual ] && { manual_main; exit 0; }
[ "$MODE" = early ] && early_main

# The kernel GRUB boots next (the newest installed) must have an nvidia module,
# or tomorrow's boot comes up without a GPU driver and the bot is gone. A kernel
# installed without its module is what took the bot down on 2026-09-25.
next_kernel="$(kernel_problem)"
if [ -n "$next_kernel" ]; then
  say "skipping tonight's shutdown: kernel $next_kernel has no nvidia module (sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04)"
  as_user notify-send -u critical "Kaya: o PC não se desligou" \
    "O kernel $next_kernel não tem o driver NVIDIA. Instala linux-modules-nvidia-595-open-generic-hwe-24.04 antes de reiniciar." \
    2>/dev/null || true
  exit 0
fi

# The wake this shutdown precedes: it sets the retry deadline and the RTC alarm.
wake_epoch="$(next_wake_epoch)"
deadline=""
if [ -n "$wake_epoch" ]; then
  deadline=$((wake_epoch - STOP_BEFORE_WAKE_MINUTES * 60))
else
  say "could not read the next wake time: one check, no retries"
fi

while true; do
  wait_until_idle
  if [ "$DRY_RUN" = 1 ]; then
    say "dry run: the PC is idle and would shut down now"
    early="$(early_wake_epoch)"
    [ -n "$early" ] && say "dry run: would wake at $(hhmm "$early") for the idea pipeline, not at the scheduled time"
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

power_off_now
