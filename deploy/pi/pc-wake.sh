#!/usr/bin/env bash
# Wake the GPU PC with a magic packet and confirm it came back. Run on the Pi by
# pc-wake.timer at config.yaml power.wake_time minus wol_lead_minutes, and by
# hand as `pi5 wake-pc` from the PC's side of things.
#
# Retries once a minute for ten minutes: a single UDP packet is cheap to lose,
# and the PC's RTC alarm is the only other thing that would wake it.
set -uo pipefail
ENV_FILE="${PC_WAKE_ENV:-/home/gustavo/kaya-gateway/deploy/pi/.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
MAC="${PC_MAC:-b4:2e:99:92:4f:be}"
BROADCAST="${PC_BROADCAST:-192.168.1.255}"
PC_HEALTH="${KAYA_PC_URL:-http://192.168.1.149:7860}/whatsapp/health"
PC_HOST="$(echo "$PC_HEALTH" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')"

for attempt in $(seq 1 10); do
  if ping -c1 -W2 "$PC_HOST" >/dev/null 2>&1; then
    echo "[pc-wake] PC is up (attempt $attempt)"
    exit 0
  fi
  wakeonlan -i "$BROADCAST" "$MAC" >/dev/null
  echo "[pc-wake] magic packet sent to $MAC via $BROADCAST (attempt $attempt)"
  sleep 60
done
echo "[pc-wake] PC did not answer after 10 minutes; check the BIOS WoL setting" >&2
exit 1
