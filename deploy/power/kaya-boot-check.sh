#!/usr/bin/env bash
# After every boot: can this PC serve Kaya? Run by kaya-boot-check.service (as
# root) once Docker is up. See README.md in this directory.
#
# 1. The NVIDIA driver must be loaded. A kernel updated without its matching
#    nvidia module boots with no driver; Docker then cannot give the GPU
#    containers their devices, gives up on them, and the bot is silently gone.
#    That happened on 2026-09-25. If the driver is missing, tell the Pi gateway,
#    which then sends the "estou em baixo" line instead of staying silent.
# 2. With the driver there, start any restart-policy container Docker gave up
#    on at boot (a start error recorded, not a container someone stopped).
#
# It never loads kernel modules itself: loading the driver under a desktop that
# booted without it froze the machine on 2026-09-25.
set -uo pipefail
ENV_FILE="${KAYA_POWER_ENV:-/etc/kaya-power.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
GATEWAY_URL="${POWER_GATEWAY_URL:-}"
RELAY_TOKEN="${KAYA_RELAY_TOKEN:-}"
DOCKER="${DOCKER_BIN:-/snap/bin/docker}"

say() { echo "[kaya-boot-check] $*"; }

for attempt in $(seq 1 24); do
  nvidia-smi -L >/dev/null 2>&1 && break
  sleep 5
done

if ! nvidia-smi -L >/dev/null 2>&1; then
  if find "/lib/modules/$(uname -r)" -name 'nvidia.ko*' 2>/dev/null | grep -q .; then
    say "ERROR: the nvidia module exists for $(uname -r) but is not loaded; reboot, then check 'journalctl -k | grep -i nvidia'"
  else
    say "ERROR: no nvidia module for kernel $(uname -r). Fix: sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04, then reboot"
  fi
  if [ -n "$GATEWAY_URL" ] && [ -n "$RELAY_TOKEN" ]; then
    curl -fsS --max-time 5 -X POST -H "X-Relay-Token: $RELAY_TOKEN" "$GATEWAY_URL/pc/degraded" >/dev/null 2>&1 \
      && say "told the Pi gateway the PC is degraded" \
      || say "could not reach the Pi gateway"
  fi
  exit 1
fi
say "driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"

for container in $("$DOCKER" ps -a --filter status=exited --format '{{.Names}}'); do
  policy="$("$DOCKER" inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$container")"
  error="$("$DOCKER" inspect -f '{{.State.Error}}' "$container")"
  if [ "$policy" = unless-stopped ] || [ "$policy" = always ]; then
    if [ -n "$error" ]; then
      say "starting $container, which failed at boot: $error"
      "$DOCKER" start "$container" >/dev/null && say "  $container started" || say "  $container failed again"
    fi
  fi
done
