#!/usr/bin/env bash
# Install the scheduled shutdown on the GPU PC. Run with sudo from the repo root:
#   sudo deploy/power/install.sh [--uninstall]
# The timer's OnCalendar lines are generated from config.yaml `power:` so the
# PC, the Pi's Wake-on-LAN timer and the gateway's offline reply cannot disagree.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${POWER_PYTHON:-python3}"

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now kaya-shutdown.timer 2>/dev/null || true
  rm -f /etc/systemd/system/kaya-shutdown.{timer,service} /usr/local/sbin/kaya-shutdown.sh
  systemctl daemon-reload
  echo "removed"; exit 0
fi

install -m 0755 "$REPO/deploy/power/kaya-shutdown.sh" /usr/local/sbin/kaya-shutdown.sh
install -m 0644 "$REPO/deploy/power/kaya-shutdown.service" /etc/systemd/system/kaya-shutdown.service

{
  echo "[Unit]"
  echo "Description=Scheduled shutdown times (generated from config.yaml power.shutdown)"
  echo
  echo "[Timer]"
  (cd "$REPO" && "$PY" -m src.gateway.schedule --config config.yaml oncalendar-shutdown) \
    | sed 's/^/OnCalendar=/'
  # Never run a missed shutdown at boot: the PC coming up at 07:00 must stay up.
  echo "Persistent=false"
  echo
  echo "[Install]"
  echo "WantedBy=timers.target"
} > /etc/systemd/system/kaya-shutdown.timer

if [ ! -f /etc/kaya-power.env ]; then
  cat > /etc/kaya-power.env <<ENV
# Read by /usr/local/sbin/kaya-shutdown.sh. Root-only: it holds the relay token.
POWER_USER=gustavo
POWER_REPO=/home/gustavo/kaya-prod
POWER_PYTHON=python3
POWER_PC_APP_URL=http://127.0.0.1:7860
POWER_GATEWAY_URL=http://192.168.1.238:8088
KAYA_RELAY_TOKEN=
ENV
  chmod 600 /etc/kaya-power.env
  echo "wrote /etc/kaya-power.env: set KAYA_RELAY_TOKEN there"
fi

# Wake-on-LAN must survive reboots; NetworkManager resets the NIC otherwise.
conn="$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: '$2=="enp4s0"{print $1}')"
if [ -n "$conn" ]; then
  nmcli connection modify "$conn" 802-3-ethernet.wake-on-lan magic
  echo "Wake-on-LAN (magic packet) enabled on '$conn'"
fi

systemctl daemon-reload
systemctl enable --now kaya-shutdown.timer
systemctl list-timers kaya-shutdown.timer --no-pager
