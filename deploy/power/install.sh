#!/usr/bin/env bash
# Install the GPU PC's power units. Run with sudo from the repo root:
#   sudo deploy/power/install.sh [--uninstall]
# kaya-going-down.service announces every poweroff to the Pi. The shutdown
# timer's OnCalendar lines are generated from config.yaml `power:` so the PC, the
# Pi's Wake-on-LAN timer and the gateway's offline reply cannot disagree; with no
# shutdown times there is no timer, and the PC is turned off by hand.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${POWER_PYTHON:-python3}"

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now kaya-shutdown.timer kaya-boot-check.service 2>/dev/null || true
  systemctl disable --now kaya-power-listener.service 2>/dev/null || true
  systemctl disable kaya-going-down.service 2>/dev/null || true
  rm -f /etc/systemd/system/kaya-shutdown.{timer,service} /usr/local/sbin/kaya-shutdown.sh \
        /etc/systemd/system/kaya-going-down.service /usr/local/sbin/kaya-going-down.sh \
        /etc/systemd/system/kaya-boot-check.service /usr/local/sbin/kaya-boot-check.sh \
        /usr/local/sbin/kaya-power-listener /etc/systemd/system/kaya-power-listener.service \
        /etc/systemd/system/kaya-shutdown-manual.service \
        /etc/apt/apt.conf.d/51kaya-no-kernel
  systemctl daemon-reload
  echo "removed"; exit 0
fi

install -m 0755 "$REPO/deploy/power/kaya-going-down.sh" /usr/local/sbin/kaya-going-down.sh
install -m 0644 "$REPO/deploy/power/kaya-going-down.service" /etc/systemd/system/kaya-going-down.service
install -m 0755 "$REPO/deploy/power/kaya-boot-check.sh" /usr/local/sbin/kaya-boot-check.sh
install -m 0644 "$REPO/deploy/power/kaya-boot-check.service" /etc/systemd/system/kaya-boot-check.service
# The manual mode needs the script even without a schedule.
install -m 0755 "$REPO/deploy/power/kaya-shutdown.sh" /usr/local/sbin/kaya-shutdown.sh
install -m 0755 "$REPO/deploy/power/kaya_power_listener.py" /usr/local/sbin/kaya-power-listener
install -m 0644 "$REPO/deploy/power/kaya-power-listener.service" /etc/systemd/system/kaya-power-listener.service
install -m 0644 "$REPO/deploy/power/kaya-shutdown-manual.service" /etc/systemd/system/kaya-shutdown-manual.service

# Kernels are upgraded by hand, never by unattended-upgrades: it upgraded the
# kernel and held back the matching nvidia module on 2026-09-25, and the next
# boot had no GPU driver. Everything else keeps getting security updates.
cat > /etc/apt/apt.conf.d/51kaya-no-kernel <<'APT'
// Installed by KayaChatBot deploy/power/install.sh. Upgrade kernels with
// `sudo apt full-upgrade` at the machine, so the nvidia module comes with them.
Unattended-Upgrade::Package-Blacklist {
    "linux-";
};
APT

oncalendar="$(cd "$REPO" && "$PY" -m src.gateway.schedule --config config.yaml oncalendar-shutdown)"
if [ -n "$oncalendar" ]; then
  install -m 0644 "$REPO/deploy/power/kaya-shutdown.service" /etc/systemd/system/kaya-shutdown.service
  {
    echo "[Unit]"
    echo "Description=Scheduled shutdown times (generated from config.yaml power.shutdown)"
    echo
    echo "[Timer]"
    echo "$oncalendar" | sed 's/^/OnCalendar=/'
    # Never run a missed shutdown at boot: the PC coming up at 07:00 must stay up.
    echo "Persistent=false"
    echo
    echo "[Install]"
    echo "WantedBy=timers.target"
  } > /etc/systemd/system/kaya-shutdown.timer
else
  systemctl disable --now kaya-shutdown.timer 2>/dev/null || true
  rm -f /etc/systemd/system/kaya-shutdown.{timer,service}
  echo "no shutdown times in config.yaml: no shutdown timer, the PC is turned off by hand"
fi

if [ ! -f /etc/kaya-power.env ]; then
  cat > /etc/kaya-power.env <<ENV
# Read by /usr/local/sbin/kaya-shutdown.sh. Root-only: it holds the relay token.
POWER_USER=gustavo
POWER_REPO=/home/gustavo/kaya-prod
POWER_PYTHON=python3
POWER_PC_APP_URL=http://127.0.0.1:7860
POWER_GATEWAY_URL=http://192.168.1.238:8088
POWER_LISTEN_PORT=8099
POWER_ALLOWED_IPS=192.168.1.238
KAYA_RELAY_TOKEN=${KAYA_RELAY_TOKEN:-}
ENV
  chmod 600 /etc/kaya-power.env
  [ -n "${KAYA_RELAY_TOKEN:-}" ] && echo "wrote /etc/kaya-power.env" \
    || echo "wrote /etc/kaya-power.env: set KAYA_RELAY_TOKEN there"
fi

# Wake-on-LAN must survive reboots; NetworkManager resets the NIC otherwise.
conn="$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: '$2=="enp4s0"{print $1}')"
if [ -n "$conn" ]; then
  nmcli connection modify "$conn" 802-3-ethernet.wake-on-lan magic
  echo "Wake-on-LAN (magic packet) enabled on '$conn'"
fi

systemctl daemon-reload
[ -n "$oncalendar" ] && systemctl enable --now kaya-shutdown.timer
systemctl enable kaya-boot-check.service
systemctl enable --now kaya-going-down.service
systemctl enable --now kaya-power-listener.service
systemctl restart kaya-power-listener.service
[ -n "$oncalendar" ] && systemctl list-timers kaya-shutdown.timer --no-pager
systemctl --no-pager --lines=0 status kaya-going-down.service
