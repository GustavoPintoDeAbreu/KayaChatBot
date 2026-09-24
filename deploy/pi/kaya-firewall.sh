#!/usr/bin/env bash
# LAN-side firewall for the Pi's published ports. Docker-published ports skip the
# INPUT chain, so the rules go in DOCKER-USER, which Docker evaluates first.
# Only the PC may reach WAHA (3000: replies, media) and the gateway's internal
# listener (8088: media, going-down, webhook). Idempotent; run after docker starts.
set -euo pipefail
PC_IP="${PC_IP:-192.168.1.149}"
iptables -N DOCKER-USER 2>/dev/null || true
for port in 3000 8088; do
  while iptables -D DOCKER-USER -i eth0 -p tcp -m conntrack --ctorigdstport "$port" ! -s "$PC_IP" -j DROP 2>/dev/null; do :; done
  iptables -I DOCKER-USER -i eth0 -p tcp -m conntrack --ctorigdstport "$port" ! -s "$PC_IP" -j DROP
done
echo "[kaya-firewall] 3000 and 8088 reachable from $PC_IP only"
