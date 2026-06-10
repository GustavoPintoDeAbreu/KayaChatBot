#!/usr/bin/env bash
#
# Show what Kaya containers are running and current GPU usage.
#
#   scripts/app_status.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Kaya containers ==="
docker ps --filter "name=kaya-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true

echo
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null || nvidia-smi
else
  echo "nvidia-smi not available on host."
fi
