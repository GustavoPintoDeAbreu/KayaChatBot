#!/usr/bin/env bash
#
# Resolve the GPU UUIDs in .env to the CURRENT nvidia-smi indices and export them
# as KAYA_GPU_PROD / KAYA_GPU_DEV for docker compose.
#
#   source scripts/gpu_env.sh
#
# Why this exists: this host's container runtime rejects a UUID in
# NVIDIA_VISIBLE_DEVICES — it routes the value to CDI, and no CDI spec is
# installed ("unresolvable CDI devices nvidia.com/gpu=GPU-..."). Only indices
# work. But indices are not stable across reboots, which is exactly why the
# UUIDs are the source of truth (gpu-power-limit.service pins by UUID for the
# same reason). So the UUID is authoritative and the index is derived here, at
# launch, every time.
#
# .env holds both:
#   KAYA_GPU_PROD_UUID / KAYA_GPU_DEV_UUID  — authoritative
#   KAYA_GPU_PROD      / KAYA_GPU_DEV       — last known index, what compose reads
#
# If the resolved index differs from what .env records, this rewrites .env so a
# direct `docker compose` invocation (which does not source this file) stays
# correct too.

_gpu_env_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_gpu_index_for_uuid() {
  local uuid="$1"
  [[ -z "$uuid" ]] && return 1
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
    | awk -F', *' -v u="$uuid" '$2==u {print $1; found=1} END{exit !found}'
}

_gpu_env_resolve() {
  local role="$1" envfile="$_gpu_env_repo/.env"
  local uuid_key="KAYA_GPU_${role}_UUID" idx_key="KAYA_GPU_${role}"
  local uuid idx recorded

  uuid="$(grep -E "^${uuid_key}=" "$envfile" 2>/dev/null | tail -1 | cut -d= -f2-)"
  recorded="$(grep -E "^${idx_key}=" "$envfile" 2>/dev/null | tail -1 | cut -d= -f2-)"

  if [[ -z "$uuid" ]]; then
    # No UUID pinned — leave whatever .env says (possibly empty -> `all`).
    export "$idx_key=$recorded"
    return 0
  fi

  if ! idx="$(_gpu_index_for_uuid "$uuid")"; then
    echo "⚠️  ${role}: GPU ${uuid} not present on this host." >&2
    echo "    Falling back to all GPUs. Check: nvidia-smi --query-gpu=index,uuid --format=csv" >&2
    export "$idx_key="
    return 0
  fi

  if [[ "$idx" != "$recorded" ]]; then
    echo "ℹ️  ${role}: GPU index moved ${recorded:-<unset>} → ${idx} (UUID unchanged); updating .env"
    if grep -qE "^${idx_key}=" "$envfile"; then
      sed -i -E "s|^${idx_key}=.*|${idx_key}=${idx}|" "$envfile"
    else
      printf '%s=%s\n' "$idx_key" "$idx" >> "$envfile"
    fi
  fi
  export "$idx_key=$idx"
}

_gpu_env_resolve PROD
_gpu_env_resolve DEV
