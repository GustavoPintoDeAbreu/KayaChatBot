#!/usr/bin/env bash
#
# Keep ~/kaya-prod/.env safe from deploys.
#
#   scripts/prod_env.sh update <env-file>   upsert the secrets found in the environment
#   scripts/prod_env.sh check  <env-file>   refuse (exit 1) an .env prod cannot run on
#
# Why (2026-10-04): the Deploy (prod) workflow rewrote .env from GitHub secrets that
# were never set. Every value came out empty, and the settings kept only in that file
# (KAYA_EDGE=pi, the broker URL, the relay token) were gone. The deploy then fell back
# to edge=local: it started a second WAHA on the PC that took Kaya's WhatsApp session
# from the Pi's, plus a tokenless cloudflared and the old kaya-llama.
#
# So `update` never truncates. A secret that is set (non-empty) replaces or adds its
# own line; an unset one leaves the existing line alone. `check` fails before
# anything is touched when a setting prod needs is missing or empty.
set -euo pipefail

# The secrets the workflow may pass in (as environment variables of the same name).
SECRET_KEYS=(XAI_API_KEY AZURE_OPENAI_API_KEY_gpt_41_mini AZURE_OPENAI_API_KEY_gpt_53_chat
             KAYA_WEB_USER KAYA_WEB_PASS CLOUDFLARE_TUNNEL_TOKEN)

value_in() { sed -n "s/^$2=//p" "$1" | tail -1 | tr -d '"'"'"; }

update() {
  local file="$1" key val tmp
  touch "$file"; chmod 600 "$file"
  for key in "${SECRET_KEYS[@]}"; do
    val="${!key:-}"
    [[ -n "$val" ]] || continue                       # unset secret: keep what is there
    tmp="$(mktemp)"
    grep -v "^$key=" "$file" > "$tmp" || true
    printf '%s=%s\n' "$key" "$val" >> "$tmp"
    cat "$tmp" > "$file"; rm -f "$tmp"                # same inode, same mode
    echo "  .env: $key updated"
  done
}

check() {
  local file="$1" missing=() key edge tunnel
  [[ -f "$file" ]] || { echo "❌ $file missing" >&2; exit 1; }
  edge="$(value_in "$file" KAYA_EDGE)"
  tunnel="$(value_in "$file" KAYA_TUNNEL)"; tunnel="${tunnel:-$edge}"
  # KAYA_EDGE has no safe default: unset meant "local", which starts a second WAHA
  # next to the Pi's. It must be stated.
  local required=(KAYA_WEB_USER KAYA_WEB_PASS KAYA_EDGE)
  [[ "$edge" == "pi" ]] && required+=(KAYA_PROD_WAHA_URL KAYA_RELAY_TOKEN)
  [[ "$tunnel" == "local" ]] && required+=(CLOUDFLARE_TUNNEL_TOKEN)
  [[ "$(value_in "$file" KAYA_INFERENCE_BACKEND)" == "ollama" ]] && required+=(KAYA_PROD_OLLAMA_URL)
  for key in "${required[@]}"; do
    [[ -n "$(value_in "$file" "$key")" ]] || missing+=("$key")
  done
  if [[ -n "$edge" && "$edge" != "pi" && "$edge" != "local" ]]; then
    echo "❌ $file: KAYA_EDGE must be pi or local, not '$edge'" >&2; exit 1
  fi
  if (( ${#missing[@]} )); then
    echo "❌ $file is missing or has empty: ${missing[*]}. Refusing to deploy; nothing was changed." >&2
    exit 1
  fi
  echo "✅ $file: edge=$edge tunnel=$tunnel, required settings present"
}

case "${1:-}" in
  update) update "${2:?env file}" ;;
  check)  check "${2:?env file}" ;;
  *) echo "usage: $0 update|check <env-file>" >&2; exit 2 ;;
esac
