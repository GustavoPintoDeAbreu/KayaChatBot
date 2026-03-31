#!/usr/bin/env bash
# Setup GitHub labels for the Copilot coding agent workflow.
# Usage: GITHUB_TOKEN=ghp_... bash .github/scripts/setup-labels.sh OWNER/REPO
#
# Requires: gh CLI (https://cli.github.com/) authenticated, or GITHUB_TOKEN set.

set -euo pipefail

REPO="${1:?Usage: setup-labels.sh OWNER/REPO}"

declare -A LABELS=(
  # Task type labels
  ["bug"]="d73a4a:Something isn't working"
  ["feature"]="a2eeef:New feature or request"
  ["improvement"]="7057ff:Enhancement to existing functionality"
  ["test"]="e4e669:Test coverage or quality improvement"

  # Priority labels
  ["priority:high"]="b60205:High priority — address soon"
  ["priority:medium"]="fbca04:Medium priority"
  ["priority:low"]="0e8a16:Low priority"

  # Agent labels
  ["agent:bug-fixer"]="d93f0b:Routed to bug-fixer agent"
  ["agent:feature-dev"]="0075ca:Routed to feature-dev agent"
  ["agent:test-specialist"]="5319e7:Routed to test-specialist agent"
)

for label in "${!LABELS[@]}"; do
  IFS=':' read -r color description <<< "${LABELS[$label]}"
  echo "Creating label: $label"
  gh label create "$label" \
    --repo "$REPO" \
    --color "$color" \
    --description "$description" \
    --force 2>/dev/null || echo "  (already exists or updated)"
done

echo "Done. All labels created."
