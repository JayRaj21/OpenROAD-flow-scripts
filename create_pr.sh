#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BODY_FILE="$SCRIPT_DIR/create_pr_body.md"

gh pr create \
  --title "pr-extension: LLM-driven P&R triage, closed-loop optimization, and congestion ML pipeline" \
  --body-file "$BODY_FILE" \
  --base master
