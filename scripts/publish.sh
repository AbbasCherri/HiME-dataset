#!/usr/bin/env bash
# Publish the verified dataset to Hugging Face. A dry run unless you pass --yes
# (add --include-imagery for chips/ and tiles/). Needs HF_TOKEN in ./.env (never echoed).
# Run from the repo root after `verify` has passed. Log: reports/publish.log
set -u -o pipefail
cd "$(dirname "$0")/.."
log=reports/publish.log
: > "$log"

if [ -f ./.env ]; then
  set -a; . ./.env; set +a
fi

echo "=== notodata publish $*" | tee -a "$log"
uv run python -m notodata publish "$@" 2>&1 | grep -v -E '_THREADS' | tee -a "$log"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
  echo "FAILED (exit $code): notodata publish $*" | tee -a "$log"
  exit "$code"
fi
