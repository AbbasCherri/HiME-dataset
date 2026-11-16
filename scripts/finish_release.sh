#!/usr/bin/env bash
# Finish the release without Docker: table, manifest, report, verify, tests.
# Run from the repo root in WSL2/Linux after `uv sync --frozen` and with data/ in place.
# Stops at the first failing step and prints which one. Log: reports/finish_release.log
set -u -o pipefail
log=reports/finish_release.log
: > "$log"

run() {
  echo "=== $*" | tee -a "$log"
  "$@" 2>&1 | grep -v -E '_THREADS' | tee -a "$log"
  code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    echo "FAILED (exit $code): $*" | tee -a "$log"
    exit "$code"
  fi
}

run uv run python -m notodata table
run uv run python -m notodata manifest
run uv run python -m notodata report
run uv run python -m notodata verify --reuse-tiles
run uv run pytest -q -rs
echo "ALL STEPS PASSED" | tee -a "$log"
