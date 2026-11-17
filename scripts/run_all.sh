#!/usr/bin/env bash
# Master script for the WSL2 machine: setup, data build (resumes or starts from scratch),
# GPU embeddings, then table/manifest/report/verify/tests.
# Usage (from the repo root):   ./scripts/run_all.sh
# Resumable: every stage skips finished work, so just re-run after a failure.
# Log: reports/run_all.log. Stops at the first failing step and names it.
set -u -o pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p reports
log=reports/run_all.log
: > "$log"

step() {
  local name="$1"; shift
  echo "##### $name  ($(date -u +%FT%TZ))" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    echo "##### FAILED at step '$name' (exit $code). Fix it and re-run ./scripts/run_all.sh" | tee -a "$log"
    exit "$code"
  fi
}

step "1/4 setup"                 bash scripts/setup_wsl.sh --allow-empty-data
export UV_NO_SYNC=1              # keep the embed extra installed; uv run must not re-sync
# source, labels, geometry, fetch, chips, coverage, imagery..clients, table, manifest, report.
# Every stage skips finished work, so an empty data/ means a full fresh build (fetch takes hours).
step "2/4 data build"            uv run python -m notodata build
step "3/4 real embeddings (GPU)" bash scripts/embed_real.sh
step "4/4 table, manifest, report, verify, tests" bash scripts/finish_release.sh

echo "##### ALL DONE. Send me reports/run_all.log (and reports/finish_release.log)." | tee -a "$log"
