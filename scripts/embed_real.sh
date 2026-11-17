#!/usr/bin/env bash
# Build the real dinov2_vits14 embeddings (needs the [embed] extra; uses the GPU if present).
# Run from the repo root in WSL2/Linux with data/ and the chips in place.
# Stops at the first failing step and prints which one. Log: reports/embed_real.log
set -u -o pipefail
log=reports/embed_real.log
mkdir -p reports
: > "$log"

run() {
  echo "=== $*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    echo "FAILED (exit $code): $*" | tee -a "$log"
    exit "$code"
  fi
}

run uv sync --frozen --extra embed
run uv run python -c 'import torch;print(torch.cuda.is_available())'
if ! tail -n 1 "$log" | grep -qx True; then
  echo "ERROR: torch.cuda.is_available() is False; embeddings must be built on the GPU" | tee -a "$log"
  exit 1
fi
run uv run python -m notodata embed --encoder dinov2_vits14
echo "ALL STEPS PASSED" | tee -a "$log"
