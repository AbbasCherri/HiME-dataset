#!/usr/bin/env bash
# Idempotent setup for a Linux/WSL2 machine without Docker. Safe to re-run.
# Checks or installs: git, curl, uv, the pinned environment (with the GPU `embed` extra),
# then checks RAM, the GPU and that the copied data/ folder is complete.
set -u -o pipefail
cd "$(dirname "$0")/.."
allow_empty=0; [ "${1:-}" = "--allow-empty-data" ] && allow_empty=1

fail() { echo "SETUP FAILED: $*" >&2; exit 1; }
ok()   { echo "ok: $*"; }

[ "$(uname -s)" = Linux ] || fail "run this inside WSL2/Linux, not Windows"
case "$PWD" in /mnt/*) fail "repo is on the Windows drive ($PWD); move it into ~/ (WSL filesystem), it is far faster" ;; esac

# system packages
need=()
for c in git curl; do command -v "$c" >/dev/null || need+=("$c"); done
if [ "${#need[@]}" -gt 0 ]; then
  echo "installing: ${need[*]} (sudo needed)"
  sudo apt-get update -y && sudo apt-get install -y "${need[@]}" || fail "apt install ${need[*]}"
fi
ok "git and curl"

# uv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install"
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || fail "uv not on PATH; reopen the shell and re-run"
ok "uv $(uv --version)"

# pinned environment, with the GPU embed extra
uv sync --frozen --extra embed || fail "uv sync --frozen --extra embed"
ok "environment synced"

# resources
mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
[ "$mem_gb" -ge 12 ] && ok "RAM ${mem_gb} GB" || echo "WARNING: only ${mem_gb} GB RAM visible to WSL; set memory=20GB in C:\\Users\\<you>\\.wslconfig then run 'wsl --shutdown'"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "WARNING: nvidia-smi not working in WSL; install the current NVIDIA Windows driver (not a Linux driver). Embeddings need the GPU."
fi
uv run --no-sync python -c 'import torch,sys; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)' \
  && ok "torch sees the GPU" || echo "WARNING: torch cannot use CUDA yet"

# data: either copied from the first machine, or absent (then `build` fetches everything)
home="${NOTODATA_HOME:-./data}"
if [ -d "$home/tiles" ] && [ -d "$home/chips" ]; then
  ok "existing data folder found ($(du -sh "$home" | cut -f1)); finished stages will be skipped"
elif [ "$allow_empty" -eq 1 ]; then
  echo "note: no data folder at $home; the build will fetch and compute everything from scratch (many hours)"
else
  fail "no data folder at $home (re-run with --allow-empty-data to build from scratch)"
fi

[ -f reports/imagery_decision.json ] || fail "reports/imagery_decision.json missing: wrong branch? run 'git checkout build/notodata && git pull'"
ok "branch $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"
echo "SETUP OK"
