#!/usr/bin/env bash
# Bootstrap the cosmos-predict2.5 environment on a fresh machine.
# Usage:  bash scripts/install_env.sh [cu128|cu130] [--china] [--offline]
#
# --offline uses only a pre-populated uv cache; set UV_CACHE_DIR first,
# e.g. export UV_CACHE_DIR=$HOME/uv-cache-cosmos-cu130
set -euo pipefail

CUDA_EXTRA="cu128"
USE_CHINA_MIRROR=0
OFFLINE=0
for arg in "$@"; do
  case "$arg" in
    --china) USE_CHINA_MIRROR=1 ;;
    --offline) OFFLINE=1 ;;
    cu128|cu130) CUDA_EXTRA="$arg" ;;
    *) echo "[install_env] Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[install_env] Repo: $REPO_ROOT"
echo "[install_env] CUDA extra: $CUDA_EXTRA"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

if [[ "$USE_CHINA_MIRROR" == "1" ]]; then
  echo "[install_env] Using China mirrors (Tsinghua)"
  export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  export UV_EXTRA_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/${CUDA_EXTRA}"
fi

SYNC_FLAGS=()
if [[ "$OFFLINE" == "1" ]]; then
  echo "[install_env] Offline mode: UV_CACHE_DIR=${UV_CACHE_DIR:-<unset>}"
  if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    echo "[install_env] ERROR: --offline requires UV_CACHE_DIR to be set" >&2
    exit 1
  fi
  if [[ ! -d "$UV_CACHE_DIR" ]]; then
    echo "[install_env] ERROR: UV_CACHE_DIR does not exist: $UV_CACHE_DIR" >&2
    exit 1
  fi
  SYNC_FLAGS+=(--offline)
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[install_env] uv not found — installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[install_env] uv: $(uv --version)"

echo "[install_env] Running: uv sync --extra=$CUDA_EXTRA ${SYNC_FLAGS[*]:-}"
echo "$CUDA_EXTRA" > .cuda-name
uv sync --extra="$CUDA_EXTRA" "${SYNC_FLAGS[@]}"

echo "[install_env] Verifying torch + CUDA"
uv run --no-sync python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
PY

cat <<EOF

[install_env] Done.
  Activate:  source .venv/bin/activate
  Or run:    uv run --no-sync python ...
EOF
