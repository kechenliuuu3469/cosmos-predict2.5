#!/usr/bin/env bash
# Bootstrap the cosmos-predict2.5 environment on a fresh cluster.
# Usage:  bash scripts/install_env.sh [cu128|cu130]
set -euo pipefail

CUDA_EXTRA="${1:-cu128}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[install_env] Repo: $REPO_ROOT"
echo "[install_env] CUDA extra: $CUDA_EXTRA"

if ! command -v uv >/dev/null 2>&1; then
  echo "[install_env] uv not found — installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[install_env] uv: $(uv --version)"

echo "[install_env] Running: uv sync --extra=$CUDA_EXTRA"
echo "$CUDA_EXTRA" > .cuda-name
uv sync --extra="$CUDA_EXTRA"

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
