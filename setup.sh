#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage:
  ./setup.sh [cpu|cu124|cu128] [env_name]

Examples:
  ./setup.sh
  ./setup.sh cpu
  ./setup.sh cu124
  ./setup.sh cu124 2026010688

Behavior:
  - Creates a conda environment automatically when conda is available
  - Reinstalls torch/torchvision/torchaudio for the selected runtime
  - Installs pinned Python dependencies from requirement.txt
  - Prints the final torch/CUDA status

Notes:
  - Default is `cu124` when `nvidia-smi` is available, otherwise `cpu`
  - Default conda env name is `2026010688`
  - Without conda, use this inside your activated venv or Python environment
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

runtime="${1:-}"
env_name="${2:-2026010688}"
if [[ -z "$runtime" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    runtime="cu124"
  else
    runtime="cpu"
  fi
fi

case "$runtime" in
  cpu)
    torch_index="https://download.pytorch.org/whl/cpu"
    ;;
  cu124)
    torch_index="https://download.pytorch.org/whl/cu124"
    ;;
  cu128)
    torch_index="https://download.pytorch.org/whl/cu128"
    ;;
  *)
    echo "Unsupported runtime: $runtime" >&2
    usage >&2
    exit 1
    ;;
esac

if ! command -v python >/dev/null 2>&1; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: neither python nor conda was found in PATH." >&2
    exit 1
  fi
fi

python_cmd=(python)
if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "[0/4] Creating conda environment: $env_name"
    conda create -n "$env_name" python=3.10 -y
  fi
  python_cmd=(conda run --no-capture-output -n "$env_name" python)
fi

echo "[1/4] Upgrading pip"
"${python_cmd[@]}" -m pip install --upgrade pip

echo "[2/4] Reinstalling torch stack for $runtime"
"${python_cmd[@]}" -m pip uninstall -y torch torchvision torchaudio || true
"${python_cmd[@]}" -m pip install --no-cache-dir --index-url "$torch_index" \
  torch torchvision torchaudio

echo "[3/4] Installing pinned project dependencies"
"${python_cmd[@]}" -m pip install --no-cache-dir -r requirement.txt

echo "[4/4] Verifying torch runtime"
"${python_cmd[@]}" -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

if [[ "$runtime" != "cpu" ]]; then
  "${python_cmd[@]}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo
    echo "Torch installed, but CUDA is still unavailable." >&2
    echo "Try one of these:" >&2
    echo "  1. ./setup.sh cu124" >&2
    echo "  2. ./setup.sh cu128" >&2
    echo "  3. nvidia-smi" >&2
    exit 1
  }
fi

if command -v conda >/dev/null 2>&1; then
  echo
  echo "Done. Use this environment with:"
  echo "  conda activate $env_name"
fi
