#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

default_user="$(id -un 2>/dev/null || whoami 2>/dev/null || echo user)"
default_env_name="foundation-models-midterm-${default_user}"

pick_runtime_from_cuda_version() {
  local cuda_version="$1"
  local major minor

  IFS=. read -r major minor <<<"$cuda_version"
  major="${major:-0}"
  minor="${minor:-0}"

  if (( major > 12 || (major == 12 && minor >= 8) )); then
    echo "cu128"
  elif (( major > 12 || (major == 12 && minor >= 4) )); then
    echo "cu124"
  else
    echo "cpu"
  fi
}

detect_runtime() {
  local smi_output cuda_version runtime_choice

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  smi_output="$(nvidia-smi 2>/dev/null || true)"
  cuda_version="$(printf '%s\n' "$smi_output" | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\+\).*/\1/p' | head -n 1)"

  if [[ -z "$cuda_version" ]]; then
    echo "cpu"
    return
  fi

  runtime_choice="$(pick_runtime_from_cuda_version "$cuda_version")"
  echo "$runtime_choice"
}

usage() {
  cat <<USAGE
Usage:
  ./setup.sh [cpu|cu124|cu128] [env_name]

Examples:
  ./setup.sh
  ./setup.sh cpu
  ./setup.sh cu124
  ./setup.sh cu124 ${default_env_name}

Behavior:
  - Creates a per-user conda environment automatically
  - Reinstalls torch/torchvision/torchaudio for the selected runtime
  - Installs pinned Python dependencies from requirement.txt
  - Prints the final torch/CUDA status

Notes:
  - Default runtime is chosen from the CUDA version reported by nvidia-smi
  - CUDA >= 12.8 -> 'cu128', CUDA >= 12.4 -> 'cu124', otherwise 'cpu'
  - Default conda env name is '${default_env_name}'
  - Conda is required
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

runtime="${1:-}"
env_name="${2:-$default_env_name}"
if [[ -z "$runtime" ]]; then
  runtime="$(detect_runtime)"
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

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH." >&2
  echo "Please install conda first, then rerun ./setup.sh." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  echo "[0/4] Creating conda environment: $env_name"
  conda create -n "$env_name" python=3.10 -y
fi
python_cmd=(conda run --no-capture-output -n "$env_name" python)

echo "[info] Selected runtime: $runtime"

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

echo
echo "Done. Use this environment with:"
echo "  conda activate $env_name"
