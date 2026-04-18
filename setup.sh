#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="2026010688"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r requirement.txt

echo "Setup complete. Run: conda activate $ENV_NAME"
