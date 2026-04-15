#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh list
  ./run.sh grounding-dino
  ./run.sh sam2
  ./run.sh qwen3-vl

Before running:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirement.txt
USAGE
}

prepare_paths() {
  if [ ! -d "samples" ]; then
    echo "Error: samples directory is missing." >&2
    exit 1
  fi

  if [ ! -e "foundation-models" ]; then
    ln -s samples foundation-models
  fi
}

run_python() {
  if command -v python >/dev/null 2>&1; then
    python "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    echo "Error: python or python3 was not found." >&2
    exit 1
  fi
}

command_name="${1:-list}"

case "$command_name" in
  list)
    usage
    ;;
  grounding-dino|grounding|dino)
    prepare_paths
    run_python "Grounding-DINO.py"
    ;;
  sam2|sam)
    prepare_paths
    run_python "SAM2-small.py"
    ;;
  qwen3-vl|qwen|vlm)
    prepare_paths
    run_python "Qwen3-VL-#B-Instruct.py"
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    usage >&2
    exit 1
    ;;
esac
