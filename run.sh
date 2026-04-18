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
  ./setup.sh
  conda activate 2026010688
USAGE
}

check_samples() {
  if [ ! -d "samples" ]; then
    echo "Error: samples directory is missing." >&2
    exit 1
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
    check_samples
    run_python "Grounding-DINO.py"
    ;;
  sam2|sam)
    check_samples
    run_python "SAM2-base-plus.py"
    ;;
  qwen3-vl|qwen|vlm)
    check_samples
    run_python "Qwen3-VL-8B-Instruct.py"
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    usage >&2
    exit 1
    ;;
esac
