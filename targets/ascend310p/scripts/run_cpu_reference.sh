#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${AI_MODEL_PYTHON:-python3}"
model_dir="${QWEN35_MODEL_DIR:?set QWEN35_MODEL_DIR to the locked checkpoint}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/model${PYTHONPATH:+:$PYTHONPATH}"
export QWEN35_MODEL_DIR="$model_dir"

"$python_bin" -m qwen35_mtp audit --model-dir "$model_dir"
"$python_bin" -m unittest discover -s "$repo_root/tests" -v
