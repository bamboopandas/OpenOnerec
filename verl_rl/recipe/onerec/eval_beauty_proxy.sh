#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export INPUT_PATH=${INPUT_PATH:-"$PROJECT_DIR/output/beauty_proxy_runs/val/validation_dumps"}
export K=${K:-10}
export OUTPUT_PATH=${OUTPUT_PATH:-"$INPUT_PATH/eval_summary.json"}
export DETAILS_PATH=${DETAILS_PATH:-"$INPUT_PATH/eval_details.json"}

python3 -u -m recipe.onerec.evaluate_beauty_proxy \
    --input_path "$INPUT_PATH" \
    --k "$K" \
    --output_path "$OUTPUT_PATH" \
    --details_path "$DETAILS_PATH"
