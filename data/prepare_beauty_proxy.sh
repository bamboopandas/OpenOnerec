#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/../output/beauty_proxy"}
export SIDECAR_INDEX_OUT=${SIDECAR_INDEX_OUT:-"sidecar_index.parquet"}
export MAX_HISTORY_LEN=${MAX_HISTORY_LEN:-50}
export CAPTION_MAX_CHARS=${CAPTION_MAX_CHARS:-96}
export MAX_ROWS_PER_SPLIT=${MAX_ROWS_PER_SPLIT:-0}
export PREDICTION_DATA_DIR=${PREDICTION_DATA_DIR:-""}
export SEQUENTIAL_FILE=${SEQUENTIAL_FILE:-""}
export ITEMS_FILE=${ITEMS_FILE:-""}

mkdir -p "$OUTPUT_DIR"

ARGS=(
    --output_dir "$OUTPUT_DIR"
    --sidecar_index_out "$SIDECAR_INDEX_OUT"
    --max_history_len "$MAX_HISTORY_LEN"
    --caption_max_chars "$CAPTION_MAX_CHARS"
    --max_rows_per_split "$MAX_ROWS_PER_SPLIT"
)

if [ -n "$PREDICTION_DATA_DIR" ]; then
    ARGS+=(--prediction_data_dir "$PREDICTION_DATA_DIR")
else
    if [ -z "$SEQUENTIAL_FILE" ] || [ -z "$ITEMS_FILE" ]; then
        echo "Either PREDICTION_DATA_DIR or both SEQUENTIAL_FILE and ITEMS_FILE must be set."
        exit 1
    fi
    ARGS+=(--sequential_file "$SEQUENTIAL_FILE" --items_file "$ITEMS_FILE")
fi

python3 "$SCRIPT_DIR/scripts/prepare_onerec_think_beauty_rl_data.py" "${ARGS[@]}"
