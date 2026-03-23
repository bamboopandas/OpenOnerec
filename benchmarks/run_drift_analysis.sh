#!/bin/bash
# Script to run Directional Drift Analysis

# Usage: bash run_drift_analysis.sh <GPU_ID> <TASK> <LIMIT>
# Example: bash run_drift_analysis.sh 0 video 100

GPU_ID="${1:-0}"
TASK="${2:-video}"
LIMIT="${3:-100}"
OUTPUT_FILE="drift_results_${TASK}.jsonl"
MODEL_PATH="../checkpoints/OneRec-1.7B"
DATA_DIR="../raw_data/onerec_data/benchmark_data_1000"

echo "Running drift analysis for task: $TASK on GPU: $GPU_ID with limit: $LIMIT"
echo "Output file: $OUTPUT_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID /home/lkzhang/miniconda3/envs/openonerec/bin/python3 drift_analysis.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --task "$TASK" \
    --limit "$LIMIT" \
    --output_file "$OUTPUT_FILE"

echo "Done."

