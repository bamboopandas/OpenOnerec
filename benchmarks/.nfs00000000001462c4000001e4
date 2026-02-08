#!/bin/bash
# Usage: bash benchmarks/eval_alignment.sh <model_path> <result_name> [custom_data_dir] [gpu_id]
# Example: bash benchmarks/eval_alignment.sh checkpoints/OneRec-1.7B align_test data_v1.0 0
# bash benchmarks/eval_alignment.sh checkpoints/OneRec-1.7B align_test raw_data/onerec_data/benchmark_data_1000 0
# Set common variables
MODEL_PATH=$1
RESULT_NAME=$2
CUSTOM_DATA_DIR=$3
GPU_ID="${4:-0}"
VERSION="${VERSION:-v1.0_alignment}"

# Read configuration from environment variables
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-.}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

# Base directory for outputs and logs
BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${RESULT_NAME}/$(basename "${MODEL_PATH}")"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/${RESULT_NAME}"

if [ -n "$CUSTOM_DATA_DIR" ]; then
    BENCHMARK_DATA_DIR="$CUSTOM_DATA_DIR"
else
    BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${BENCHMARK_BASE_DIR}/data_${DATA_VERSION}}"
fi
DATA_DIR="$BENCHMARK_DATA_DIR"

# Create output directory and log directory
mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

# Write debug info to log file
{
    echo "========== Alignment Configuration =========="
    echo "DATE: $(date)"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "DATA_DIR: $DATA_DIR"
    echo "GPU_ID: $GPU_ID"
    echo "========================================="
} >> "${BASE_LOG_NAME}.log"

export PYTHONPATH="${BENCHMARK_BASE_DIR}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID

PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# List of tasks to run (matching the structure of benchmark data)
TASKS=("ad" "product" "video")

for TASK in "${TASKS[@]}"; do
    # Try multiple common patterns
    POSSIBLE_PATHS=(
        "${DATA_DIR}/${TASK}/test.parquet"
        "${DATA_DIR}/${TASK}/${TASK}_test.parquet"
        "${DATA_DIR}/${TASK}.parquet"
    )
    
    TASK_DATA_PATH=""
    for P in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$P" ]; then
            TASK_DATA_PATH="$P"
            break
        fi
    done

    if [ -n "$TASK_DATA_PATH" ]; then
        echo "Running alignment for task: $TASK"
        echo "Processing $TASK_DATA_PATH..."
        
        $PYTHON_EXEC -u benchmarks/run_inference_alignment.py \
            --data_path "$TASK_DATA_PATH" \
            --model_path "$MODEL_PATH" \
            --output_path "${BASE_OUTPUT_DIR}/${TASK}_alignment.jsonl" \
            --num_samples "${NUM_SAMPLES:-1000}" \
            --top_k 5 \
            --alpha 0.5 \
            --beta0 0.0 \
            --beta1 1.0 >> "${BASE_LOG_NAME}.log" 2>&1
            
        echo "$TASK task alignment completed successfully"
    else
        echo "Warning: Data file for task $TASK not found at $TASK_DATA_PATH. Skipping."
    fi
done

echo "All alignment tasks completed successfully. Results in $BASE_OUTPUT_DIR"
