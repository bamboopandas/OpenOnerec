#!/bin/bash

# Set common variables
MODEL_PATH=$1
if [ -n "$MODEL_PATH" ] && [ -e "$MODEL_PATH" ]; then
    MODEL_PATH=$(readlink -f "$MODEL_PATH")
fi
VERSION="${VERSION:-power_decoding_test_vtemp}"
ENABLE_THINKING=$3
CUSTOM_DATA_DIR=$4

echo "DEBUG: Script arguments:"
echo "  1 (MODEL_PATH): '$1'"
echo "  2 (VERSION suffix): '$2'"
echo "  3 (ENABLE_THINKING): '$3'"
echo "  4 (CUSTOM_DATA_DIR): '$4'"

# Detect script location and root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Assuming script is in benchmarks/eval_script/ or benchmarks/
# If in eval_script, parent is benchmarks. If in benchmarks, current is benchmarks.
if [[ "$(basename "$SCRIPT_DIR")" == "eval_script" ]]; then
    BENCHMARK_BASE_DIR="$(dirname "$SCRIPT_DIR")"
else
    BENCHMARK_BASE_DIR="$SCRIPT_DIR"
fi

# Fallback for when running from root or elsewhere if detection fails
if [ ! -d "$BENCHMARK_BASE_DIR/benchmark" ]; then
    # Maybe we are in root and BENCHMARK_BASE_DIR should be ./benchmarks
    if [ -d "benchmarks/benchmark" ]; then
        BENCHMARK_BASE_DIR="benchmarks"
    else
        BENCHMARK_BASE_DIR="."
    fi
fi

# Read configuration from environment variables
DATA_VERSION="${DATA_VERSION:-v1.0}"
PYTHON_EXEC="${PYTHON_EXEC:-/home/lkzhang/miniconda3/envs/openonerec/bin/python3}"

# Output Config
BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/$2"

if [ -n "$CUSTOM_DATA_DIR" ]; then
    BENCHMARK_DATA_DIR="$CUSTOM_DATA_DIR"
else
    BENCHMARK_DATA_DIR="${BENCHMARK_BASE_DIR}/data_${DATA_VERSION}"
fi
DATA_DIR="$BENCHMARK_DATA_DIR"

mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

# Sample size config
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"  # Default to 1000 per user hint, was 10

{
    echo "========== Task Configuration (Power Decoding) =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "Benchmark Base: $BENCHMARK_BASE_DIR"
    echo "Sample Size: $SAMPLE_SIZE"
    echo "Python Exec: $PYTHON_EXEC"
    echo "Model Path: $MODEL_PATH"
    echo "======== Task Config End ========"
} >> "${BASE_LOG_NAME}.log"

THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

export PYTHONPATH="${BENCHMARK_BASE_DIR}:$PYTHONPATH"

echo "Running Power Decoding Evaluation"
echo "Log file: ${BASE_LOG_NAME}.log"

# Script location relative to BENCHMARK_BASE_DIR or absolute
# Using the new script in scripts/ray-vllm/
EVAL_PY="${BENCHMARK_BASE_DIR}/scripts/ray-vllm/evaluate_power_decoding.py"
echo "Eval script: $EVAL_PY"

# Task: ad
echo "Starting task: ad"
$PYTHON_EXEC -u "$EVAL_PY" \
    --task_types ad \
    --num_gpus 1 \
    --gpu_ids 2 \
    --gpu_memory_utilization 0.8 \
    --max_num_seqs 256 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --overwrite \
    --use_power_decoding \
    --alpha 2.0 \
    --top_k_candidates 5 \
    --max_rollouts 5 \
    --max_lookahead 3 \
    --crit_threshold 0.5 \
    --sample_size "$SAMPLE_SIZE" \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

# Task: product
echo "Starting task: product"
$PYTHON_EXEC -u "$EVAL_PY" \
    --task_types product \
    --num_gpus 1 \
    --gpu_ids 2 \
    --gpu_memory_utilization 0.8 \
    --max_num_seqs 256 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --overwrite \
    --use_power_decoding \
    --alpha 2.0 \
    --top_k_candidates 5 \
    --max_rollouts 5 \
    --max_lookahead 3 \
    --crit_threshold 0.5 \
    --sample_size "$SAMPLE_SIZE" \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

# Task: video
echo "Starting task: video"
$PYTHON_EXEC -u "$EVAL_PY" \
    --task_types video \
    --num_gpus 1 \
    --gpu_ids 2 \
    --gpu_memory_utilization 0.8 \
    --max_num_seqs 256 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --overwrite \
    --use_power_decoding \
    --alpha 2.0 \
    --top_k_candidates 5 \
    --max_rollouts 5 \
    --max_lookahead 3 \
    --crit_threshold 0.5 \
    --sample_size "$SAMPLE_SIZE" \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Power Decoding evaluation completed"
