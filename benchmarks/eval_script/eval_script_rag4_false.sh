#!/bin/bash
# bash eval_script_rag4.sh ../checkpoints/OneRec-1.7B results_1.7B false ../raw_data/onerec_data/benchmark_data_1000


unset LD_PRELOAD
export PATH=/home/lkzhang/miniconda3/envs/openonerec/bin:$PATH

# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0_1000_thinkfalse}"
# VERSION="${VERSION:-v10.0_false_RAG_id_v4_cleandata}"
ENABLE_THINKING=$3
CUSTOM_DATA_DIR=$4

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-.}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/$2"

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
    echo "========== Task Configuration =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"

export PYTHONPATH="${BENCHMARK_BASE_DIR}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_TORCH_COMPILE_LEVEL=0
export no_proxy=localhost,127.0.0.1,192.168.1.13,::1
export NO_PROXY=localhost,127.0.0.1,192.168.1.13,::1

echo "Running all tasks"
# nvidia-smi

PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# Cleanup function
cleanup() {
    echo "UnCleaning up..."
    # echo "Cleaning up..."
    # pkill -9 -f "scripts/ray-vllm/evaluate.py" || true
    # sleep 30
}

# Task: rec_reason
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types rec_reason \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: item_understand
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types item_understand \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: ad
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types ad \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: product
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types product \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: label_cond
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types label_cond \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: video
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types video \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: interactive
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types interactive \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

# Task: label_pred
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 5 \
    --task_types label_pred \
    --gpu_memory_utilization 0.7 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 32768 \
    --worker_batch_size 16 \
    --max_logprobs 10000 \
    --overwrite \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
cleanup

echo "All tasks completed successfully"
