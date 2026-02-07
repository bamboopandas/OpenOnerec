#!/bin/bash
# bash eval_script_rag4.sh ../checkpoints/OneRec-1.7B results_1.7B false ../raw_data/onerec_data/benchmark_data_1000
# bash eval_script_rag4.sh ../checkpoints/OneRec-1.7B results_1.7B true ../raw_data/onerec_data/benchmark_data_1000


# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0_1000_thinkfalse_tryspeed}"
ENABLE_THINKING=$3
CUSTOM_DATA_DIR=$4
GPU_ID="${5:-0}"

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-.}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}/$(basename "${MODEL_PATH}")"
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

# Function to force cleanup processes on the specific GPU
cleanup_gpu() {
    local target_gpu=$1
    echo "Cleaning up processes on GPU $target_gpu..."
    
    # Get PIDs running on this GPU
    # nvidia-smi query returns PIDs. We filter to ensure we only kill our own processes if needed,
    # but strictly speaking 'kill' only works on own processes anyway.
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $target_gpu)
    
    if [ -n "$pids" ]; then
        # Convert newlines to spaces
        pids=$(echo $pids | tr '\n' ' ')
        echo "Found lingering processes on GPU $target_gpu: $pids. Killing..."
        kill -9 $pids 2>/dev/null || true
    fi
    
    # Run python GC just in case
    $PYTHON_EXEC -c "import gc; import torch; torch.cuda.empty_cache()" > /dev/null 2>&1 || true
    echo "Cleanup on GPU $target_gpu complete."
}

echo "Running all tasks"



PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# Task: ad
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types ad \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Ad task completed successfully"
cleanup_gpu $GPU_ID
# Task: product
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types product \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Product task completed successfully"
cleanup_gpu $GPU_ID
# Task: video
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types video \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Video task completed successfully"
cleanup_gpu $GPU_ID
######
# Task: rec_reason
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types rec_reason \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Rec_reason task completed successfully"
cleanup_gpu $GPU_ID
# Task: item_understand
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types item_understand \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Item_understand task completed successfully"
cleanup_gpu $GPU_ID

# Task: label_cond
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types label_cond \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Label_cond task completed successfully"
cleanup_gpu $GPU_ID

# Task: interactive
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types interactive \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Interactive task completed successfully"
cleanup_gpu $GPU_ID
# Task: label_pred
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids $GPU_ID \
    --task_types label_pred \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 3200 \
    --max_logprobs 10000 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Label_pred task completed successfully"
cleanup_gpu $GPU_ID
echo "All tasks completed successfully"
