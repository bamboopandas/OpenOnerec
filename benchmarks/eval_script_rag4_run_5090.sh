#!/bin/bash
# bash eval_script_rag4_run_5090.sh ../checkpoints/OneRec-1.7B results_1.7B true ../raw_data/onerec_data/benchmark_data_1000 "0 1 2 3 4 5 6 7"

export no_proxy="localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12,$(hostname -I | tr ' ' ',')"
export NO_PROXY="$no_proxy"
export VLLM_ATTENTION_BACKEND=FLASHINFER


# Set common variables
MODEL_PATH=$1
# VERSION="${VERSION:-a}"
VERSION="${VERSION:-aaaaaa}"
# VERSION="${VERSION:-v1.0_1000_5090_chazhi}"
ENABLE_THINKING=$3
CUSTOM_DATA_DIR=$4

# Handle multiple GPUs (space separated)
GPU_IDS="${@:5}"
if [ -z "$GPU_IDS" ]; then
    GPU_IDS="1"
fi
NUM_GPUS=$(echo $GPU_IDS | wc -w)

echo "Using GPUs: $GPU_IDS (Total: $NUM_GPUS)"

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-.}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

# BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
# BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}/$(basename "${MODEL_PATH}")"

BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/result0210/${VERSION}/results_${2}_${HOSTNAME}/$(basename "${MODEL_PATH}")"


BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/${2}_${HOSTNAME}"

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
    echo "GPU IDs: $GPU_IDS"
    echo "Number of GPUs: $NUM_GPUS"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"

export PYTHONPATH="${BENCHMARK_BASE_DIR}:$PYTHONPATH"

# Function to force cleanup processes on the specific GPUs
cleanup_gpu() {
    local gpu_list=$1
    for target_gpu in $gpu_list; do
        echo "Cleaning up processes on GPU $target_gpu..."
        
        # Get PIDs running on this GPU
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $target_gpu 2>/dev/null)
        
        if [ -n "$pids" ]; then
            # Convert newlines to spaces
            pids=$(echo $pids | tr '\n' ' ')
            echo "Found lingering processes on GPU $target_gpu: $pids. Killing..."
            kill -9 $pids 2>/dev/null || true
        fi
    done
    
    # Run python GC just in case
    $PYTHON_EXEC -c "import gc; import torch; torch.cuda.empty_cache()" > /dev/null 2>&1 || true
    echo "Cleanup on all specified GPUs complete."
}

echo "Running all tasks"



PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# Task: ad
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus $NUM_GPUS \
    --gpu_ids $GPU_IDS \
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
cleanup_gpu "$GPU_IDS"

# Task: product
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus $NUM_GPUS \
    --gpu_ids $GPU_IDS \
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
cleanup_gpu "$GPU_IDS"

# Task: video
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus $NUM_GPUS \
    --gpu_ids $GPU_IDS \
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
cleanup_gpu "$GPU_IDS"

# ######
# # Task: rec_reason
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus $NUM_GPUS \
#     --gpu_ids $GPU_IDS \
#     --task_types rec_reason \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Rec_reason task completed successfully"
# cleanup_gpu "$GPU_IDS"
# # Task: item_understand
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus $NUM_GPUS \
#     --gpu_ids $GPU_IDS \
#     --task_types item_understand \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Item_understand task completed successfully"
# cleanup_gpu "$GPU_IDS"

# # Task: label_cond
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus $NUM_GPUS \
#     --gpu_ids $GPU_IDS \
#     --task_types label_cond \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Label_cond task completed successfully"
# cleanup_gpu "$GPU_IDS"

# # Task: interactive
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus $NUM_GPUS \
#     --gpu_ids $GPU_IDS \
#     --task_types interactive \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Interactive task completed successfully"
# cleanup_gpu "$GPU_IDS"
# # Task: label_pred
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus $NUM_GPUS \
#     --gpu_ids $GPU_IDS \
#     --task_types label_pred \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 3200 \
#     --max_logprobs 10000 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Label_pred task completed successfully"
# cleanup_gpu "$GPU_IDS"
echo "All tasks completed successfully"
