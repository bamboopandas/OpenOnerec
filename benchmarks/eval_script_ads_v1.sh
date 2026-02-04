#!/bin/bash
# Mirror of eval_script_rag4.sh using ADS generator
# Usage: bash eval_script_ads.sh <model_path> <result_name> <enable_thinking> <custom_data_dir>
# bash benchmarks/eval_script_ads.sh ../checkpoints/OneRec-1.7B results_ads_1.7B true ../raw_data/onerec_data/benchmark_data_1000
# bash eval_script_ads_v1.sh ../checkpoints/OneRec-1.7B results_ads_1.7B true ../raw_data/onerec_data/benchmark_data_1000

# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0_1000_ads_grounding_v1ad}"
ENABLE_THINKING=${3:-true}  # ADS is most effective with thinking enabled
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
    echo "========== Task Configuration (ADS) =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "Generator Type: ads"
    echo "ADS Top-K: 1"
    echo "========================================="
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"

# Ensure benchmark package is in path
export PYTHONPATH=".:$PYTHONPATH"
# Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running all tasks with ADS Generator"

PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# Common ADS arguments
ADS_ARGS="--generator_type ads --ads_top_k 1 --sample_size 1000"

# Task: ad
$PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
    --num_gpus 1 \
    --gpu_ids 1 \
    --task_types ad \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 --max_model_len 8192 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
echo "Ad task completed successfully"

# # Task: product
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types product \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Product task completed successfully"

# # Task: video
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types video \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Video task completed successfully"

# # Task: rec_reason
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types rec_reason \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Rec_reason task completed successfully"

# # Task: item_understand
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types item_understand \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Item_understand task completed successfully"

# # Task: label_cond
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types label_cond \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Label_cond task completed successfully"

# # Task: interactive
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
#     --task_types interactive \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 --max_model_len 8192 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Interactive task completed successfully"

# # Task: label_pred
# $PYTHON_EXEC -u scripts/ray-vllm/evaluate.py \
#     --num_gpus 1 \
#     --gpu_ids 1 \
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
#     $ADS_ARGS $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1
# echo "Label_pred task completed successfully"

echo "All ADS grounded tasks completed successfully"
