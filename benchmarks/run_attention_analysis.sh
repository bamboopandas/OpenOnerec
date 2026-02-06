#!/bin/bash
# Attention Analysis Experiment Script

# Environment Setup
export PYTHONPATH=".:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_EXEC="/home/lkzhang/miniconda3/envs/openonerec/bin/python3"

# Configurations
MODEL_PATH="../checkpoints/OneRec-1.7B"
DATA_DIR="../raw_data/onerec_data/benchmark_data_1000"
OUTPUT_DIR="results/attention_analysis_exp"
TASK_NAME="ad"
GPU_IDS="4" # As requested/used in the original script

echo "Starting Attention Analysis Experiment..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_IDS"

# Run Analysis
$PYTHON_EXEC analyze_attention_exp.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task_name "$TASK_NAME" \
    --gpu_ids "$GPU_IDS"

echo "Analysis Completed. Results saved to $OUTPUT_DIR"
