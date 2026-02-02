#!/bin/bash

# ================= Configuration =================
# These must match run_pretrain_stg1_local.sh
BASE_MODEL_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/data/code/onerec_pretrain/hf_models/Qwen3-0.6B_itemic"
STG1_OUTPUT_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/stg1_opt"
TRAINING_STEPS=2000

# Script paths
CONVERT_SCRIPT="scripts/convert_checkpoint_to_hf.sh"
STG2_SCRIPT="run_pretrain_stg2_local.sh"

# Process pattern to monitor (looks for the python script and the output dir)
PROCESS_PATTERN="recipes/train_qwen3.py.*${STG1_OUTPUT_DIR}"
# =============================================

echo "🔍 Starting monitor for Stage 1 training..."
echo "   Target Output: ${STG1_OUTPUT_DIR}"
echo "   Expected Steps: ${TRAINING_STEPS}"

while true; do
    # Check if the process is running
    if pgrep -f "$PROCESS_PATTERN" > /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 is still running. Waiting..."
        sleep 60
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 process not found. Checking for completion..."
        
        # Check if the final checkpoint exists
        EXPECTED_CKPT="${STG1_OUTPUT_DIR}/step${TRAINING_STEPS}/global_step${TRAINING_STEPS}"
        
        if [ -d "$EXPECTED_CKPT" ]; then
            echo "✅ Stage 1 completed successfully! (Checkpoint found: $EXPECTED_CKPT)"
            break
        else
            echo "❌ Stage 1 finished but checkpoint not found at $EXPECTED_CKPT"
            echo "   Please check the logs in ${STG1_OUTPUT_DIR}"
            exit 1
        fi
    fi
done

# ================= Conversion =================
echo "🔄 Starting Checkpoint Conversion..."
cd "$(dirname "$0")" # Ensure we are in the pretrain/ directory

if [ -f "$CONVERT_SCRIPT" ]; then
    bash "$CONVERT_SCRIPT" "$BASE_MODEL_DIR" "$STG1_OUTPUT_DIR" "$TRAINING_STEPS"
    
    if [ $? -eq 0 ]; then
        echo "✅ Conversion successful."
    else
        echo "❌ Conversion failed. Exiting."
        exit 1
    fi
else
    echo "❌ Conversion script not found: $CONVERT_SCRIPT"
    exit 1
fi

# ================= Run Stage 2 =================
echo "🚀 Starting Stage 2..."
if [ -f "$STG2_SCRIPT" ]; then
    bash "$STG2_SCRIPT"
else
    echo "❌ Stage 2 script not found: $STG2_SCRIPT"
    exit 1
fi
