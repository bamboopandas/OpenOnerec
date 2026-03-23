#!/bin/bash
# RL data splitting script: Merge multiple RL task datasets and split into training and test sets

set -e

# Configuration
# onerec dataset output path, rl uses datasets starting with sft
REC_DATA_PATH="../output"

# Tasks that RL depends on
VIDEO_REC=${REC_DATA_PATH}/sft_video_rec.parquet
AD_REC=${REC_DATA_PATH}/sft_ad_rec.parquet
PRODUCT_REC=${REC_DATA_PATH}/sft_product_rec.parquet
INTERACTIVE_REC=${REC_DATA_PATH}/sft_interactive_rec.parquet
LABEL_COND_REC=${REC_DATA_PATH}/sft_label_cond_rec.parquet

# Output configuration
OUTPUT_DIR="../output/rl_data"
TEST_SIZE=1000
SEED=42
ENGINE="pyarrow"
HISTORY_LIMIT=20
CAPTION_MAX_CHARS=96

# Optional sidecar inputs. Space-separated parquet paths.
SID_MAPPING_FILES=${SID_MAPPING_FILES:-""}
CAPTION_FILES=${CAPTION_FILES:-""}
SIDECAR_INDEX_OUT=${SIDECAR_INDEX_OUT:-"sidecar_index.parquet"}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define all task files to process
declare -a TASK_FILES=(
    "${VIDEO_REC}"
    "${AD_REC}"
    "${PRODUCT_REC}"
    "${INTERACTIVE_REC}"
    "${LABEL_COND_REC}"
)

# Check if input files exist
echo "Checking input files..."
MISSING_FILES=0
for file in "${TASK_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Warning: File does not exist: ${file}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ ${MISSING_FILES} -eq ${#TASK_FILES[@]} ]; then
    echo "Error: All input files do not exist"
    exit 1
fi

# Execute train_test_split, merge all files and process them together
echo ""
echo "Starting RL data splitting..."
echo "=========================================="
echo "Input files:"
for file in "${TASK_FILES[@]}"; do
    if [ -f "${file}" ]; then
        echo "  - ${file}"
    fi
done
echo "Output directory: ${OUTPUT_DIR}"
echo "Test set size: ${TEST_SIZE}"
echo "=========================================="

declare -a MAPPING_ARGS=()
declare -a CAPTION_ARGS=()

if [ -n "${SID_MAPPING_FILES}" ]; then
    read -r -a MAPPING_FILE_ARRAY <<< "${SID_MAPPING_FILES}"
    MAPPING_ARGS=(--mapping_files "${MAPPING_FILE_ARRAY[@]}")
fi

if [ -n "${CAPTION_FILES}" ]; then
    read -r -a CAPTION_FILE_ARRAY <<< "${CAPTION_FILES}"
    CAPTION_ARGS=(--caption_files "${CAPTION_FILE_ARRAY[@]}")
fi

python3 "${SCRIPT_DIR}/scripts/prepare_rubric_rl_data.py" \
    --input_files "${TASK_FILES[@]}" \
    --test_size "${TEST_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --engine "${ENGINE}" \
    --test_filename "test.parquet" \
    --train_filename "train.parquet" \
    --sidecar_index_out "${SIDECAR_INDEX_OUT}" \
    --history_limit "${HISTORY_LIMIT}" \
    --caption_max_chars "${CAPTION_MAX_CHARS}" \
    "${MAPPING_ARGS[@]}" \
    "${CAPTION_ARGS[@]}"

echo ""
echo "=========================================="
echo "RL data processing completed!"
echo "Output directory: ${OUTPUT_DIR}"
echo "  - train.parquet (training set)"
echo "  - test.parquet (test set)"
echo "  - ${SIDECAR_INDEX_OUT} (SID semantic sidecar)"
echo "=========================================="
