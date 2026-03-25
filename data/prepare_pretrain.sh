#!/bin/bash
# Data splitting script: Merge general text and recommendation data, then split by every 1000 samples

set -e

# Configuration
# Both general and onerec use datasets starting with pretrain
GENERAL_TEXT_PATH="../raw_data/general_text/pretrain"
REC_DATA_PATH="../output/onerec_pretrain"
OUTPUT_DIR="../output/split_data_pretrain"
MAX_ROWS=1000
ENGINE="pyarrow"

# Check if paths exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${GENERAL_TEXT_PATH}" ]; then
    echo "Error: General text path does not exist: ${GENERAL_TEXT_PATH}"
    exit 1
fi

if [ ! -e "${REC_DATA_PATH}" ]; then
    echo "Error: Recommendation data path does not exist: ${REC_DATA_PATH}"
    exit 1
fi

# Execute
python3 "${SCRIPT_DIR}/scripts/split_data.py" \
    --general_text_path "${GENERAL_TEXT_PATH}" \
    --rec_data_path "${REC_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_rows "${MAX_ROWS}" \
    --engine "${ENGINE}"

