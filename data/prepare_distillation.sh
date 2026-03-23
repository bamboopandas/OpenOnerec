#!/bin/bash
# Data sampling script: Sample specified number of samples from general dataset for on-policy distillation

set -e

# Configuration
INPUT_PATH="../raw_data/general_text/sft"
OUTPUT_FILE="../output/onpolicy_distillation.parquet"
TEMP_FILE="../output/onpolicy_distillation_temp.parquet"
NUM_SAMPLES=200000
SEED=42
ENGINE="pyarrow"

# Check if paths exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${INPUT_PATH}" ]; then
    echo "Error: Input path does not exist: ${INPUT_PATH}"
    exit 1
fi

# Step 1: Sample data
echo "Step 1: Sampling data..."
python3 "${SCRIPT_DIR}/scripts/sample_data.py" \
    --input "${INPUT_PATH}" \
    --output "${TEMP_FILE}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --engine "${ENGINE}"

# Step 2: Fix unicode encoding
echo ""
echo "Step 2: Fixing unicode encoding..."
python3 "${SCRIPT_DIR}/scripts/parquet_unicode_fix.py" \
    --input "${TEMP_FILE}" \
    --output "${OUTPUT_FILE}" \
    --engine "${ENGINE}"

# Clean up temporary files
if [ -f "${TEMP_FILE}" ]; then
    rm "${TEMP_FILE}"
    echo "Temporary files cleaned up"
fi

echo ""
echo "Processing completed! Output file: ${OUTPUT_FILE}"

