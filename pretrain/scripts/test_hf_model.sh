#!/bin/bash

# HuggingFace Model Testing Script
# Tests a HuggingFace model with text generation or chat mode
# 
# Configuration:
#   - MODEL_PATH: Path to HuggingFace model directory
#   - TEST_FILE: Path to JSON test cases file (optional, use --use_default if not set)
#   - Generation parameters: MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
#   - Chat options: ENABLE_THINKING, SHOW_TEMPLATE, SHOW_INPUT_IDS
#   - Output: COMPARE_GROUND_TRUTH

set -e

# Model path - receive from command line argument
MODEL_PATH="$1"

# Check if MODEL_PATH is empty
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH cannot be empty"
    echo "Usage: $0 <MODEL_PATH>"
    echo "Example: $0 /path/to/model"
    exit 1
fi

# Check if model path exists
if [ ! -e "${MODEL_PATH}" ]; then
    echo "WARNING: model path does not exist: ${MODEL_PATH}"
    exit 1
fi

# Test case: use default or specify a test file
# Option 1: Use built-in default test cases
USE_DEFAULT=true

# Option 2: Use custom test file (comment out USE_DEFAULT and uncomment below)
# USE_DEFAULT=false
# TEST_FILE=tools/model_test/test_cases_example.json

# Generation parameters
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.9
REPETITION_PENALTY=1.2

# Chat mode options
ENABLE_THINKING=false
SHOW_TEMPLATE=false
SHOW_INPUT_IDS=false

# Output options
COMPARE_GROUND_TRUTH=false

# Device and data type
DEVICE=auto
DTYPE=bf16

# Build command
CMD="python3 tools/model_test/test_hf_model.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --device $DEVICE"
CMD="$CMD --dtype $DTYPE"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --repetition_penalty $REPETITION_PENALTY"

# Test case source
if [ "$USE_DEFAULT" = true ]; then
    CMD="$CMD --use_default"
elif [ -n "$TEST_FILE" ]; then
    CMD="$CMD --test_file $TEST_FILE"
fi

# Chat mode options
[ "$ENABLE_THINKING" = true ] && CMD="$CMD --enable_thinking"
[ "$SHOW_TEMPLATE" = true ] && CMD="$CMD --show_template"
[ "$SHOW_INPUT_IDS" = true ] && CMD="$CMD --show_input_ids"

# Output options
[ "$COMPARE_GROUND_TRUTH" = true ] && CMD="$CMD --compare_ground_truth"

# Execute
eval $CMD


