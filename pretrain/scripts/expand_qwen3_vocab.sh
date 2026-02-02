#!/bin/bash

set -e

HF_MODEL_DIR=/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/data/code/onerec_pretrain/hf_models/Qwen3-0.6B
OUTPUT_MODEL_DIR=/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/data/code/onerec_pretrain/hf_models/Qwen3-0.6B_itemic
ITEMIC_LAYER_N=3
VOCAB_SIZE_PER_LAYER=8192

python3 /zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/pretrain/tools/model_converter/expand_qwen3_vocab.py \
    --hf_model_dir $HF_MODEL_DIR \
    --output_model_dir $OUTPUT_MODEL_DIR \
    --itemic_layer_n $ITEMIC_LAYER_N \
    --vocab_size_per_layer $VOCAB_SIZE_PER_LAYER


