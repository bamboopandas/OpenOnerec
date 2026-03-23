#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/output/beauty_proxy"}
export VAL_SPLIT=${VAL_SPLIT:-"val"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/${VAL_SPLIT}.parquet]"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_DIR/output/beauty_proxy_runs/${VAL_SPLIT}"}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"$OUTPUT_DIR/validation_dumps"}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-10}
export ENABLE_THINK=${ENABLE_THINK:-True}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-True}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-1024}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-768}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}
export REWARD_MODE=${REWARD_MODE:-objective_rubric}
export RUBRIC_DIR=${RUBRIC_DIR:-"$SCRIPT_DIR/rubrics/beauty_proxy"}
export SIDECAR_INDEX_PATH=${SIDECAR_INDEX_PATH:-"$DATA_DIR/sidecar_index.parquet"}
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_Beauty_Proxy"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"beauty_proxy_${REWARD_MODE}_${VAL_SPLIT}"}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-25}

if [ "$REWARD_MODE" = "objective_only" ]; then
    export JUDGE_BASE_URL=none
    export JUDGE_MODEL=none
fi

echo "==================================="
echo "Beauty proxy GRPO"
echo "==================================="
echo "Data dir: $DATA_DIR"
echo "Validation split: $VAL_SPLIT"
echo "Reward mode: $REWARD_MODE"
echo "Validation dumps: $VALIDATION_DATA_DIR"
echo "==================================="

bash "$SCRIPT_DIR/run_grpo.sh" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    "$@"
