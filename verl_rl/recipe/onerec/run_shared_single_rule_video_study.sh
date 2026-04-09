#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
BENCHMARK_ROOT="${REPO_ROOT}/benchmarks"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/raw_data/onerec_data}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${RAW_DATA_ROOT}/benchmark_data_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/shared_single_rule_video_study}"
MODEL_PATH="${MODEL_PATH:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
HISTORY_SUMMARY_MODEL_PATH="${HISTORY_SUMMARY_MODEL_PATH:-}"
GEN_GPU_ID="${GEN_GPU_ID:-0}"
RERANK_GPU_ID="${RERANK_GPU_ID:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
NUM_SPLITS="${NUM_SPLITS:-2}"
TEST_SHARD_ID="${TEST_SHARD_ID:-1}"
PAIRWISE_TOP_N="${PAIRWISE_TOP_N:-4}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
WORKER_BATCH_SIZE="${WORKER_BATCH_SIZE:-256}"
TASK_NAME="video"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required" >&2
    exit 1
fi
if [[ -z "${JUDGE_MODEL_PATH}" ]]; then
    echo "JUDGE_MODEL_PATH is required" >&2
    exit 1
fi

MODEL_NAME=$(basename "${MODEL_PATH}")
RUN_ROOT="${OUTPUT_ROOT}/${TASK_NAME}_${MODEL_NAME}"
GEN_OUTPUT_DIR="${RUN_ROOT}/generation"
STUDY_ROOT="${RUN_ROOT}/shared_single_rule"
RAW_TEST_DIR="${STUDY_ROOT}/raw_test"
SHARED_TEST_DIR="${STUDY_ROOT}/shared_test"
COMPARE_SUMMARY_PATH="${STUDY_ROOT}/compare_summary.json"
GENERATION_FILE="${GEN_OUTPUT_DIR}/${MODEL_NAME}/${TASK_NAME}/test_generated.json"
TASK_DATA_FILE="${BENCHMARK_DATA_DIR}/${TASK_NAME}/${TASK_NAME}_test.parquet"
CAPTION_FILE="${RAW_DATA_ROOT}/pid2caption.parquet"
GEN_DATA_DIR="${BENCHMARK_DATA_DIR}"
MINI_DATA_DIR="${RUN_ROOT}/benchmark_subset"

mkdir -p "${GEN_OUTPUT_DIR}" "${RAW_TEST_DIR}" "${SHARED_TEST_DIR}"

cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${REPO_ROOT}/verl_rl:${PYTHONPATH:-}"

mkdir -p "${MINI_DATA_DIR}/${TASK_NAME}"
python - <<'PY' "${TASK_DATA_FILE}" "${MINI_DATA_DIR}/${TASK_NAME}/${TASK_NAME}_test.parquet" "${MAX_SAMPLES}"
import sys
from pathlib import Path
import pandas as pd

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
max_samples = int(sys.argv[3])

df = pd.read_parquet(source_path)
df = df.iloc[:max_samples].copy()
target_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(target_path, index=False)
print(target_path)
PY
for aux_name in sid2pid.json sid2iid.json; do
    if [[ -f "${BENCHMARK_DATA_DIR}/${aux_name}" ]]; then
        cp "${BENCHMARK_DATA_DIR}/${aux_name}" "${MINI_DATA_DIR}/${aux_name}"
    fi
done
GEN_DATA_DIR="${MINI_DATA_DIR}"

if [[ ! -f "${GENERATION_FILE}" ]]; then
    echo "Running generation for ${TASK_NAME} -> ${GENERATION_FILE}"
    python -u scripts/ray-vllm/evaluate.py \
        --model_path "${MODEL_PATH}" \
        --task_types "${TASK_NAME}" \
        --splits test \
        --data_dir "${GEN_DATA_DIR}" \
        --output_dir "${GEN_OUTPUT_DIR}" \
        --overwrite \
        --num_gpus 1 \
        --gpu_ids "${GEN_GPU_ID}" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --worker_batch_size "${WORKER_BATCH_SIZE}" \
        --dtype bfloat16 \
        --num_beams 32 \
        --num_return_sequences 32 \
        --num_return_thinking_sequences 1
fi

echo "Building raw test summary"
CUDA_VISIBLE_DEVICES="${RERANK_GPU_ID}" \
python -u -m recipe.onerec.offline_openonerec_benchmark_eval \
    --generation_file "${GENERATION_FILE}" \
    --task_name "${TASK_NAME}" \
    --task_data_file "${TASK_DATA_FILE}" \
    --output_dir "${RAW_TEST_DIR}" \
    --rubric_dir "${SCRIPT_DIR}/rubrics" \
    --judge_model_path none \
    --sidecar_index_path "${RAW_TEST_DIR}/sidecar_index.parquet" \
    --mapping_files "${RAW_DATA_ROOT}/video_ad_pid2sid.parquet" "${RAW_DATA_ROOT}/product_pid2sid.parquet" \
    --caption_files "${CAPTION_FILE}" \
    --max_samples "${MAX_SAMPLES}" \
    --k 32 \
    --pass_ks 1 5 10 32 \
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
    --pairwise_top_n 0 \
    --pairwise_raw_rank_anchor 0.0 \
    --shard_id "${TEST_SHARD_ID}" \
    --num_shards "${NUM_SPLITS}" \
    > "${RAW_TEST_DIR}/run.log" 2>&1

echo "Running shared single-rule rerank"
CUDA_VISIBLE_DEVICES="${RERANK_GPU_ID}" \
python -u -m recipe.onerec.offline_openonerec_benchmark_eval \
    --generation_file "${GENERATION_FILE}" \
    --task_name "${TASK_NAME}" \
    --task_data_file "${TASK_DATA_FILE}" \
    --output_dir "${SHARED_TEST_DIR}" \
    --rubric_dir "${SCRIPT_DIR}/rubrics" \
    --judge_model_path "${JUDGE_MODEL_PATH}" \
    --history_summary_model_path "${HISTORY_SUMMARY_MODEL_PATH}" \
    --sidecar_index_path "${SHARED_TEST_DIR}/sidecar_index.parquet" \
    --mapping_files "${RAW_DATA_ROOT}/video_ad_pid2sid.parquet" "${RAW_DATA_ROOT}/product_pid2sid.parquet" \
    --caption_files "${CAPTION_FILE}" \
    --max_samples "${MAX_SAMPLES}" \
    --k 32 \
    --pass_ks 1 5 10 32 \
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
    --pairwise_top_n "${PAIRWISE_TOP_N}" \
    --pairwise_raw_rank_anchor 0.0 \
    --pairwise_judge_mode single_rule_audit \
    --shard_id "${TEST_SHARD_ID}" \
    --num_shards "${NUM_SPLITS}" \
    > "${SHARED_TEST_DIR}/run.log" 2>&1

python - <<'PY' "${RAW_TEST_DIR}/summary.json" "${SHARED_TEST_DIR}/summary.json" "${COMPARE_SUMMARY_PATH}"
import json
import sys
from pathlib import Path

raw_summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shared_summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
output_path = Path(sys.argv[3])

compare = {
    "raw_ranking": raw_summary.get("raw_ranking", {}),
    "shared_rules_single_rule_rerank": {
        "raw_ranking": shared_summary.get("raw_ranking", {}),
        "rubric_rerank": shared_summary.get("rubric_rerank", {}),
        "delta": shared_summary.get("delta", {}),
        "rule_completion_rate": shared_summary.get("rule_completion_rate", 0.0),
        "retry_rate": shared_summary.get("retry_rate", 0.0),
        "forced_tie_rate": shared_summary.get("forced_tie_rate", 0.0),
        "swap_consistency_rate": shared_summary.get("swap_consistency_rate", 0.0),
        "history_summary_nonempty_rate": shared_summary.get("history_summary_nonempty_rate", 0.0),
    },
}
output_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(compare, ensure_ascii=False, indent=2))
PY
