#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?usage: bash eval_script_thinking_lead.sh <model_path> <result_name> <task> [stage1_mode] [candidate_budget] [sample_size] [max_new_thinking_tokens]}
RESULT_NAME=${2:?usage: bash eval_script_thinking_lead.sh <model_path> <result_name> <task> [stage1_mode] [candidate_budget] [sample_size] [max_new_thinking_tokens]}
TASK_NAME=${3:?usage: bash eval_script_thinking_lead.sh <model_path> <result_name> <task> [stage1_mode] [candidate_budget] [sample_size] [max_new_thinking_tokens]}
STAGE1_MODE=${4:-lead}
CANDIDATE_BUDGET=${5:-32}
SAMPLE_SIZE=${6:-2000}
MAX_NEW_THINKING_TOKENS=${7:-}

BENCHMARK_BASE_DIR=${BENCHMARK_BASE_DIR:-"./benchmarks"}
BENCHMARK_DATA_DIR=${BENCHMARK_DATA_DIR:-"./raw_data/onerec_data/benchmark_data_id_v4_cleandata"}
OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/thinking_lead/${RESULT_NAME}/${TASK_NAME}/${STAGE1_MODE}_budget${CANDIDATE_BUDGET}"

cd "${BENCHMARK_BASE_DIR}/.."

python_args=(
  benchmarks/scripts/evaluate_thinking_lead.py
  --model_path "${MODEL_PATH}"
  --task_types "${TASK_NAME}"
  --data_dir "${BENCHMARK_DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --sample_size "${SAMPLE_SIZE}"
  --stage1_decode_mode "${STAGE1_MODE}"
  --candidate_budget "${CANDIDATE_BUDGET}"
  --overwrite
)

if [[ -n "${MAX_NEW_THINKING_TOKENS}" ]]; then
  python_args+=(--max_new_thinking_tokens "${MAX_NEW_THINKING_TOKENS}")
fi

python "${python_args[@]}"
