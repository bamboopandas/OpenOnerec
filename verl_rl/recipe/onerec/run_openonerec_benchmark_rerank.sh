#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
BENCHMARK_ROOT="${REPO_ROOT}/benchmarks"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/raw_data/onerec_data}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${RAW_DATA_ROOT}/benchmark_data_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/openonerec_benchmark_rerank}"
RUBRIC_DIR="${RUBRIC_DIR:-${SCRIPT_DIR}/rubrics}"
MODEL_PATH="${MODEL_PATH:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
TASK_TYPES="${TASK_TYPES:-video ad product interactive label_cond}"
GEN_GPU_IDS="${GEN_GPU_IDS:-0,1,3,4}"
RERANK_GPU_IDS="${RERANK_GPU_IDS:-0,1,3,4}"
RERANK_NUM_SHARDS="${RERANK_NUM_SHARDS:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
WORKER_BATCH_SIZE="${WORKER_BATCH_SIZE:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-384}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required" >&2
    exit 1
fi

if [[ -z "${JUDGE_MODEL_PATH}" ]]; then
    echo "JUDGE_MODEL_PATH is required" >&2
    exit 1
fi

IFS=' ' read -r -a TASK_ARRAY <<< "${TASK_TYPES}"
IFS=',' read -r -a GEN_GPU_ARRAY <<< "${GEN_GPU_IDS}"
IFS=',' read -r -a RERANK_GPU_ARRAY <<< "${RERANK_GPU_IDS}"
NUM_GEN_GPUS="${#GEN_GPU_ARRAY[@]}"
if [[ "${RERANK_NUM_SHARDS}" == "0" ]]; then
    RERANK_NUM_SHARDS="${#RERANK_GPU_ARRAY[@]}"
fi

MODEL_NAME=$(basename "${MODEL_PATH}")
GEN_OUTPUT_DIR="${OUTPUT_ROOT}/generation"
RERANK_OUTPUT_DIR="${OUTPUT_ROOT}/rerank/${MODEL_NAME}"
SIDECAR_INDEX_PATH="${OUTPUT_ROOT}/artifacts/sidecar_index.parquet"
CAPTION_FILE="${RAW_DATA_ROOT}/pid2caption.parquet"

mkdir -p "${GEN_OUTPUT_DIR}" "${RERANK_OUTPUT_DIR}" "$(dirname "${SIDECAR_INDEX_PATH}")"

EXTRA_EVAL_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" && -n "${MAX_SAMPLES}" ]]; then
    EXTRA_EVAL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${REPO_ROOT}/verl_rl:${PYTHONPATH:-}"

echo "Running benchmark generation on tasks: ${TASK_TYPES}"
python -u scripts/ray-vllm/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --task_types "${TASK_ARRAY[@]}" \
    --splits test \
    --data_dir "${BENCHMARK_DATA_DIR}" \
    --output_dir "${GEN_OUTPUT_DIR}" \
    --overwrite \
    --num_gpus "${NUM_GEN_GPUS}" \
    --gpu_ids "${GEN_GPU_ARRAY[@]}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --worker_batch_size "${WORKER_BATCH_SIZE}" \
    --dtype bfloat16 \
    --num_beams 32 \
    --num_return_sequences 32 \
    --num_return_thinking_sequences 1

echo "Running offline rubric rerank"
for task in "${TASK_ARRAY[@]}"; do
    generation_file="${GEN_OUTPUT_DIR}/${MODEL_NAME}/${task}/test_generated.json"
    task_data_file="${BENCHMARK_DATA_DIR}/${task}/${task}_test.parquet"
    task_output_dir="${RERANK_OUTPUT_DIR}/${task}"
    mkdir -p "${task_output_dir}"

    pids=()
    shard_dirs=()
    for ((shard_id = 0; shard_id < ${RERANK_NUM_SHARDS}; ++shard_id)); do
        gpu="${RERANK_GPU_ARRAY[$((shard_id % ${#RERANK_GPU_ARRAY[@]}))]}"
        shard_output_dir="${task_output_dir}/shard_${shard_id}"
        mkdir -p "${shard_output_dir}"
        shard_dirs+=("${shard_output_dir}")

        CUDA_VISIBLE_DEVICES="${gpu}" \
        python -u -m recipe.onerec.offline_openonerec_benchmark_eval \
            --generation_file "${generation_file}" \
            --task_name "${task}" \
            --task_data_file "${task_data_file}" \
            --output_dir "${shard_output_dir}" \
            --rubric_dir "${RUBRIC_DIR}" \
            --judge_model_path "${JUDGE_MODEL_PATH}" \
            --sidecar_index_path "${shard_output_dir}/sidecar_index.parquet" \
            --mapping_files "${RAW_DATA_ROOT}/video_ad_pid2sid.parquet" "${RAW_DATA_ROOT}/product_pid2sid.parquet" \
            --caption_files "${CAPTION_FILE}" \
            --k 32 \
            --pass_ks 1 5 10 32 \
            --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
            --shard_id "${shard_id}" \
            --num_shards "${RERANK_NUM_SHARDS}" \
            "${EXTRA_EVAL_ARGS[@]}" \
            > "${shard_output_dir}/run.log" 2>&1 &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "${pid}"
    done

    python -u -m recipe.onerec.merge_openonerec_benchmark_rerank \
        --shard_dirs "${shard_dirs[@]}" \
        --output_dir "${task_output_dir}" \
        --k 32 \
        --pass_ks 1 5 10 32 \
        > "${task_output_dir}/merge.log" 2>&1
done

echo "Generation output: ${GEN_OUTPUT_DIR}/${MODEL_NAME}"
echo "Rerank summaries: ${RERANK_OUTPUT_DIR}"
