#!/bin/bash

set -euo pipefail

MODEL_PATH=${1:?MODEL_PATH is required}
RUN_NAME=${2:?RUN_NAME is required}
CUSTOM_DATA_DIR=${3:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_BASE_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_EXEC="${PYTHON_EXEC:-/home/lkzhang/miniconda3/envs/openonerec_beauty_proxy/bin/python}"
DATA_DIR="${CUSTOM_DATA_DIR:-../raw_data/onerec_data/benchmark_data_id_v4_cleandata}"
SAMPLE_SIZE="${SAMPLE_SIZE:-2000}"
MODES="${MODES:-vanilla lead}"
LATENT_TOPK="${LATENT_TOPK:-64}"
PERSISTENCE_WINDOW="${PERSISTENCE_WINDOW:-3}"
MAX_SWITCHES="${MAX_SWITCHES:-5}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"

BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/lead_decoding/${RUN_NAME}"
BASE_LOG_DIR="${BENCHMARK_BASE_DIR}/auto_eval_logs/lead_decoding/${RUN_NAME}"
mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

declare -A TASK_GPU=(
  [ad]="${GPU_AD:-0}"
  [product]="${GPU_PRODUCT:-1}"
  [video]="${GPU_VIDEO:-3}"
)

run_task() {
  local mode="$1"
  local task="$2"
  local gpu_id="${TASK_GPU[$task]}"
  local output_dir="${BASE_OUTPUT_DIR}/${mode}/${task}"
  local log_file="${BASE_LOG_DIR}/${mode}_${task}.log"

  mkdir -p "$output_dir"
  echo "[$(date '+%F %T')] mode=${mode} task=${task} gpu=${gpu_id} sample_size=${SAMPLE_SIZE}" | tee "$log_file"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONPATH="${BENCHMARK_BASE_DIR}:${PYTHONPATH:-}" \
  "$PYTHON_EXEC" -u "${BENCHMARK_BASE_DIR}/scripts/evaluate_lead_decoding.py" \
    --task_types "$task" \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$output_dir" \
    --dtype bfloat16 \
    --overwrite \
    --decode_mode "$mode" \
    --latent_topk "$LATENT_TOPK" \
    --persistence_window "$PERSISTENCE_WINDOW" \
    --max_switches "$MAX_SWITCHES" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --max_new_tokens 3 \
    --sample_size "$SAMPLE_SIZE" >> "$log_file" 2>&1
}

summarize_mode() {
  local mode="$1"
  local summary_path="${BASE_OUTPUT_DIR}/${mode}/summary.json"
  PYTHONPATH="${BENCHMARK_BASE_DIR}:${PYTHONPATH:-}" "$PYTHON_EXEC" - <<'PY' "$BASE_OUTPUT_DIR" "$mode" "$summary_path"
import json
import sys
from pathlib import Path

base_output_dir = Path(sys.argv[1])
mode = sys.argv[2]
summary_path = Path(sys.argv[3])

mode_dir = base_output_dir / mode
summary = {"mode": mode, "tasks": {}}

for task in ("ad", "product", "video"):
    task_dir = mode_dir / task
    eval_path = task_dir / "eval_results.json"
    stats_path = task_dir / "decode_stats_summary.json"
    task_metrics = {}

    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        if task in eval_data:
            task_metrics.update(eval_data[task].get("overall_metrics", {}))

    if stats_path.exists():
        stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_data.get("task_name") == task:
            task_metrics.update(
                {
                    "mean_avg_entropy": stats_data.get("mean_avg_entropy"),
                    "mean_latent_step_ratio": stats_data.get("mean_latent_step_ratio"),
                    "latent_trigger_rate": stats_data.get("latent_trigger_rate"),
                    "mean_switch_count": stats_data.get("mean_switch_count"),
                    "mean_elapsed_sec": stats_data.get("mean_elapsed_sec"),
                }
            )

    summary["tasks"][task] = task_metrics

summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(summary_path)
PY
}

for mode in $MODES; do
  pids=()
  for task in ad product video; do
    run_task "$mode" "$task" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  summarize_mode "$mode" >/dev/null
done

echo "LEAD decoding batch evaluation completed. Results: $BASE_OUTPUT_DIR"
