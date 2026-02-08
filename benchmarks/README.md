# Benchmark

---

## Quick Start

### Requirements

- **Python**: >= 3.10
- **CUDA**: GPU environment with CUDA support
- **Hardware**: Multi-GPU setup recommended for faster evaluation

**Main Dependencies**:
- PyTorch 2.5.1
- Transformers 4.52.0
- Ray 2.43.0
- vLLM 0.7.3

### Step 1: Prepare Evaluation Data

Place the evaluation data in the project directory:

```bash
# Data directory structure
./data_v1.0/
├── rec_reason/
├── item_understand/
├── ad/
├── product/
├── label_cond/
├── video/
├── interactive/
└── label_pred/
```

### Step 2: Install Dependencies

```bash
cd benchmarks

pip install -e . --no-deps --no-build-isolation

```

### Step 3: Start Ray Cluster

```bash
# Initialize multi-node multi-GPU environment
bash scripts/init_ray_cluster.sh
```

### Step 4: Run Evaluation

```bash
export BENCHMARK_BASE_DIR="./benchmarks"
export BENCHMARK_DATA_DIR="./raw_data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"

bash eval_script.sh <model_path> <result_name> <enable_thinking>
```

**Parameters**:
| Parameter | Description | Example |
|-----------|-------------|---------|
| model_path | Path to the model to evaluate | `model_output/sft/global_step10/converted` |
| result_name | Name identifier for output directory | `sft_nonthink` |
| enable_thinking | `true` or `false` | `false` |

**Examples**:
```bash
# Without thinking mode
bash eval_script.sh \
    /path/to/model \
    model_nonthink \
    false

# With thinking mode
bash eval_script.sh \
    /path/to/model \
    model_think \
    true
```

### Step 5: View Results

After evaluation completes, results are saved in:
```
./benchmarks/results/v1.0/results_<result_name>/
```

Log files are located at:
```
./benchmarks/auto_eval_logs/v1.0/<result_name>.log
```


---

## Evaluation Tasks

| Task Name | Source | Description |
|-----------|--------|-------------|
| ad | Kuaishou Internal | 27,677 | Predict next clicked advertisement |
| product | Kuaishou Internal | 27,910 | Predict next clicked product |
| interactive | Kuaishou Internal | 1,000 | Predict next interacted video |
| video | Kuaishou Internal | 38,781  | Next video prediction |
| label_cond | Kuaishou Internal | 34,891 | Predict next video given specified consumption behavior |
| label_pred | Kuaishou Internal | 346,190 | Predict user engagement with video content |
| item_understand | Kuaishou Internal | 500 | Video SID to Caption generation task |
| rec_reason | Kuaishou Internal | 470 | Recommendation reason inference |


---

## Inference-Time Alignment (Training-Free)

We provide a standalone script for inference-time re-alignment using ASP-SCP (Attribution Selection & Structured Context), TPD (Text-Prior Direction), PDC/CDC (Drift Correction), and Uncertainty Gating (UG).

### Usage

```bash
python benchmarks/run_inference_alignment.py \
    --data_path <path_to_parquet_data> \
    --model_path checkpoints/OneRec-1.7B \
    --output_path alignment_results.jsonl \
    --num_samples 100 \
    --top_k 5 \
    --alpha 0.5 \
    --beta0 0.0 \
    --beta1 1.0
```

### Outputs

The script produces a JSONL file where each line corresponds to a test instance and contains:
- `no_think_topk`: Baseline recommendations (No-Think).
- `think_topk`: Recommendations with Chain-of-Thought (CoT).
- `pdc_topk`: Recommendations corrected via Projected Drift Correction.
- `cdc_topk`: Recommendations corrected via Contrastive Drift Correction.
- `ours_topk`: Final recommendations using Uncertainty Gating (UG).
- `metrics`: Diagnostic metrics (drift magnitude, alignment, entropy gap, etc.).
- `gt_items`: Ground truth items.

```