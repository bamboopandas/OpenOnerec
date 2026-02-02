import sys
import os

# Add benchmarks directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import Benchmark

# Configuration
GENERATION_RESULTS_DIR = "benchmarks/results/v3.0_false_RAG/results_results_1.7B"
OUTPUT_PATH = f"{GENERATION_RESULTS_DIR}/eval_results.json"
DATA_DIR = "raw_data/onerec_data/benchmark_data_rag"
TASK_TYPES = ["product"]

print(f"Starting manual evaluation...")
print(f"Results Dir: {GENERATION_RESULTS_DIR}")
print(f"Data Dir: {DATA_DIR}")

# Run evaluation
Benchmark.evaluate_dev(
    generation_results_dir=GENERATION_RESULTS_DIR,
    output_path=OUTPUT_PATH,
    data_dir=DATA_DIR,
    overwrite=True, # We want to overwrite the empty/failed results file
    task_types=TASK_TYPES
)

print("Evaluation complete.")
