import os
import subprocess
import json
import re
import sys

# Configuration
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
# ALPHAS = [0.5] # Test run
MODEL_PATH = "../checkpoints/OneRec-1.7B"
SUFFIX_BASE = "results_1.7B"
ENABLE_THINKING = "true"
DATA_DIR = "../raw_data/onerec_data/benchmark_data_1000"
GPU_IDS = "0 1 2 3 4 5 6 7"

SCRIPT_PATH = "./eval_script_rag4_run_5090_alpha.sh"

def run_tuning():
    results = {}
    
    print(f"Starting Alpha Tuning with alphas: {ALPHAS}")
    
    for alpha in ALPHAS:
        print(f"\n{'='*50}")
        print(f"Testing Alpha: {alpha}")
        print(f"{ '='*50}\n")
        
        # Unique version/suffix for this run to avoid overwriting
        version = f"v1.0_tuning_alpha_{alpha}"
        suffix = f"{SUFFIX_BASE}_alpha_{alpha}"
        
        # Set environment variables
        env = os.environ.copy()
        env["CD_ALPHA"] = str(alpha)
        env["VERSION"] = version
        
        # Construct command
        # bash eval_script... MODEL_PATH SUFFIX ENABLE_THINKING DATA_DIR GPU_IDS
        cmd = [
            "bash", SCRIPT_PATH,
            MODEL_PATH,
            suffix,
            ENABLE_THINKING,
            DATA_DIR,
            GPU_IDS
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the shell script
            subprocess.run(cmd, env=env, check=True)
            
            # Locate result file
            # Path logic from shell script:
            # BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/result0210/${VERSION}/results_${2}_${HOSTNAME}/$(basename "${MODEL_PATH}")"
            # Assuming BENCHMARK_BASE_DIR is "."
            hostname = subprocess.check_output("hostname", shell=True).decode().strip()
            model_basename = os.path.basename(MODEL_PATH)
            result_dir = f"./results/result0210/{version}/results_{suffix}_{hostname}/{model_basename}"
            eval_results_path = os.path.join(result_dir, "eval_results.json")
            
            print(f"Looking for results at: {eval_results_path}")
            
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'r') as f:
                    data = json.load(f)
                
                # Extract metric. Assuming 'ad' task.
                # Structure: {"OneRec-1.7B": {"ad": {"test": {"NDCG@5": ..., "HR@5": ...}}}}
                # Need to find the key dynamically or assume 'ad'
                
                # Find model key (might be absolute path or basename)
                model_keys = list(data.keys())
                if not model_keys:
                    print("Empty results file")
                    continue
                
                model_key = model_keys[0] # Should be OneRec-1.7B or similar
                
                task_metrics = data[model_key].get("ad", {}).get("test", {})
                
                # Pick a primary metric, e.g., NDCG@5 or HR@5
                # Let's print all found keys to help debug
                print(f"Found metrics: {task_metrics.keys()}")
                
                primary_metric = "NDCG@5" # Adjust if needed
                score = task_metrics.get(primary_metric, 0.0)
                
                print(f"Alpha {alpha} -> {primary_metric}: {score}")
                results[alpha] = score
                
            else:
                print(f"Result file not found: {eval_results_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running for alpha {alpha}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\n\n" + "="*50)
    print("Tuning Completed")
    print("="*50)
    
    if results:
        print("\nResults:")
        best_alpha = None
        best_score = -1.0
        
        for alpha, score in results.items():
            print(f"Alpha: {alpha}, Score: {score}")
            if score > best_score:
                best_score = score
                best_alpha = alpha
                
        print(f"\nBest Alpha: {best_alpha} (Score: {best_score})")
    else:
        print("No results collected.")

if __name__ == "__main__":
    run_tuning()
