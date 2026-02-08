import json
import os
import re
import pandas as pd

# File paths
INPUT_FILE = "benchmarks/results/v1.0_1000_thinktrue_tryspeed/results_results_8B/OneRec-8B/eval_results_all.json"
# INPUT_FILE = "benchmarks/results/v1.0_1000_thinktrue_tryspeed/results_results_8B/OneRec-8B-pro/eval_results_all.json"
OUTPUT_FILE = "eval_results_pid.xlsx"
# OUTPUT_FILE = "eval_results_pid_pro.xlsx"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Assume structure is {"OneRec-8B-pro": { "task_name": { "test": { metrics... } } }}
    root_key = list(data.keys())[0]
    tasks_data = data[root_key]

    all_k_values = set()
    metric_types = set()

    # Pre-scan to find all K values and metric types
    for task, content in tasks_data.items():
        if task.startswith("_") or "test" not in content:
            continue
        
        metrics = content["test"]
        for key in metrics.keys():
            if key.startswith("pid_"):
                # Parse "pid_pass@1" -> "pid_pass", "1"
                match = re.match(r"(pid_[a-zA-Z]+)@(\d+)", key)
                if match:
                    m_type, k = match.groups()
                    metric_types.add(m_type)
                    all_k_values.add(int(k))

    sorted_k = sorted(list(all_k_values))
    sorted_metrics = sorted(list(metric_types))

    # Custom sort order for metrics
    def sort_key(m):
        if "pass" in m: return 0
        if "recall" in m: return 1
        if "ndcg" in m: return 2
        return 3
    
    sorted_metrics_custom = sorted(sorted_metrics, key=sort_key)

    rows = []
    for task, content in tasks_data.items():
        if task.startswith("_") or "test" not in content:
            continue
        
        test_metrics = content["test"]
        
        for m_type in sorted_metrics_custom:
            row = {"Task": task, "Metric": m_type}
            has_data = False
            for k in sorted_k:
                key = f"{m_type}@{k}"
                val = test_metrics.get(key, None)
                if val is not None:
                    has_data = True
                row[f"@{k}"] = val
            
            # Add row even if some values are None, but ensure at least some structure exists
            if has_data:
                rows.append(row)

    if not rows:
        print("No valid PID metrics found.")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to ensure Task, Metric, @1, @3... order
    # Filter cols to only include those present in df (handle missing K values if any)
    desired_cols = ["Task", "Metric"] + [f"@{k}" for k in sorted_k]
    cols = [c for c in desired_cols if c in df.columns]
    
    df = df[cols]

    # Save to Excel
    try:
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Successfully created {OUTPUT_FILE}")
    except ImportError as e:
        print(f"Error saving Excel file: {e}. Please ensure openpyxl is installed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
