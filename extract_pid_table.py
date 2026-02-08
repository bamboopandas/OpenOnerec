import json
import os
import re

# File path
FILE_PATH = "benchmarks/results/v1.0_1000_thinkfalse_tryspeed/results_results_1.7B/OneRec-1.7B/eval_results_all.json"

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    # Assume structure is {"OneRec-1.7B": { "task_name": { "test": { metrics... } } }}
    # Adjust if root key varies, but based on previous context, it's "OneRec-1.7B"
    root_key = list(data.keys())[0]
    tasks_data = data[root_key]

    # Collect all unique K values and Metric Types
    all_k_values = set()
    metric_types = set()

    # Pre-scan to find all K values
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
    sorted_metrics = sorted(list(metric_types)) # e.g., pid_ndcg, pid_pass, pid_recall

    # Print Header
    header = ["Task", "Metric"] + [f"@{k}" for k in sorted_k]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    # Print Rows
    for task, content in tasks_data.items():
        if task.startswith("_") or "test" not in content:
            continue
        
        test_metrics = content["test"]
        
        # Sort metrics for consistent grouping: Pass, Recall, NDCG is common, but alphabetic is fine too.
        # Let's try to order them logically: Pass, Recall, NDCG if possible, else alphabetic.
        # Custom sort order: pass, recall, ndcg
        def sort_key(m):
            if "pass" in m: return 0
            if "recall" in m: return 1
            if "ndcg" in m: return 2
            return 3
        
        sorted_metrics_custom = sorted(sorted_metrics, key=sort_key)

        for m_type in sorted_metrics_custom:
            row = [task, m_type]
            for k in sorted_k:
                key = f"{m_type}@{k}"
                val = test_metrics.get(key, "-")
                if isinstance(val, (float, int)):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            
            print("| " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
