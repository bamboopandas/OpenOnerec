import json
import random
import re
import os
import sys

# File paths
think_true_path = "benchmarks/results/v1.0_1000_thinktrue/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"
think_false_path = "benchmarks/results/v1.0_1000_thinkfalse/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"

def clear_screen():
    os.system('clear')

def load_json(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_think_content(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No thought process found."

def print_metrics_comparison(metrics_true, metrics_false, metric_names):
    print(f"{ 'Metric':<25} | { 'Think Mode':<15} | { 'False Mode':<15}")
    print("-" * 65)
    for metric in metric_names:
        val_true = metrics_true.get(metric, "N/A")
        val_false = metrics_false.get(metric, "N/A")
        
        # Format floats
        if isinstance(val_true, float):
            val_true = f"{val_true:.4f}"
        if isinstance(val_false, float):
            val_false = f"{val_false:.4f}"
            
        print(f"{metric:<25} | {str(val_true):<15} | {str(val_false):<15}")

def main():
    print("Loading data...")
    data_true = load_json(think_true_path)
    data_false = load_json(think_false_path)

    if not data_true or not data_false:
        return

    samples_true = data_true.get("samples", {})
    samples_false = data_false.get("samples", {})

    # Find common keys
    common_keys = list(set(samples_true.keys()) & set(samples_false.keys()))
    print(f"Found {len(common_keys)} common samples.")

    if not common_keys:
        print("No common samples found.")
        return

    metrics_to_show = [
        "pass@1", "pass@32", 
        "recall@1", "recall@32",
        "pid_pass@1", "pid_pass@32",
        "pid_recall@1", "pid_recall@32",
        "position1_pass@1", "position1_pass@32"
    ]

    while True:
        clear_screen()
        
        # Sample 1 random key
        selected_key = random.choice(common_keys)
        
        print(f"{ '='*30} Sample ID: {selected_key} {'='*30}")
        
        item_true = samples_true[selected_key]
        item_false = samples_false[selected_key]

        # 1. Comprehensive Metrics Comparison
        print("\n--- Comprehensive Metrics Comparison ---")
        print_metrics_comparison(item_true, item_false, metrics_to_show)

        # 2. Thought Process (Think Mode)
        generations_true = item_true.get("generations", [])
        thought_process = "N/A"
        if generations_true:
            thought_process = extract_think_content(generations_true[0])
            
        print("\n--- Thought Process (Think Mode) ---")
        print(thought_process)
        
        print("\n" + "="*76)
        try:
            user_input = input("Press Enter to sample another point, or type 'q' to quit: ")
            if user_input.lower() == 'q':
                break
        except EOFError:
            break

if __name__ == "__main__":
    main()