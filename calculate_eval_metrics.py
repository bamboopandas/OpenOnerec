import json
import math
import os
import re
import random
from typing import List, Dict, Set, Tuple

# Configuration
K_VALUES = [1, 3, 5, 10, 16, 32]
TASKS = ["ad", "product", "video"]
# Modified BASE_DIR for OneRec-8B
BASE_DIR = "benchmarks/results/v1.0_1000_thinktrue_tryspeed/results_results_8B/OneRec-8B/OneRec-8B" ##
# Output file in the same directory as eval_results.json for OneRec-8B
OUTPUT_FILE = "benchmarks/results/v1.0_1000_thinktrue_tryspeed/results_results_8B/OneRec-8B/eval_results_all.json"
DATA_DIR = "raw_data/onerec_data/benchmark_data"

# Constants for SID encoding
CODE_MULTIPLIER_1 = 8192 * 8192
CODE_MULTIPLIER_2 = 8192

def load_pid_mapping(mapping_path: str) -> Dict[int, List[Dict[str, int]]]:
    print(f"Loading mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def extract_sid_codes_from_text(text: str):
    pattern = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return (int(matches[0][0]), int(matches[0][1]), int(matches[0][2]))

def encode_sid(c1: int, c2: int, c3: int) -> int:
    return c1 * CODE_MULTIPLIER_1 + c2 * CODE_MULTIPLIER_2 + c3

def get_pid_from_sid(sid_str: str, mapping: Dict[int, List[Dict]], strategy="most_popular_after_downsampling") -> int:
    sid_codes = extract_sid_codes_from_text(sid_str)
    if not sid_codes:
        return 0
    
    encoded_sid = encode_sid(*sid_codes)
    pid_info_list = mapping.get(encoded_sid, [])
    
    if not pid_info_list:
        return 0
        
    if strategy == "most_popular_after_downsampling":
        max_count = max(info.get("count_after_downsample", 0) for info in pid_info_list)
        candidates = [info for info in pid_info_list if info.get("count_after_downsample", 0) == max_count]
        chosen = random.choice(candidates)
        return chosen.get("pid", chosen.get("iid", 0))
    else:
        # Default or fallback
        chosen = pid_info_list[0]
        return chosen.get("pid", chosen.get("iid", 0))

def extract_ids_from_answer(answer: str) -> List[str]:
    """Extract SIDs from ground truth string."""
    ids = []
    for part in answer.split('<|sid_begin|>'):
        if '<|sid_end|>' in part:
            sid = part.split('<|sid_end|>')[0].strip()
            if sid:
                ids.append(sid)
    return ids

def extract_id_from_generation(generation: str) -> str:
    """Extract SID from generation string."""
    generation = generation.strip()
    if '</think>' in generation:
        generation = generation.split('</think>')[-1].strip()
    
    if '<|sid_begin|>' in generation:
        for part in generation.split('<|sid_begin|>'):
            if '<|sid_end|>' in part:
                return part.split('<|sid_end|>')[0].strip()
            elif part.strip() and not part.startswith('<s_'): 
                 # Handle malformed tags if any, but regex extraction later handles format
                 pass
                 
    # Also try to just find the pattern directly if tags are missing
    if '<s_a_' in generation:
        match = re.search(r'<s_a_\d+><s_b_\d+><s_c_\d+>', generation)
        if match:
            return match.group(0)
            
    return generation

def compute_metrics(predicted, ground_truth, k_values, prefix=""):
    metrics = {}
    gt_set = set(ground_truth)
    # Remove 0 for PIDs if present
    if 0 in gt_set:
        gt_set.remove(0)
    
    if not gt_set:
        for k in k_values:
            metrics[f"{prefix}pass@{k}"] = 0.0
            metrics[f"{prefix}recall@{k}"] = 0.0
            metrics[f"{prefix}ndcg@{k}"] = 0.0
        return metrics

    for k in k_values:
        top_k = predicted[:k]
        
        # Pass@K
        passed = any(p in gt_set for p in top_k if p != 0)
        metrics[f"{prefix}pass@{k}"] = 1.0 if passed else 0.0
        
        # Recall@K
        hits = len(set(p for p in top_k if p != 0) & gt_set)
        metrics[f"{prefix}recall@{k}"] = hits / len(gt_set)
        
        # NDCG@K
        dcg = 0.0
        seen_hits = set()
        for i, p in enumerate(top_k):
            if p == 0: continue
            rel = 0.0
            if p in gt_set and p not in seen_hits:
                rel = 1.0
                seen_hits.add(p)
            dcg += rel / math.log2(i + 2)
            
        ideal_len = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        metrics[f"{prefix}ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics

def main():
    # Load mappings
    sid2pid = load_pid_mapping(os.path.join(DATA_DIR, "sid2pid.json"))
    sid2iid = load_pid_mapping(os.path.join(DATA_DIR, "sid2iid.json"))
    
    all_results = {"OneRec-8B": {}}
    
    for task in TASKS:
        file_path = os.path.join(BASE_DIR, task, "test_generated.json")
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        print(f"Processing {task}...")
        
        # Choose mapping
        if task == "product":
            mapping = sid2iid
        else:
            mapping = sid2pid
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        samples = data.get("samples", {})
        
        valid_samples = 0
        sum_metrics = {}
        
        for sample in samples.values():
            generations = sample.get("generations", [])
            ground_truth_str = sample.get("ground_truth", "")
            
            # SID Extraction (from ground_truth field)
            gt_sids = extract_ids_from_answer(ground_truth_str)
            pred_sids = [extract_id_from_generation(gen) for gen in generations]
            
            # SID Metrics
            sid_metrics = compute_metrics(pred_sids, gt_sids, K_VALUES, prefix="")
            
            # PID Conversion for PREDICTIONS
            pred_pids = [get_pid_from_sid(sid, mapping) for sid in pred_sids]
            
            # Extract PID GROUND TRUTH directly from metadata
            metadata = sample.get("metadata", {})
            gt_pids = metadata.get("answer_pid")
            if gt_pids is None:
                gt_pids = metadata.get("answer_iid")
            
            if gt_pids is None:
                # Fallback? Or 0?
                gt_pids = []
            elif isinstance(gt_pids, str):
                 # Not sure if it's stored as string or list, eval code suggests it might be list or str
                 # But grep showed "answer_pid": [ ...
                 try:
                     gt_pids = json.loads(gt_pids)
                 except:
                     pass # Assume it's already a list if loads fails or wasn't string
            
            # Ensure gt_pids is a list of ints
            if not isinstance(gt_pids, list):
                gt_pids = [] # Or handle if it's single int
            
            # PID Metrics
            pid_metrics = compute_metrics(pred_pids, gt_pids, K_VALUES, prefix="pid_")
            
            # Combine
            combined = {**sid_metrics, **pid_metrics}
            
            for key, value in combined.items():
                sum_metrics[key] = sum_metrics.get(key, 0.0) + value
            
            valid_samples += 1

        if valid_samples > 0:
            avg_metrics = {k: v / valid_samples for k, v in sum_metrics.items()}
            all_results["OneRec-8B"][task] = {
                "test": {
                    "total_samples": valid_samples,
                    **avg_metrics
                }
            }
        else:
             all_results["OneRec-8B"][task] = {"test": {"total_samples": 0}}

    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()