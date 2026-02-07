import os
import sys
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
from typing import List, Dict, Any, Tuple
import re
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark import DataLoaderWrapper
from benchmark.tasks.v1_0.registry import get_evaluator, get_task_config
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def analyze_attention(model, tokenizer, full_text, prompt_text, device):
    """
    Run forward pass and extract attention.
    Context = Prompt + Thinking + Prefix
    Query = Semantic ID Sequence (Answer)
    """
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    ids_list = input_ids[0].tolist()
    
    # Target IDs
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    
    # Find all occurrences
    sid_begin_indices = [i for i, x in enumerate(ids_list) if x == sid_begin_id]
    
    split_idx = -1
    end_idx = len(ids_list)
    
    if sid_begin_indices:
        # Use the LAST occurrence to identify the generated answer
        split_idx = sid_begin_indices[-1]
        
        # Find corresponding end tag
        try:
            end_idx = ids_list.index(sid_end_id, split_idx) + 1 # Include end tag
        except ValueError:
            end_idx = len(ids_list)
    else:
        # Fallback if no SID found
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_end_id in ids_list:
             locs = [i for i, x in enumerate(ids_list) if x == think_end_id]
             split_idx = locs[-1] + 1
        else:
             # Fallback to Prompt Length
             prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
             split_idx = prompt_enc.input_ids.shape[1]

    context_indices = list(range(0, split_idx))
    query_indices = list(range(split_idx, end_idx))
    
    if not query_indices:
        # Return valid empty result
        return [], {'mean': np.zeros(len(context_indices)), 'first': np.zeros(len(context_indices))}

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Get Last Layer Attention
    last_layer_attn = outputs.attentions[-1] # (1, Heads, Seq, Seq)
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # (Seq, Seq)
    
    # Slice the attention matrix
    # Rows: Queries (Answer)
    # Cols: Keys (Context)
    attn_block = avg_attn[query_indices, :][:, context_indices] # (Query_Len, Context_Len)
    
    attentions = {}
    
    if attn_block.shape[0] > 0:
        # Mean: Average of all query tokens
        attentions['mean'] = attn_block.mean(dim=0).to(torch.float32).cpu().numpy()
        
        # First: First Semantic ID (Index 1)
        if attn_block.shape[0] >= 2:
            attentions['first'] = attn_block[1, :].to(torch.float32).cpu().numpy()
        else:
            attentions['first'] = attn_block[0, :].to(torch.float32).cpu().numpy()
    else:
        zeros = np.zeros(len(context_indices))
        attentions['mean'] = zeros
        attentions['first'] = zeros
    
    tokens = []
    for idx in context_indices:
        tid = ids_list[idx]
        t_str = tokenizer.decode([tid], skip_special_tokens=False)
        tokens.append(t_str)
    
    return tokens, attentions

def calculate_metrics(tokens, attention_values):
    """
    Calculate MDI and AEI metrics.
    
    O: Semantic ID tokens (<s_...>)
    T: Other tokens (Text, tags, etc.)
    """
    # Regex for Semantic ID
    # Ensure raw string and correct backslash for digits
    sid_pattern = re.compile(r"^<s_[abc]_\d+>$")
    
    sum_attn_O = 0.0
    count_O = 0
    sum_attn_T = 0.0
    count_T = 0
    
    for token, attn_val in zip(tokens, attention_values):
        if sid_pattern.match(token):
            sum_attn_O += attn_val
            count_O += 1
        else:
            sum_attn_T += attn_val
            count_T += 1
            
    total_raw = sum_attn_T + sum_attn_O
    if total_raw == 0:
        return {"MDI": 0, "AEI": 0, "A_T": 0, "A_O": 0, "count_T": count_T, "count_O": count_O}
        
    A_T = sum_attn_T / total_raw
    A_O = sum_attn_O / total_raw
    
    # MDI
    if count_T > 0 and count_O > 0 and A_O > 0:
        avg_attn_T = A_T / count_T
        avg_attn_O = A_O / count_O
        MDI = avg_attn_T / avg_attn_O
    else:
        MDI = 0.0
        
    # AEI
    if (count_T + count_O) > 0 and count_T > 0:
        Q_T = count_T / (count_T + count_O)
        AEI = A_T / Q_T
    else:
        AEI = 0.0
        
    return {
        "MDI": float(MDI), 
        "AEI": float(AEI),
        "count_T": int(count_T), 
        "count_O": int(count_O)
    }

def evaluate_samples(task_name, samples, data_dir, output_dir):
    """
    Use Benchmark Evaluator to score samples.
    """
    evaluator_class = get_evaluator(task_name)
    task_config = get_task_config(task_name)
    
    evaluator = evaluator_class(
        samples=samples,
        task_name=task_name,
        predictions_dir=output_dir, # Dummy
        debug=False,
        task_config=task_config,
        data_dir=data_dir
    )
    
    metrics, per_sample_metrics = evaluator.evaluate()
    return per_sample_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="ad")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 1. Load Data
    print(f"Loading data for task: {args.task_name} (Think=True)...")
    loader_on = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=True)
    data_on = loader_on.load_data(args.task_name, split="test", sample_size=None)
    
    print(f"Loading data for task: {args.task_name} (Think=False)...")
    loader_off = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=False)
    data_off = loader_off.load_data(args.task_name, split="test", sample_size=None)
    
    # Intersect IDs
    sample_ids = list(set(data_on.keys()) & set(data_off.keys()))
    print(f"Total Common Samples: {len(sample_ids)}")
    
    # 2. Generation (vLLM)
    print("Starting vLLM Generation...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=2048,
        stop=["<|sid_end|", "<|im_end|>"],
        seed=42
    )
    
    # Prepare Prompts
    prompts_on = []
    prompts_off = []
    
    for sid in sample_ids:
        prompts_on.append(data_on[sid]["prompt"])
        prompts_off.append(data_off[sid]["prompt"])
        
    print(f"Generating {len(prompts_on)} samples (Think=On)...")
    outputs_on = llm.generate(prompts_on, sampling_params)
    
    print(f"Generating {len(prompts_off)} samples (Think=Off)...")
    outputs_off = llm.generate(prompts_off, sampling_params)
    
    # Store Generation Results
    gen_data_on = {}
    gen_data_off = {}
    
    for i, sid in enumerate(sample_ids):
        # On
        text_on = outputs_on[i].outputs[0].text
        gen_data_on[sid] = {
            "generations": [text_on],
            "ground_truth": data_on[sid]["ground_truth"],
            "metadata": data_on[sid].get("metadata", {}),
            "full_text": prompts_on[i] + text_on,
            "prompt": prompts_on[i]
        }
        
        # Off
        text_off = outputs_off[i].outputs[0].text
        gen_data_off[sid] = {
            "generations": [text_off],
            "ground_truth": data_off[sid]["ground_truth"],
            "metadata": data_off[sid].get("metadata", {}),
            "full_text": prompts_off[i] + text_off,
            "prompt": prompts_off[i]
        }

    # Cleanup vLLM
    print("Cleaning up vLLM...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Evaluation
    print("Evaluating Performance...")
    metrics_on = evaluate_samples(args.task_name, gen_data_on, args.data_dir, args.output_dir)
    metrics_off = evaluate_samples(args.task_name, gen_data_off, args.data_dir, args.output_dir)
    
    # Extract Score
    sample_key_on = list(metrics_on.keys())[0]
    metric_keys = list(metrics_on[sample_key_on].keys())
    pass_key = next((k for k in metric_keys if k.startswith("pass@")), None)
    if not pass_key:
        print("Warning: No pass@k metric found. Using 'pass' if exists.")
        pass_key = "pass" # fallback
    else:
        print(f"Using metric: {pass_key} for performance.")

    # 4. Attention Analysis (HF)
    print("Starting HF Analysis...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    results_on = []
    results_off = []
    
    print("Analyzing Attention...")
    for sid in tqdm(sample_ids):
        # --- Think=On ---
        full_text_on = gen_data_on[sid]["full_text"]
        prompt_on = gen_data_on[sid]["prompt"]
        tokens_on, attns_on = analyze_attention(model, tokenizer, full_text_on, prompt_on, model.device)
        m_on = calculate_metrics(tokens_on, attns_on['mean'])
        
        # Calculate CoT Length
        try:
            think_start = tokens_on.index("<think>")
            think_end = tokens_on.index("</think>")
            cot_len = think_end - think_start - 1
            if cot_len < 0: cot_len = 0
        except ValueError:
            cot_len = 0
            
        perf_on = 1.0 if metrics_on[sid].get(pass_key, False) else 0.0
        
        results_on.append({
            "Sample_ID": sid,
            "MDI": m_on["MDI"],
            "AEI": m_on["AEI"],
            "CoT_Len": cot_len,
            "Performance": perf_on,
            "Count_T": m_on["count_T"],
            "Count_O": m_on["count_O"]
        })

        # --- Think=Off ---
        full_text_off = gen_data_off[sid]["full_text"]
        prompt_off = gen_data_off[sid]["prompt"]
        tokens_off, attns_off = analyze_attention(model, tokenizer, full_text_off, prompt_off, model.device)
        m_off = calculate_metrics(tokens_off, attns_off['mean'])
        
        # CoT Length for Off should be 0 (or negligible if empty tags exist)
        try:
             # Check if tags exist even if empty
             if "<think>" in tokens_off and "</think>" in tokens_off:
                 ts = tokens_off.index("<think>")
                 te = tokens_off.index("</think>")
                 cot_len_off = te - ts - 1
                 if cot_len_off < 0: cot_len_off = 0
             else:
                 cot_len_off = 0
        except ValueError:
             cot_len_off = 0

        perf_off = 1.0 if metrics_off[sid].get(pass_key, False) else 0.0
        
        results_off.append({
            "Sample_ID": sid,
            "MDI": m_off["MDI"],
            "AEI": m_off["AEI"],
            "CoT_Len": cot_len_off,
            "Performance": perf_off,
            "Count_T": m_off["count_T"],
            "Count_O": m_off["count_O"]
        })

    # 5. Save Results
    df_on = pd.DataFrame(results_on)
    df_off = pd.DataFrame(results_off)
    
    file_on = os.path.join(args.output_dir, "results_all_on.csv")
    file_off = os.path.join(args.output_dir, "results_all_off.csv")
    
    df_on.to_csv(file_on, index=False)
    df_off.to_csv(file_off, index=False)
    
    print(f"\nSaved Think=On results to {file_on}")
    print(f"Saved Think=Off results to {file_off}")
    
    # 6. Basic Statistics Printout
    print("\n=== Summary Statistics (On) ===")
    print(df_on.describe())
    
    print("\n=== Summary Statistics (Off) ===")
    print(df_off.describe())
    
    # 7. Correlation (Delta) for quick check
    # Merge on Sample_ID
    df_merged = pd.merge(df_on, df_off, on="Sample_ID", suffixes=(_on, _off))
    df_merged["Delta_Perf"] = df_merged["Performance_on"] - df_merged["Performance_off"]
    df_merged["Delta_MDI"] = df_merged["MDI_on"] - df_merged["MDI_off"]
    df_merged["Delta_AEI"] = df_merged["AEI_on"] - df_merged["AEI_off"]
    
    print("\n=== Correlations (Delta) ===")
    print(df_merged[["Delta_Perf", "Delta_MDI", "Delta_AEI"]].corr())

if __name__ == "__main__":
    setup_seed(42)
    main()
