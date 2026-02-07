import os
import sys
import torch
import re
import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.benchmark import DataLoaderWrapper

# Config
MODEL_PATH = "../checkpoints/OneRec-1.7B"
DATA_DIR = "../raw_data/onerec_data/benchmark_data_1000"
TASK_NAME = "ad"

def debug_sample():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("Loading data...")
    loader = DataLoaderWrapper(MODEL_PATH, "v1.0", DATA_DIR, enable_thinking=True)
    data = loader.load_data(TASK_NAME, split="test", sample_size=1)
    sid = list(data.keys())[0]
    sample = data[sid]
    prompt = sample["prompt"]
    
    print(f"Sample ID: {sid}")
    print(f"Prompt (Truncated): {prompt[-200:]!r}")
    
    # Analyze Tokens in Prompt (Think=Off context)
    inputs = tokenizer(prompt, return_tensors="pt")
    ids = inputs.input_ids[0].tolist()
    
    # Check sid_begin_id
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    print(f"SID Begin ID: {sid_begin_id}")
    if sid_begin_id in ids:
        print("SID Begin ID found in prompt.")
    else:
        print("SID Begin ID NOT found in prompt.")
        
    # Emulate analyze_attention loop
    tokens = []
    for tid in ids:
        t_str = tokenizer.decode([tid], skip_special_tokens=False)
        tokens.append(t_str)
    
    # Regex
    sid_pattern = re.compile(r"^<s_[abc]_\d+>$")
    
    count_O = 0
    count_T = 0
    sids_found = []
    
    print("\n--- Token Analysis (Prompt/Think=Off) ---")
    for i, t in enumerate(tokens):
        if sid_pattern.match(t):
            count_O += 1
            sids_found.append(t)
        else:
            count_T += 1
            
    print(f"Total Tokens: {len(tokens)}")
    print(f"SIDs Found (Count_O): {count_O}")
    print(f"Text Tokens (Count_T): {count_T}")
    print(f"First 5 SIDs: {sids_found[:5]}")

    print("\nLoading data (Think=False)...")
    loader_off = DataLoaderWrapper(MODEL_PATH, "v1.0", DATA_DIR, enable_thinking=False)
    data_off = loader_off.load_data(TASK_NAME, split="test", sample_size=1)
    prompt_off = data_off[sid]["prompt"]
    print(f"Prompt Off (Truncated): {prompt_off[-200:]!r}")
    
    if "/think" in prompt and "/no_think" in prompt_off:
        print("SUCCESS: Prompts distinguish Think modes correctly.")
    else:
        print("FAILURE: Prompts do not match expected patterns.")
        
    # Check if Off prompt has <think></think> pre-filled
    # Template: <think>\n\n</think>\n\n
    if "<think>" in prompt_off:
        print("Prompt Off contains <think> tag (Template pre-filled empty thought).")
    else:
        print("Prompt Off does NOT contain <think> tag.")

if __name__ == "__main__":
    debug_sample()