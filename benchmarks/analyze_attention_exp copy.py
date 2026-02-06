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
import matplotlib.colors as mcolors

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark import DataLoaderWrapper
from benchmark.tasks.v1_0.registry import get_loader
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_vllm(model_path, prompt, enable_thinking=True):
    """
    Generate using vLLM.
    We initialize vLLM inside this function (or a context manager) 
    because we might need to clear it to load the HF model later.
    However, for efficiency, if we run both On/Off, we can keep it alive 
    if we generate both before switching to HF.
    """
    # Note: Initializing LLM is heavy. Ideally we do it once.
    pass

def analyze_attention(model, tokenizer, full_text, prompt_text, device):
    """
    Run forward pass and extract attention.
    Context = Prompt + Thinking + Prefix
    Query = Semantic ID Sequence (Answer)
    
    Split Strategy: Find the LAST <|sid_begin|> tag. 
    Everything before it is Context. 
    The SID sequence itself is the Query.
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
        # (Previous occurrences might be in the prompt history)
        split_idx = sid_begin_indices[-1]
        
        # Find corresponding end tag
        try:
            end_idx = ids_list.index(sid_end_id, split_idx) + 1 # Include end tag
        except ValueError:
            end_idx = len(ids_list)
            
        print(f"DEBUG: Found last <|sid_begin|> at token {split_idx}. Treating previous as Context.")
    else:
        # Fallback if no SID found (e.g. error or text-only answer)
        # Try finding </think>
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_end_id in ids_list:
             locs = [i for i, x in enumerate(ids_list) if x == think_end_id]
             split_idx = locs[-1] + 1
             print(f"DEBUG: No SID found. Splitting at </think> (token {split_idx}).")
        else:
             # Fallback to Prompt Length
             prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
             split_idx = prompt_enc.input_ids.shape[1]
             print(f"DEBUG: No SID/Think found. Splitting at Prompt Length ({split_idx}).")

    context_indices = list(range(0, split_idx))
    query_indices = list(range(split_idx, end_idx))
    
    if not query_indices:
        print("Error: No query tokens found.")
        # Return valid empty result to avoid crash
        zeros = np.zeros(len(context_indices))
        return [], {'mean': zeros, 'first': zeros}

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Get Last Layer Attention
    last_layer_attn = outputs.attentions[-1] # (1, Heads, Seq, Seq)
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # (Seq, Seq)
    
    # Slice the attention matrix
    # Rows: Queries (Answer)
    # Cols: Keys (Context)
    attn_block = avg_attn[query_indices, :][:, context_indices] # (Query_Len, Context_Len)
    
    # Calculate Attentions
    attentions = {}
    
    if attn_block.shape[0] > 0:
        # Mean: Average of all query tokens
        attentions['mean'] = attn_block.mean(dim=0).to(torch.float32).cpu().numpy()
        
        # First: First Semantic ID (Index 1)
        # If query has <|sid_begin|> and IDs, index 0 is tag, index 1 is first ID.
        if attn_block.shape[0] >= 2:
            attentions['first'] = attn_block[1, :].to(torch.float32).cpu().numpy()
        else:
            # Fallback to index 0 (tag) if no IDs follow
            attentions['first'] = attn_block[0, :].to(torch.float32).cpu().numpy()
    else:
        zeros = np.zeros(len(context_indices))
        attentions['mean'] = zeros
        attentions['first'] = zeros
    
    # Get tokens for visualization
    tokens = []
    for idx in context_indices:
        tid = ids_list[idx]
        t_str = tokenizer.decode([tid], skip_special_tokens=False)
        tokens.append(t_str)
    
    # Debug
    print(f"DEBUG: Context Size: {len(tokens)}, Query Size: {len(query_indices)}")
    if any("<think>" in t for t in tokens):
        print("DEBUG: <think> tag found in tokens context.")
    else:
        print("DEBUG: <think> tag NOT found in tokens context.")
    
    return tokens, attentions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="ad")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Load Custom Font
    import matplotlib.font_manager as fm
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        print(f"Loaded font from {font_path}")
    else:
        font_prop = fm.FontProperties(family=['sans-serif', 'WenQuanYi Micro Hei', 'SimHei'])
        print("Warning: SimHei.ttf not found, using system fallback.")
    
    # 1. Select Sample
    print(f"Loading data for task: {args.task_name}")
    loader_on = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=True)
    data_on = loader_on.load_data(args.task_name, split="test", sample_size=100)
    
    loader_off = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=False)
    data_off = loader_off.load_data(args.task_name, split="test", sample_size=100)
    
    common_ids = list(set(data_on.keys()) & set(data_off.keys()))
    if not common_ids:
        raise ValueError("No common samples found!")
    
    sample_id = random.choice(common_ids)
    print(f"Selected Sample ID: {sample_id}")
    
    sample_on = data_on[sample_id]
    sample_off = data_off[sample_id]
    
    prompt_on = sample_on["prompt"]
    prompt_off = sample_off["prompt"]
    
    # Force Think Mode if not present
    if "<think>" not in prompt_on and "<|im_start|>think" not in prompt_on:
        print("DEBUG: <think> not in prompt. Appending manually.")
        if prompt_on.endswith("\n"):
            prompt_on += "<think>\n"
        else:
            prompt_on += "\n<think>\n"
            
    print(f"DEBUG: Prompt On Tail: {prompt_on[-50:]!r}")
    
    # Save Sample Metadata
    with open(os.path.join(args.output_dir, "sample_metadata.json"), "w") as f:
        json.dump({"sample_id": sample_id, "seed": 42}, f)
        
    # 2. Generation (vLLM)
    print("Starting vLLM Generation...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        dtype="bfloat16"
    )
    
    # Add stop token to ensure we only generate one item (as per user request)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=2048,
        stop=["<|sid_end|", "<|im_end|>"] 
    )
    
    # Run Think=On
    print(f"Generating On...")
    output_on = llm.generate([prompt_on], sampling_params)[0]
    text_on_generated = output_on.outputs[0].text
    full_text_on = prompt_on + text_on_generated
    
    # Run Think=Off
    print(f"Generating Off...")
    output_off = llm.generate([prompt_off], sampling_params)[0]
    text_off_generated = output_off.outputs[0].text
    full_text_off = prompt_off + text_off_generated
    
    # Cleanup vLLM
    print("Cleaning up vLLM...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Analysis (HF)
    print("Starting HF Analysis...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Analyze On
    print("Analyzing On...")
    tokens_on, attns_on = analyze_attention(model, tokenizer, full_text_on, prompt_on, model.device)
    
    # Analyze Off
    print("Analyzing Off...")
    tokens_off, attns_off = analyze_attention(model, tokenizer, full_text_off, prompt_off, model.device)
    
    # 4. Visualization
    # Robust Normalization (98th Percentile)
    def get_robust_limit(arr):
        if len(arr) == 0: return 1.0
        # Exclude BOS (index 0) from stats if possible
        data = arr
        if len(arr) > 1:
            data = arr[1:]
        return np.percentile(data, 98) # 98th percentile
    
    # Calculate limits based on 'first' since it's primary
    vmax_first = max(get_robust_limit(attns_on['first']), get_robust_limit(attns_off['first']))
    vmax_mean = max(get_robust_limit(attns_on['mean']), get_robust_limit(attns_off['mean']))
    vmin = 0
    
    print(f"Visualization VMax (First): {vmax_first}")
    print(f"Visualization VMax (Mean): {vmax_mean}")
    
    def plot_text_heatmap(tokens, attn, title, filename, v_max):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig_width = 14
        # Dynamic height estimation
        # Rough calc: chars per line ~ 80?
        total_chars = sum(len(t) for t in tokens)
        est_lines = total_chars / 60 + 2
        fig_height = max(2, est_lines * 0.3)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        
        ax.set_xlim(0, 100)
        # Use YlOrRd for scientific standard
        cmap = plt.get_cmap('YlOrRd')
        
        # PowerNorm for better visibility of low values
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=v_max)
        
        fontsize = 9
        
        current_x = 0
        current_y = 0
        line_height = 5
        max_x = 100
        
        for i, token in enumerate(tokens):
            display_token = token.replace('\n', '\\n').replace('\t', '→')
            # Handle special tokens
            if display_token.startswith("<|"):
                display_token = display_token 
            
            # Estimate width: 1.2 per char roughly
            w = 1.0 + 1.2 * len(display_token)
            
            # Wrap
            if current_x + w > max_x:
                current_x = 0
                current_y -= line_height
            
            val = attn[i]
            # Clip
            color_val = min(max(val, vmin), v_max)
            bg_color = cmap(norm(color_val))
            
            # Text Color: Black for light (low attention), White for dark (high attention)
            r, g, b, _ = bg_color
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = 'black' if luminance > 0.5 else 'white'
            
            rect = Rectangle((current_x, current_y - line_height*0.85), w, line_height, 
                             facecolor=bg_color, edgecolor='none')
            ax.add_patch(rect)
            
            ax.text(current_x + w/2, current_y - line_height*0.35, display_token, 
                    color=text_color, fontsize=fontsize, ha='center', va='center',
                    fontproperties=font_prop)
            
            current_x += w
            
        y_min = current_y - line_height
        y_max = line_height
        
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.05)
        cbar.set_label('Attention Value', fontproperties=font_prop)
        
        plt.title(title, fontproperties=font_prop)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    # Save Results
    # Convert numpy arrays to lists for JSON serialization
    res_on = {
        "tokens": tokens_on, 
        "attention_values_first": attns_on['first'].tolist(),
        "attention_values_mean": attns_on['mean'].tolist(),
        "full_text": full_text_on
    }
    with open(os.path.join(args.output_dir, "results_think_on.json"), "w") as f:
        json.dump(res_on, f, indent=2, ensure_ascii=False)
        
    res_off = {
        "tokens": tokens_off, 
        "attention_values_first": attns_off['first'].tolist(),
        "attention_values_mean": attns_off['mean'].tolist(),
        "full_text": full_text_off
    }
    with open(os.path.join(args.output_dir, "results_think_off.json"), "w") as f:
        json.dump(res_off, f, indent=2, ensure_ascii=False)
        
    # Plot First ID Attention (Requested)
    plot_text_heatmap(tokens_on, attns_on['first'], f"Attention Map (Think=On, First ID)\nSample {sample_id}", os.path.join(args.output_dir, "heatmap_on_first.pdf"), vmax_first)
    plot_text_heatmap(tokens_off, attns_off['first'], f"Attention Map (Think=Off, First ID)\nSample {sample_id}", os.path.join(args.output_dir, "heatmap_off_first.pdf"), vmax_first)
    
    # Plot Mean Attention (Optional/Reference)
    plot_text_heatmap(tokens_on, attns_on['mean'], f"Attention Map (Think=On, Mean)\nSample {sample_id}", os.path.join(args.output_dir, "heatmap_on_mean.pdf"), vmax_mean)
    plot_text_heatmap(tokens_off, attns_off['mean'], f"Attention Map (Think=Off, Mean)\nSample {sample_id}", os.path.join(args.output_dir, "heatmap_off_mean.pdf"), vmax_mean)
    
    # Copy 'first' to standard names for ease of access
    import shutil
    shutil.copy(os.path.join(args.output_dir, "heatmap_on_first.pdf"), os.path.join(args.output_dir, "heatmap_on.pdf"))
    shutil.copy(os.path.join(args.output_dir, "heatmap_off_first.pdf"), os.path.join(args.output_dir, "heatmap_off.pdf"))
    
    print("Done.")

if __name__ == "__main__":
    setup_seed(42)
    main()