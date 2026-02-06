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
import matplotlib.ticker as ticker

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
    
    sample_id = "41"
    if sample_id not in common_ids:
        print(f"Warning: Requested sample_id {sample_id} not in data. Falling back to random.")
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
        from matplotlib.backends.backend_pdf import PdfPages
        import math
        
        # A4 Dimensions (inches)
        A4_WIDTH = 8.27
        A4_HEIGHT = 11.69
        
        # Layout Settings
        MARGIN_X = 0.5
        MARGIN_TOP = 0.9
        MARGIN_BOTTOM = 1.0 # For legend
        
        USABLE_WIDTH = A4_WIDTH - 2 * MARGIN_X
        USABLE_HEIGHT = A4_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
        
        # Abstract Layout Units (0-100 width)
        MAX_X = 100
        LINE_HEIGHT = 6.0
        FONT_SIZE = 10
        
        # Mapping Units to Inches
        # 100 units = USABLE_WIDTH inches
        unit_to_inch = USABLE_WIDTH / MAX_X
        line_height_inch = LINE_HEIGHT * unit_to_inch
        
        lines_per_page = int(USABLE_HEIGHT / line_height_inch)
        
        # Colormap setup
        cmap = plt.get_cmap('YlOrRd')
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=v_max)
        
        # Robust Width Calculation
        def get_visual_length(s):
            l = 0
            for c in s:
                if ord(c) > 127: l += 1.8 # CJK character width factor
                else: l += 1.0 # ASCII
            return l

        # --- Pre-calculate Layout ---
        layout_items = []
        current_x = 0
        current_y = 0 # Starts at 0, goes down
        
        for i, token in enumerate(tokens):
            display_token = token.replace('\n', '\\n').replace('\t', '→')
            if display_token.startswith("<|"):
                display_token = display_token
            
            # Estimate width based on visual length
            # Base padding + char width * visual length
            # Tuned for font size 10 on this canvas width
            # Multiplier 1.2 ensures enough space for both ASCII and CJK
            w = 1.0 + 1.2 * get_visual_length(display_token)
            
            # Wrap
            if current_x + w > MAX_X:
                current_x = 0
                current_y -= LINE_HEIGHT
            
            val = attn[i]
            color_val = min(max(val, vmin), v_max)
            bg_color = cmap(norm(color_val))
            
            r, g, b, _ = bg_color
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = 'black' if luminance > 0.5 else 'white'
            
            layout_items.append({
                'text': display_token,
                'x': current_x,
                'y': current_y,
                'w': w,
                'h': LINE_HEIGHT,
                'bg': bg_color,
                'fg': text_color
            })
            
            current_x += w
            
        # --- Split into Pages ---
        # Group by line index
        pages = []
        if layout_items:
            # Determine total lines
            min_y = layout_items[-1]['y']
            # y goes 0, -6, -12...
            # line_index = abs(y) / 6
            
            current_page_items = []
            current_page_idx = 0
            
            for item in layout_items:
                line_idx = abs(item['y']) / LINE_HEIGHT
                page_idx = int(line_idx // lines_per_page)
                
                if page_idx > current_page_idx:
                    pages.append(current_page_items)
                    current_page_items = []
                    current_page_idx = page_idx
                
                # Adjust Y to be relative to page top (0)
                # Global Y is e.g. -150. Page start Y is e.g. -120 (if 20 lines/page * 6)
                # Local Y = -150 - (-120) = -30
                page_start_y = -(page_idx * lines_per_page * LINE_HEIGHT)
                local_y = item['y'] - page_start_y
                
                # Clone item with new Y
                local_item = item.copy()
                local_item['y'] = local_y
                current_page_items.append(local_item)
            
            if current_page_items:
                pages.append(current_page_items)
        
        # --- Render PDF ---
        with PdfPages(filename) as pdf:
            for p_idx, items in enumerate(pages):
                fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
                # Adjust figure layout to use more space
                # rect is [left, bottom, right, top] in normalized 0-1 coords
                # USABLE_HEIGHT is what we pre-calculated.
                # Let's adjust the subplot explicitly.
                ax = fig.add_axes([MARGIN_X/A4_WIDTH, MARGIN_BOTTOM/A4_HEIGHT, 
                                   USABLE_WIDTH/A4_WIDTH, USABLE_HEIGHT/A4_HEIGHT])
                
                # Set coordinate system
                ax.set_xlim(0, MAX_X)
                # Y goes from approx 0 to -Height
                y_span_units = lines_per_page * LINE_HEIGHT
                
                # Shifted ylim to start exactly at top (0)
                ax.set_ylim(-y_span_units, 0.5) 
                
                ax.axis('off')
                
                # Draw Items
                for item in items:
                    rect = Rectangle((item['x'], item['y'] - item['h']*0.85), item['w'], item['h'], 
                                     facecolor=item['bg'], edgecolor='none')
                    ax.add_patch(rect)
                    
                    ax.text(item['x'] + item['w']/2, item['y'] - item['h']*0.35, item['text'], 
                            color=item['fg'], fontsize=FONT_SIZE, ha='center', va='center',
                            fontproperties=font_prop)
                
                # Title - Removed as per request
                # plt.title(f"{title}\nPage {p_idx+1}/{len(pages)}", fontproperties=font_prop, y=0.98, fontsize=12)
                
                # Colorbar (Legend)
                # Position: [left, bottom, width, height] relative to figure
                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.locator = ticker.MaxNLocator(nbins=5)
                cbar.update_ticks()
                cbar.set_label('Attention Value', fontproperties=font_prop)
                
                pdf.savefig(fig)
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