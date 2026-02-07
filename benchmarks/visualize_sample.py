import os
import sys
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
import re
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark import DataLoaderWrapper

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def analyze_attention(model, tokenizer, full_text, prompt_text, device):
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    ids_list = input_ids[0].tolist()
    
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    
    sid_begin_indices = [i for i, x in enumerate(ids_list) if x == sid_begin_id]
    
    split_idx = -1
    end_idx = len(ids_list)
    
    if sid_begin_indices:
        split_idx = sid_begin_indices[-1]
        try:
            end_idx = ids_list.index(sid_end_id, split_idx) + 1
        except ValueError:
            end_idx = len(ids_list)
    else:
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_end_id in ids_list:
             locs = [i for i, x in enumerate(ids_list) if x == think_end_id]
             split_idx = locs[-1] + 1
        else:
             prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
             split_idx = prompt_enc.input_ids.shape[1]

    context_indices = list(range(0, split_idx))
    query_indices = list(range(split_idx, end_idx))
    
    if not query_indices:
        return [], {'mean': np.zeros(len(context_indices)), 'first': np.zeros(len(context_indices))}

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    last_layer_attn = outputs.attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0)
    
    attn_block = avg_attn[query_indices, :][:, context_indices]
    
    attentions = {}
    if attn_block.shape[0] > 0:
        attentions['mean'] = attn_block.mean(dim=0).to(torch.float32).cpu().numpy()
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
    # CORRECT REGEX
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
    
    if count_T > 0 and count_O > 0 and A_O > 0:
        avg_attn_T = A_T / count_T
        avg_attn_O = A_O / count_O
        MDI = avg_attn_T / avg_attn_O
    else:
        MDI = 0.0
        
    if (count_T + count_O) > 0 and count_T > 0:
        Q_T = count_T / (count_T + count_O)
        AEI = A_T / Q_T
    else:
        AEI = 0.0
        
    return {
        "MDI": float(MDI), 
        "AEI": float(AEI), 
        "A_T": float(A_T), 
        "A_O": float(A_O), 
        "count_T": int(count_T), 
        "count_O": int(count_O)
    }

def plot_text_heatmap(tokens, attn, title, filename, v_max):
    # Simple matplotlib heatmap logic
    A4_WIDTH = 8.27
    A4_HEIGHT = 11.69
    MARGIN_X = 0.5
    MARGIN_TOP = 0.9
    MARGIN_BOTTOM = 1.0
    USABLE_WIDTH = A4_WIDTH - 2 * MARGIN_X
    USABLE_HEIGHT = A4_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    MAX_X = 100
    LINE_HEIGHT = 6.0
    FONT_SIZE = 10
    
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    
    # Font
    import matplotlib.font_manager as fm
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
    else:
        font_prop = fm.FontProperties(family=['sans-serif'])

    cmap = plt.get_cmap('YlOrRd')
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=v_max)
    
    def get_visual_length(s):
        l = 0
        for c in s:
            if ord(c) > 127: l += 1.8 
            else: l += 1.0 
        return l

    layout_items = []
    current_x = 0
    current_y = 0
    
    for i, token in enumerate(tokens):
        display_token = token.replace('\n', '\\n').replace('\t', '→')
        w = 1.0 + 1.2 * get_visual_length(display_token)
        
        if current_x + w > MAX_X:
            current_x = 0
            current_y -= LINE_HEIGHT
        
        val = attn[i]
        color_val = min(max(val, 0), v_max)
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
        
    # Pages
    lines_per_page = int(USABLE_HEIGHT / (LINE_HEIGHT * (USABLE_WIDTH / MAX_X)))
    pages = []
    if layout_items:
        current_page_items = []
        current_page_idx = 0
        for item in layout_items:
            line_idx = abs(item['y']) / LINE_HEIGHT
            page_idx = int(line_idx // lines_per_page)
            if page_idx > current_page_idx:
                pages.append(current_page_items)
                current_page_items = []
                current_page_idx = page_idx
            
            page_start_y = -(page_idx * lines_per_page * LINE_HEIGHT)
            local_y = item['y'] - page_start_y
            local_item = item.copy()
            local_item['y'] = local_y
            current_page_items.append(local_item)
        if current_page_items: pages.append(current_page_items)
    
    with PdfPages(filename) as pdf:
        for p_idx, items in enumerate(pages):
            fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
            ax = fig.add_axes([MARGIN_X/A4_WIDTH, MARGIN_BOTTOM/A4_HEIGHT, 
                               USABLE_WIDTH/A4_WIDTH, USABLE_HEIGHT/A4_HEIGHT])
            ax.set_xlim(0, MAX_X)
            y_span_units = lines_per_page * LINE_HEIGHT
            ax.set_ylim(-y_span_units, 0.5)
            ax.axis('off')
            
            for item in items:
                rect = Rectangle((item['x'], item['y'] - item['h']*0.85), item['w'], item['h'], 
                                 facecolor=item['bg'], edgecolor='none')
                ax.add_patch(rect)
                ax.text(item['x'] + item['w']/2, item['y'] - item['h']*0.35, item['text'], 
                        color=item['fg'], fontsize=FONT_SIZE, ha='center', va='center',
                        fontproperties=font_prop)
            
            # Colorbar
            cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            pdf.savefig(fig)
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="ad")
    parser.add_argument("--sample_id", type=str, default="41")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 1. Load Data
    print("Loading data...")
    loader_on = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=True)
    data_on = loader_on.load_data(args.task_name, split="test", sample_size=None)
    
    loader_off = DataLoaderWrapper(args.model_path, "v1.0", args.data_dir, enable_thinking=False)
    data_off = loader_off.load_data(args.task_name, split="test", sample_size=None)
    
    sid = args.sample_id
    if sid not in data_on or sid not in data_off:
        print(f"Sample {sid} not found. Using random.")
        common = list(set(data_on.keys()) & set(data_off.keys()))
        sid = random.choice(common)
    
    print(f"Visualizing Sample: {sid}")
    
    prompt_on = data_on[sid]["prompt"]
    prompt_off = data_off[sid]["prompt"]
    
    # 2. Generate
    print("Generating...")
    llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.6, dtype="bfloat16")
    sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.9, max_tokens=2048, stop=["<|sid_end|", "<|im_end|>"], seed=42)
    
    out_on = llm.generate([prompt_on], sampling_params)[0].outputs[0].text
    out_off = llm.generate([prompt_off], sampling_params)[0].outputs[0].text
    
    full_text_on = prompt_on + out_on
    full_text_off = prompt_off + out_off
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Analyze
    print("Analyzing...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    tokens_on, attns_on = analyze_attention(model, tokenizer, full_text_on, prompt_on, model.device)
    tokens_off, attns_off = analyze_attention(model, tokenizer, full_text_off, prompt_off, model.device)
    
    m_on = calculate_metrics(tokens_on, attns_on['mean'])
    m_off = calculate_metrics(tokens_off, attns_off['mean'])
    
    print(f"\nMetrics for Sample {sid}:")
    print(f"ON:  MDI={m_on['MDI']:.4f}, AEI={m_on['AEI']:.4f}, Count_T={m_on['count_T']}, Count_O={m_on['count_O']}")
    print(f"OFF: MDI={m_off['MDI']:.4f}, AEI={m_off['AEI']:.4f}, Count_T={m_off['count_T']}, Count_O={m_off['count_O']}")
    
    # Save JSON
    res_on = {
        "tokens": tokens_on, 
        "attention_values_mean": attns_on['mean'].tolist(),
        "metrics": m_on
    }
    with open(os.path.join(args.output_dir, "results_think_on.json"), "w") as f:
        json.dump(res_on, f, indent=2, ensure_ascii=False)
        
    res_off = {
        "tokens": tokens_off, 
        "attention_values_mean": attns_off['mean'].tolist(),
        "metrics": m_off
    }
    with open(os.path.join(args.output_dir, "results_think_off.json"), "w") as f:
        json.dump(res_off, f, indent=2, ensure_ascii=False)
        
    # Plot
    def get_robust_limit(arr):
        if len(arr) == 0: return 1.0
        return np.percentile(arr[1:], 98) if len(arr)>1 else 1.0

    vmax = max(get_robust_limit(attns_on['mean']), get_robust_limit(attns_off['mean']))
    
    plot_text_heatmap(tokens_on, attns_on['mean'], "Think=On", os.path.join(args.output_dir, "heatmap_on_mean.pdf"), vmax)
    plot_text_heatmap(tokens_off, attns_off['mean'], "Think=Off", os.path.join(args.output_dir, "heatmap_off_mean.pdf"), vmax)
    
    print(f"Saved PDF heatmaps to {args.output_dir}")

if __name__ == "__main__":
    setup_seed(42)
    main()
