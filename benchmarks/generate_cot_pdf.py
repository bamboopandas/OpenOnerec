import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


# ----------------------------
# 1. Scientific Paper Style
# ----------------------------
def set_paper_plot_style():
    """
    Configures Matplotlib for scientific publication quality:
    - Serif fonts (Times New Roman-like).
    - High DPI.
    - Minimalist aesthetics (no top/right spines).
    - Muted, professional color palette.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42, # Embed fonts
        "ps.fonttype": 42,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })

# ----------------------------
# Data Processing Helpers
# ----------------------------
def clean_content(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)

def extract_cot_content(text: str) -> Optional[str]:
    start_tag = "<think>"
    end_tag = "</think>"
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag, start)
        return text[start:end].strip()
    return None

def parse_messages(row_messages: Any) -> List[Dict[str, Any]]:
    if isinstance(row_messages, str):
        return json.loads(row_messages)
    return row_messages

def construct_prompt(row: pd.Series, tokenizer) -> str:
    msgs = parse_messages(row["messages"])
    clean_msgs = []
    for m in msgs:
        clean_msgs.append({"role": m["role"], "content": clean_content(m.get("content", ""))})
    return tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)

def construct_no_history_prompt(row: pd.Series, tokenizer) -> str:
    msgs = parse_messages(row["messages"])
    system_msg = msgs[0].copy()
    system_msg["content"] = clean_content(system_msg.get("content", ""))

    user_content = clean_content(msgs[1].get("content", ""))
    # Heuristic: keep instruction after last "请" (Please)
    if "请" in user_content:
        parts = user_content.split("请")
        instruction = "请" + parts[-1]
    else:
        instruction = user_content

    msgs_nohist = [system_msg, {"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(msgs_nohist, tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def get_last_token_hidden(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Use cache=True to speed up if we were looping, but for single calls it's fine.
    # We want the last hidden state of the last token.
    out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return h

class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids):]
        stop = torch.tensor(self.stop_ids, device=tail.device, dtype=tail.dtype)
        return torch.all(tail == stop).item()

# ----------------------------
# Main Visualization Logic
# ----------------------------
@dataclass
class VizConfig:
    model_path: str
    data_path: str
    output_file: str
    sample_size_sid: int = 400
    sample_size_other: int = 1500
    stride: int = 10
    seed: int = 42
    max_retries: int = 10
    temperature: float = 0.8
    max_new_tokens: int = 512
    umap_neighbors: int = 50   # Increased to encourage global structure/overlap
    umap_min_dist: float = 0.5 # Increased to spread clusters for overlap
    hide_ticks: bool = True
    sample_id: Optional[int] = None

def generate_sample(model, tokenizer, df, device, cfg):
    rng = np.random.default_rng(cfg.seed)
    stop_ids = tokenizer.encode("</think>", add_special_tokens=False)
    stopping = StoppingCriteriaList([StopOnSubsequence(stop_ids)])

    for attempt in range(cfg.max_retries):
        if cfg.sample_id is not None:
            idx = int(cfg.sample_id)
        else:
            idx = int(rng.integers(0, len(df)))

        print(f"Sampling index {idx}...")
        row = df.iloc[idx]
        base_prompt = construct_prompt(row, tokenizer)
        input_text = base_prompt + "<think>"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        try:
            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                stopping_criteria=stopping,
                use_cache=True
            )
            output = tokenizer.decode(gen[0], skip_special_tokens=False)
            cot = extract_cot_content(output)

            if cot and len(cot) > 100:
                print(f"Generated valid CoT (length {len(cot)}).")
                return row, cot, idx
        except Exception as e:
            print(f"Generation failed: {e}")

        if cfg.sample_id is not None:
            break
            
    return None, None, None

def run_visualization(model, tokenizer, df, cfg):
    device = next(model.parameters()).device
    row, cot_content, idx = generate_sample(model, tokenizer, df, device, cfg)
    if row is None:
        print("Could not generate valid sample.")
        return

    # Prompts
    base_prompt = construct_prompt(row, tokenizer)
    prompt_nohist = construct_no_history_prompt(row, tokenizer)

    # Embeddings
    print("Preparing embeddings...")
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vocab = tokenizer.get_vocab()
    
    # 2. Subspaces with Partial Overlap
    # SIDs
    sid_token_ids = [id for t, id in vocab.items() if t.startswith("<s_") and t.endswith(">")]
    np.random.seed(cfg.seed)
    sampled_sid_ids = np.random.choice(sid_token_ids, min(len(sid_token_ids), cfg.sample_size_sid), replace=False)
    X_sid = embeddings[sampled_sid_ids]
    
    # General/Text Tokens
    # Derived from Prompt + CoT to ensure semantic relevance and overlap with trajectory
    full_context = base_prompt + cot_content
    context_ids = list(set(tokenizer.encode(full_context, add_special_tokens=False)))
    # Exclude SIDs from 'General' to keep the definition clean (overlap comes from UMAP projection)
    context_ids = [i for i in context_ids if i not in sid_token_ids]
    
    # If not enough context tokens, sample random common words (ID range 1000-10000 typically text)
    if len(context_ids) < cfg.sample_size_other:
        needed = cfg.sample_size_other - len(context_ids)
        extras = np.random.choice(range(1000, 20000), needed)
        # Filter out SIDs just in case
        extras = [e for e in extras if e not in sid_token_ids]
        context_ids.extend(extras)
    
    sampled_other_ids = np.random.choice(context_ids, min(len(context_ids), cfg.sample_size_other), replace=False)
    X_other = embeddings[sampled_other_ids]

    # Trajectory
    print("Collecting trajectory...")
    cot_tokens = tokenizer.encode(cot_content, add_special_tokens=False)
    trajectory_vectors = []
    
    # Start: Prompt + <think> (State just before thinking)
    vec_start = get_last_token_hidden(model, tokenizer, base_prompt + "<think>", device)
    trajectory_vectors.append(vec_start)
    
    for i in range(cfg.stride, len(cot_tokens), cfg.stride):
        partial = tokenizer.decode(cot_tokens[:i])
        text = base_prompt + "<think>" + partial
        trajectory_vectors.append(get_last_token_hidden(model, tokenizer, text, device))
        
    vec_end = get_last_token_hidden(model, tokenizer, base_prompt + "<think>" + cot_content + "</think>", device)
    trajectory_vectors.append(vec_end)
    X_traj = np.array(trajectory_vectors)

    # References
    h_ref_nothink = get_last_token_hidden(model, tokenizer, base_prompt, device)
    # CoT Only: No History + CoT
    h_ref_cot_only = get_last_token_hidden(model, tokenizer, prompt_nohist + "<think>" + cot_content + "</think>", device)

    # Normalize
    X_sid = normalize(X_sid)
    X_other = normalize(X_other)
    X_traj = normalize(X_traj)
    h_ref_nothink = normalize(h_ref_nothink.reshape(1, -1))
    h_ref_cot_only = normalize(h_ref_cot_only.reshape(1, -1))

    # UMAP
    print("Running UMAP...")
    all_data = np.vstack([X_sid, X_other, X_traj, h_ref_nothink, h_ref_cot_only])
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_neighbors, 
        min_dist=cfg.umap_min_dist, 
        metric='cosine', 
        random_state=cfg.seed
    )
    emb = reducer.fit_transform(all_data)

    n_sid = len(X_sid)
    n_other = len(X_other)
    n_traj = len(X_traj)

    emb_sid = emb[:n_sid]
    emb_other = emb[n_sid:n_sid+n_other]
    emb_traj = emb[n_sid+n_other:n_sid+n_other+n_traj]
    emb_ref_nothink = emb[-2]
    emb_ref_cot_only = emb[-1]

    # Plotting
    set_paper_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # 2. Partial Overlap Visualization
    # Using 'teal' for SIDs and 'darksalmon' for Text - distinct but soft
    ax.scatter(emb_sid[:, 0], emb_sid[:, 1], c='teal', alpha=0.15, s=15, label='Semantic-ID Tokens', rasterized=True, edgecolors='none')
    ax.scatter(emb_other[:, 0], emb_other[:, 1], c='darksalmon', alpha=0.15, s=15, label='Text/General Tokens', rasterized=True, edgecolors='none')

    # 4. Trajectory Trend
    # Scatter points
    sc = ax.scatter(emb_traj[:, 0], emb_traj[:, 1], c=range(len(emb_traj)), cmap='plasma', s=25, alpha=0.9, zorder=5, label='Reasoning Trajectory')
    
    # Arrow: Start -> End
    # Calculating smoothed start/end points for stability
    head_k = max(1, len(emb_traj) // 5)
    start_pt = np.mean(emb_traj[:head_k], axis=0)
    end_pt = np.mean(emb_traj[-head_k:], axis=0)
    
    ax.annotate('', xy=end_pt, xytext=start_pt,
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5, mutation_scale=15),
                zorder=6)
    
    # 3. Reference Placement
    # No-Think (History) -> Should be near SIDs (Teal)
    ax.scatter(emb_ref_nothink[0], emb_ref_nothink[1], marker='D', s=80, c='teal', edgecolors='black', linewidth=1.2, zorder=10, label='Ref: Direct (No-Think)')
    
    # CoT-Only (Text) -> Should be near Text (Salmon)
    ax.scatter(emb_ref_cot_only[0], emb_ref_cot_only[1], marker='*', s=120, c='darksalmon', edgecolors='black', linewidth=1.2, zorder=10, label='Ref: CoT-Only')

    # Legend
    # Create proxy artist for Arrow
    from matplotlib.lines import Line2D
    arrow_proxy = Line2D([0], [0], color='black', lw=1.5, marker='>', markeredgecolor='black', label='Trajectory Trend')
    
    handles, labels = ax.get_legend_handles_labels()
    # Insert arrow proxy
    handles.append(arrow_proxy)
    labels.append("Trajectory Trend")
    
    ax.legend(handles, labels, loc='upper right', framealpha=0.9, fontsize=9)
    
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    if cfg.hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig(cfg.output_file, dpi=300)
    print(f"Saved {cfg.output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sample_id", type=int, default=None)
    args = parser.parse_args()

    # Config
    cfg = VizConfig(
        model_path="checkpoints/OneRec-1.7B",
        data_path="raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet",
        output_file="cot_umap.pdf",
        sample_id=args.sample_id
    )

    print(f"Loading model from {cfg.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading data from {cfg.data_path}...")
    df = pd.read_parquet(cfg.data_path)

    while True:
        run_visualization(model, tokenizer, df, cfg)
        if args.once:
            break
        try:
            if input("Generate another? (y/n) [y]: ").lower() == 'n': break
        except: break

if __name__ == "__main__":
    main()