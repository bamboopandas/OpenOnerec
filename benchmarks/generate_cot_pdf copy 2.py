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
# Plot style (paper-ready)
# ----------------------------
def set_paper_plot_style(font_scale: float = 1.0) -> None:
    """
    Paper-like matplotlib style:
    - consistent fonts & sizes
    - vector-friendly PDF embedding
    - clean axes
    """
    base = {
        "font.size": 10 * font_scale,
        "axes.titlesize": 11 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType (good for paper submission)
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.grid": False,
        "legend.frameon": False,
    }
    mpl.rcParams.update(base)


def beautify_axes(ax: mpl.axes.Axes, hide_ticks: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


# ----------------------------
# Stopping criteria: stop on "</think>"
# ----------------------------
class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # batch size assumed 1
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids):]
        stop = torch.tensor(self.stop_ids, device=tail.device, dtype=tail.dtype)
        return torch.all(tail == stop).item()


# ----------------------------
# Core helpers
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
    msgs = parse_messages(row["messages"])  # <- fixed: correct else branch
    system_msg = msgs[0].copy()
    system_msg["content"] = clean_content(system_msg.get("content", ""))

    user_content = clean_content(msgs[1].get("content", ""))
    # heuristic: keep only the tail instruction after the last "请"
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
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return h


def get_model_input_device(model) -> torch.device:
    # works well with device_map="auto" (takes the first param device)
    return next(iter(model.parameters())).device


def sample_sid_token_ids(tokenizer) -> List[int]:
    vocab = tokenizer.get_vocab()
    return [tid for tok, tid in vocab.items() if tok.startswith("<s_") and tok.endswith(">")]


# ----------------------------
# Generation & visualization
# ----------------------------
@dataclass
class VizConfig:
    model_path: str
    data_path: str
    output_file: str
    sample_size_sid: int = 500
    sample_size_other: int = 2000
    stride: int = 10
    seed: int = 42
    max_retries: int = 10
    temperature: float = 0.8
    max_new_tokens: int = 512
    umap_neighbors: int = 30
    umap_min_dist: float = 0.3
    connect_trajectory: bool = False
    hide_ticks: bool = True
    add_title: bool = False
    sample_id: Optional[int] = None  # fixed sample index if given


def generate_one_valid_cot(
    model, tokenizer, df: pd.DataFrame, device: torch.device, cfg: VizConfig
) -> Tuple[Optional[pd.Series], Optional[str], Optional[int]]:
    rng = np.random.default_rng(cfg.seed)

    stop_ids = tokenizer.encode("</think>", add_special_tokens=False)
    stopping = StoppingCriteriaList([StopOnSubsequence(stop_ids)])

    for attempt in range(cfg.max_retries):
        if cfg.sample_id is not None:
            idx = int(cfg.sample_id)
        else:
            idx = int(rng.integers(0, len(df)))

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
                use_cache=True,
            )
            output = tokenizer.decode(gen[0], skip_special_tokens=False)
            cot = extract_cot_content(output)

            if cot is not None and len(cot) > 50:
                return row, cot, idx

        except Exception as e:
            print(f"[warn] generation failed at attempt {attempt+1}: {e}")

        # if sample_id is fixed, don't retry other indices endlessly
        if cfg.sample_id is not None:
            break

    return None, None, None


def run_visualization(model, tokenizer, df: pd.DataFrame, cfg: VizConfig) -> None:
    device = get_model_input_device(model)

    row, cot_content, idx = generate_one_valid_cot(model, tokenizer, df, device, cfg)
    if row is None:
        print("[error] could not obtain a valid CoT sample.")
        return

    # ----- prompts -----
    base_prompt = construct_prompt(row, tokenizer)
    prompt_nohist = construct_no_history_prompt(row, tokenizer)

    # ----- embeddings -----
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    sid_token_ids = sample_sid_token_ids(tokenizer)

    rng = np.random.default_rng(cfg.seed)

    # semantic-id subspace
    sid_ids = np.array(sid_token_ids, dtype=np.int64)
    sampled_sid_ids = rng.choice(sid_ids, size=min(len(sid_ids), cfg.sample_size_sid), replace=False)
    X_sid = embeddings[sampled_sid_ids]

    # "other" token subspace: tokens appearing in prompt+cot (unique ids), then subsample
    full_context = base_prompt + cot_content
    context_ids = tokenizer.encode(full_context, add_special_tokens=False)
    context_ids = np.array(sorted(set(context_ids)), dtype=np.int64)
    context_ids = context_ids[~np.isin(context_ids, sid_ids)]

    if len(context_ids) == 0:
        # fallback if context token set is empty (rare)
        context_ids = rng.integers(1000, min(embeddings.shape[0], 5000), size=cfg.sample_size_other)

    sampled_other_ids = rng.choice(
        context_ids, size=min(len(context_ids), cfg.sample_size_other), replace=False
    )
    X_other = embeddings[sampled_other_ids]

    # ----- trajectory: hidden states along CoT tokens (stride sampling) -----
    cot_token_ids = tokenizer.encode(cot_content, add_special_tokens=False)
    # positions to sample: last token of prefix "<think>" (start) + every stride in cot
    trajectory_vectors = []

    # start vector: base_prompt + "<think>"
    vec_start = get_last_token_hidden(model, tokenizer, base_prompt + "<think>", device)
    trajectory_vectors.append(vec_start)

    # intermediate vectors: base_prompt + "<think>" + partial_cot  (use last token of partial_cot)
    for i in range(cfg.stride, len(cot_token_ids) + 1, cfg.stride):
        partial = tokenizer.decode(cot_token_ids[:i], skip_special_tokens=False)
        text = base_prompt + "<think>" + partial
        trajectory_vectors.append(get_last_token_hidden(model, tokenizer, text, device))

    # end vector: base_prompt + "<think>" + cot_content + "</think>"
    vec_end = get_last_token_hidden(model, tokenizer, base_prompt + "<think>" + cot_content + "</think>", device)
    trajectory_vectors.append(vec_end)

    X_traj = np.asarray(trajectory_vectors)

    # ----- references -----
    h_ref_nothink = get_last_token_hidden(model, tokenizer, base_prompt, device)
    h_ref_cot_only = get_last_token_hidden(model, tokenizer, prompt_nohist + "<think>" + cot_content + "</think>", device)

    # ----- normalize -----
    X_sid = normalize(X_sid)
    X_other = normalize(X_other)
    X_traj = normalize(X_traj)
    h_ref_nothink = normalize(h_ref_nothink.reshape(1, -1))
    h_ref_cot_only = normalize(h_ref_cot_only.reshape(1, -1))

    # ----- UMAP -----
    all_data = np.vstack([X_sid, X_other, X_traj, h_ref_nothink, h_ref_cot_only])
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    emb = reducer.fit_transform(all_data)

    n_sid = len(X_sid)
    n_other = len(X_other)
    n_traj = len(X_traj)

    emb_sid = emb[:n_sid]
    emb_other = emb[n_sid:n_sid + n_other]
    emb_traj = emb[n_sid + n_other:n_sid + n_other + n_traj]
    emb_ref_nothink = emb[-2]
    emb_ref_cot_only = emb[-1]

    # ----- Plot -----
    set_paper_plot_style(font_scale=1.0)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))  # paper-friendly aspect
    beautify_axes(ax, hide_ticks=cfg.hide_ticks)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Background subspaces: rasterize to keep PDF small (common paper trick)
    ax.scatter(
        emb_other[:, 0], emb_other[:, 1],
        s=8, alpha=0.18, label="Text/General tokens",
        linewidths=0, rasterized=True
    )
    ax.scatter(
        emb_sid[:, 0], emb_sid[:, 1],
        s=8, alpha=0.18, label="Semantic-ID tokens",
        linewidths=0, rasterized=True
    )

    # Trajectory (colored by step)
    steps = np.arange(len(emb_traj))
    sc = ax.scatter(
        emb_traj[:, 0], emb_traj[:, 1],
        c=steps, s=16, alpha=0.9,
        cmap="viridis", label="CoT trajectory",
        linewidths=0, zorder=5, rasterized=True
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.045)
    cbar.set_label("Step index")

    # Optional connecting line (some papers prefer this)
    if cfg.connect_trajectory and len(emb_traj) >= 2:
        ax.plot(emb_traj[:, 0], emb_traj[:, 1], alpha=0.35, zorder=4)

    # Trend arrow (robust direction via averaged head/tail)
    head_k = max(1, len(emb_traj) // 5)
    start_pt = emb_traj[:head_k].mean(axis=0)
    end_pt = emb_traj[-head_k:].mean(axis=0)
    ax.annotate(
        "", xy=end_pt, xytext=start_pt,
        arrowprops=dict(arrowstyle="->", lw=1.8),
        zorder=6
    )
    # Add a legend proxy for arrow
    ax.plot([], [], label="Trajectory trend", linewidth=1.8)

    # Reference points
    ax.scatter(
        emb_ref_nothink[0], emb_ref_nothink[1],
        marker="x", s=70, linewidths=1.6, label="No-think ref", zorder=10
    )
    ax.scatter(
        emb_ref_cot_only[0], emb_ref_cot_only[1],
        marker="^", s=55, linewidths=0, label="CoT-only ref", zorder=10
    )
    ax.plot(
        [emb_ref_nothink[0], emb_ref_cot_only[0]],
        [emb_ref_nothink[1], emb_ref_cot_only[1]],
        linestyle=":", alpha=0.6, zorder=3
    )

    # Legend: compact, paper-like
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    if cfg.add_title:
        ax.set_title(f"UMAP of normalized hidden states (sample {idx})")

    fig.tight_layout()
    fig.savefig(cfg.output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved: {cfg.output_file} (sample idx={idx})")


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/OneRec-1.7B")
    parser.add_argument("--data_path", type=str, default="raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet")
    parser.add_argument("--output", type=str, default="cot_umap.pdf")

    parser.add_argument("--sample_size_sid", type=int, default=500)
    parser.add_argument("--sample_size_other", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)

    parser.add_argument("--connect_trajectory", action="store_true")
    parser.add_argument("--show_ticks", action="store_true", help="Show ticks (default: hidden)")
    parser.add_argument("--title", action="store_true", help="Add a title (default: off for paper)")
    parser.add_argument("--once", action="store_true", help="Run only once and exit")
    parser.add_argument("--sample_id", type=int, default=None, help="Fix a specific row index for reproducibility")

    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = VizConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_file=args.output,
        sample_size_sid=args.sample_size_sid,
        sample_size_other=args.sample_size_other,
        stride=args.stride,
        seed=args.seed,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        connect_trajectory=args.connect_trajectory,
        hide_ticks=not args.show_ticks,
        add_title=args.title,
        sample_id=args.sample_id,
    )

    print(f"[info] loading tokenizer/model from: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"[info] loading data from: {cfg.data_path}")
    # only messages are used; reading fewer columns saves time
    df = pd.read_parquet(cfg.data_path, columns=["messages"])

    while True:
        run_visualization(model, tokenizer, df, cfg)
        if cfg.output_file.lower().endswith(".pdf"):
            pass

        if args.once:
            break

        try:
            user_input = input("Generate another sample? (y/n) [y]: ").strip().lower()
            if user_input == "n":
                break
        except EOFError:
            break


if __name__ == "__main__":
    main()
