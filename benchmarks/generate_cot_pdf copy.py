import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LinearRegression
import json
import sys
import argparse
from sklearn.preprocessing import normalize
import random

# Configuration
MODEL_PATH = "checkpoints/OneRec-1.7B"
DATA_PATH = "raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet"
OUTPUT_FILE = "cot_umap.pdf"
SAMPLE_SIZE_SID = 500
STRIDE = 10 

def get_hidden_state(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

def extract_cot_content(text):
    start_tag = "<think>"
    end_tag = "</think>"
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()
    return None

def clean_content(content):
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return content

def construct_prompt(row, tokenizer):
    messages_str = row['messages']
    if isinstance(messages_str, str):
        messages = json.loads(messages_str)
    else:
        messages = messages_str
    clean_messages = []
    for msg in messages:
        clean_messages.append({"role": msg["role"], "content": clean_content(msg.get('content', ''))})
    return tokenizer.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)

def construct_no_history_prompt(row, tokenizer):
    messages_str = row['messages']
    if isinstance(messages_str, str):
        msgs = json.loads(messages_str)
    else:
        msgs = msgs
    system_msg = msgs[0].copy()
    system_msg['content'] = clean_content(system_msg.get('content', ''))
    user_content = clean_content(msgs[1]['content'])
    if "请" in user_content:
        parts = user_content.split("请")
        instruction = "请" + parts[-1]
    else:
        instruction = user_content 
    msgs_nohist = [system_msg, {"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(msgs_nohist, tokenize=False, add_generation_prompt=True)

def generate_sample(model, tokenizer, df, device):
    # Randomly select a sample that produces valid CoT
    max_retries = 10
    for attempt in range(max_retries):
        idx = random.randint(0, len(df) - 1)
        row = df.iloc[idx]
        print(f"Sampling index {idx} (Attempt {attempt+1})...")
        
        base_prompt = construct_prompt(row, tokenizer)
        input_text = base_prompt + "<think>"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids, 
                    max_new_tokens=512, 
                    stop_strings=["</think>"], 
                    tokenizer=tokenizer,
                    do_sample=True,
                    temperature=0.8
                )
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            cot_content = extract_cot_content(output)
            
            if cot_content and len(cot_content) > 50: # Check for decent length
                print(f"Generated valid CoT (length {len(cot_content)}).")
                return row, cot_content, idx
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
            
    print("Could not generate valid CoT after multiple attempts.")
    return None, None, None

def run_visualization(model, tokenizer, df, device):
    row, cot_content, idx = generate_sample(model, tokenizer, df, device)
    if row is None:
        return

    base_prompt = construct_prompt(row, tokenizer)

    # 1. Prepare Subspace Embeddings
    print("Preparing token embeddings subspaces...")
    vocab = tokenizer.get_vocab()
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    
    sid_token_ids = [id for t, id in vocab.items() if t.startswith("<s_") and t.endswith(">")]
    
    # Sample random SIDs
    np.random.seed(random.randint(0, 10000)) # Random seed for variety
    sampled_sid_ids = np.random.choice(sid_token_ids, min(len(sid_token_ids), SAMPLE_SIZE_SID), replace=False)
    X_sid = embeddings[sampled_sid_ids]
    
    # Context Subspace
    full_context_text = base_prompt + cot_content
    context_ids = list(set(tokenizer.encode(full_context_text, add_special_tokens=False)))
    context_ids = [i for i in context_ids if i not in sid_token_ids]
    
    if len(context_ids) < 100:
        common_ids = np.random.choice([i for i in range(1000, 5000)], 200)
        context_ids.extend(common_ids)
        
    X_other = embeddings[context_ids]

    # 2. Trajectory
    print("Collecting trajectory...")
    cot_tokens = tokenizer.encode(cot_content, add_special_tokens=False)
    trajectory_vectors = []
    
    vec_start = get_hidden_state(model, tokenizer, base_prompt + "<think>", device)
    trajectory_vectors.append(vec_start)
    
    for i in range(STRIDE, len(cot_tokens), STRIDE):
        partial_cot = tokenizer.decode(cot_tokens[:i])
        text = base_prompt + "<think>" + partial_cot + "</think>"
        trajectory_vectors.append(get_hidden_state(model, tokenizer, text, device))
        
    vec_end = get_hidden_state(model, tokenizer, base_prompt + "<think>" + cot_content + "</think>", device)
    trajectory_vectors.append(vec_end)
    X_trajectory = np.array(trajectory_vectors)

    # 3. Reference Points
    h_ref_nothink = get_hidden_state(model, tokenizer, base_prompt, device)
    prompt_nohist = construct_no_history_prompt(row, tokenizer)
    h_ref_cot_only = get_hidden_state(model, tokenizer, prompt_nohist + "<think>" + cot_content + "</think>", device)

    # 4. Normalization
    X_sid = normalize(X_sid)
    X_other = normalize(X_other)
    X_trajectory = normalize(X_trajectory)
    h_ref_nothink = normalize(h_ref_nothink.reshape(1, -1))
    h_ref_cot_only = normalize(h_ref_cot_only.reshape(1, -1))

    # 5. UMAP
    all_data = np.vstack([X_sid, X_other, X_trajectory, h_ref_nothink, h_ref_cot_only])
    
    print(f"Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine')
    embedding = reducer.fit_transform(all_data)
    
    n_sid = len(X_sid)
    n_other = len(X_other)
    n_traj = len(X_trajectory)
    
    emb_sid = embedding[:n_sid]
    emb_other = embedding[n_sid:n_sid+n_other]
    emb_traj = embedding[n_sid+n_other:n_sid+n_other+n_traj]
    emb_ref_nothink = embedding[-2]
    emb_ref_cot_only = embedding[-1]

    # 6. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_sid[:, 0], emb_sid[:, 1], c='lightgreen', alpha=0.3, s=30, label='Subspace: Semantic IDs', edgecolors='none')
    plt.scatter(emb_other[:, 0], emb_other[:, 1], c='lightgrey', alpha=0.3, s=30, label='Subspace: Text/Reasoning', edgecolors='none')
    
    # Trajectory - Scatter only (no connecting lines)
    sc = plt.scatter(emb_traj[:, 0], emb_traj[:, 1], c=range(len(emb_traj)), 
                     cmap='viridis', s=25, label='Trajectory (CoT Steps)', zorder=5, alpha=0.8)
    
    # Calculate trend arrow (from average of first few points to average of last few points)
    # This provides a stable direction for the "trend"
    start_point = np.mean(emb_traj[:max(1, len(emb_traj)//5)], axis=0)
    end_point = np.mean(emb_traj[-max(1, len(emb_traj)//5):], axis=0)
    
    plt.annotate('', xy=end_point, xytext=start_point,
                 arrowprops=dict(arrowstyle='->', color='black', lw=2.5, mutation_scale=20),
                 label='CoT Evolution Trend')
    
    # Add a dummy line for the legend label of the arrow
    plt.plot([], [], 'k-', label='CoT Evolution Trend')
    
    plt.scatter(emb_ref_nothink[0], emb_ref_nothink[1], c='red', marker='x', s=100, label='No Think', zorder=10)
    plt.scatter(emb_ref_cot_only[0], emb_ref_cot_only[1], c='red', marker='^', s=100, label='CoT Only', zorder=10)
    plt.plot([emb_ref_nothink[0], emb_ref_cot_only[0]], [emb_ref_nothink[1], emb_ref_cot_only[1]], 'r:', alpha=0.5)

    plt.legend()
    plt.title(f'Sample {idx}: Normalized Hidden States')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run only once and exit")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    device = model.device

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    while True:
        run_visualization(model, tokenizer, df, device)
        if args.once:
            break
        try:
            user_input = input("Generate another sample? (y/n) [y]: ")
            if user_input.lower() == 'n':
                break
        except EOFError:
            break

if __name__ == "__main__":
    main()