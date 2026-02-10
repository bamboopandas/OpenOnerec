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

# Configuration
MODEL_PATH = "../checkpoints/OneRec-1.7B"
DATA_PATH = "../raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet"
OUTPUT_FILE = "cot_umap.pdf"
SAMPLE_SIZE_SUBSPACES = 30  # Number of samples to define the "subspaces"
STRIDE = 5  # Step size for CoT trajectory

def get_hidden_state(model, tokenizer, text, device):
    """
    Get the last hidden state of the last token.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last layer hidden state, last token: (Batch, Seq, Hidden) -> (Hidden,)
    last_hidden_state = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return last_hidden_state

def extract_cot_content(text):
    """
    Extract content between <think> and </think>.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()
    return None

def construct_prompt(row, tokenizer):
    """
    Construct the full prompt using chat template from the 'messages' column.
    """
    messages_str = row['messages']
    if isinstance(messages_str, str):
        messages = json.loads(messages_str)
    else:
        messages = messages_str # Assume it's already a list
        
    # Preprocess messages: Convert content list to string if necessary
    clean_messages = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            # Extract text from list of dicts
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            clean_content = "".join(text_parts)
        else:
            clean_content = content
            
        clean_messages.append({"role": msg["role"], "content": clean_content})
        
    # Apply chat template
    # We want to generate the answer, so add_generation_prompt=True
    text = tokenizer.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
    return text

def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = model.device

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Select a subset for subspace definition
    if len(df) > SAMPLE_SIZE_SUBSPACES:
        subspace_samples = df.iloc[:SAMPLE_SIZE_SUBSPACES]
    else:
        subspace_samples = df
        
    print("Collecting subspace vectors...")
    vectors_nothink = []
    vectors_cot_end = []
    
    # Collection for Subspaces
    for idx, row in subspace_samples.iterrows():
        try:
            prompt = construct_prompt(row, tokenizer)
        except Exception as e:
            print(f"Skipping sample {idx} due to prompt error: {e}")
            continue
        
        # 1. No Thinking State (Subspace A)
        h_nothink = get_hidden_state(model, tokenizer, prompt, device)
        vectors_nothink.append(h_nothink)
        
        # 2. Full CoT State (Subspace B)
        # We append <think> to prompt to force thinking
        input_text = prompt + "<think>"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            # Generate until </think>
            generated_ids = model.generate(
                input_ids, 
                max_new_tokens=256, 
                stop_strings=["</think>"], 
                tokenizer=tokenizer,
                do_sample=False # Greedy for deterministic subspaces
            )
        
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        # Construct input for hidden state: Prompt + CoT (including tags)
        
        h_cot = get_hidden_state(model, tokenizer, full_text, device)
        vectors_cot_end.append(h_cot)

    X_nothink = np.array(vectors_nothink)
    X_cot_end = np.array(vectors_cot_end)
    
    print("Collecting trajectory for Sample 0...")
    # Pick one sample for trajectory
    target_row = subspace_samples.iloc[0]
    target_prompt = construct_prompt(target_row, tokenizer)
    
    # Generate long CoT for this sample to get a good trajectory
    input_text = target_prompt + "<think>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=512, 
            stop_strings=["</think>"], 
            tokenizer=tokenizer,
            do_sample=True, # Sample to get a potentially longer/richer CoT
            temperature=0.7
        )
    
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    cot_content = extract_cot_content(full_output)
    
    if not cot_content:
        print("Failed to generate CoT for trajectory sample. Using raw generation.")
        # Try to find where <think> starts
        if "<think>" in full_output:
            cot_content = full_output.split("<think>")[-1].replace("</think>", "").strip()
        else:
            cot_content = "Default CoT content"
    
    print(f"CoT Length: {len(cot_content)} chars")
    
    # Tokenize CoT content
    cot_tokens = tokenizer.encode(cot_content, add_special_tokens=False)
    
    trajectory_vectors = []
    
    # Collect trajectory
    # Start with empty CoT (just <think>)
    # Then add tokens
    
    # Initial point (just <think>)
    text_base = target_prompt + "<think>"
    trajectory_vectors.append(get_hidden_state(model, tokenizer, text_base, device))
    
    for i in range(1, len(cot_tokens), STRIDE):
        partial_cot = tokenizer.decode(cot_tokens[:i])
        text = text_base + partial_cot
        trajectory_vectors.append(get_hidden_state(model, tokenizer, text, device))
        
    X_trajectory = np.array(trajectory_vectors)
    
    # Reference Points for this sample
    print("Collecting reference points...")
    # 1. No Thinking (Personalized/Intuitive)
    h_ref_nothink = get_hidden_state(model, tokenizer, target_prompt, device)
    
    # 2. CoT Only (General/Reasoned?)
    # "masking out the previous prompt history"
    # We construct a prompt that only asks to reason based on internal knowledge?
    # Or just feed "<think> CoT </think>" as the context?
    # Usually "CoT Only" means we rely on the reasoning path without the specific user history context influencing the *final* decision directly (other than through CoT).
    # But technically, if we feed "<think> CoT </think>" to the model, it has no context to predict the ad.
    # UNLESS the CoT itself contains the necessary summary of user history.
    # The prompt says: "visualizing the output logit when using only the chain of thought for this data (masking out the previous prompt history, keeping everything else the same)"
    # This implies the input should be `System Promopt + <think> CoT </think> + Answer Generation Prompt`.
    # Removing "User History".
    # I need to parse `messages` to find the System Prompt and remove the User History part of the User message?
    # In the sample:
    # System: "You are a recommender..."
    # User: "User history is... Please predict..."
    # I should replace User content with just "Please predict..."?
    # And then insert the CoT.
    
    # Construct "No History" Prompt
    messages_str = target_row['messages']
    if isinstance(messages_str, str):
        msgs = json.loads(messages_str)
    else:
        msgs = messages_str
        
    # Helper to clean message content
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

    # Clean system message
    system_msg = msgs[0].copy()
    system_msg['content'] = clean_content(system_msg.get('content', ''))

    # Extract user message text safely
    user_content = msgs[1]['content']
    user_msg = clean_content(user_content)

    # Split by "Please predict" (Chinese: 请根据用户的观看和广告点击历史，预测用户接下来可能点击的广告视频。)
    # The history is before that.
    # Heuristic: Keep only the instruction at the end.
    if "请" in user_msg:
        parts = user_msg.split("请")
        instruction = "请" + parts[-1]
    else:
        instruction = user_msg # Fallback
        
    msgs_nohist = [system_msg, {"role": "user", "content": instruction}]
    
    # We can't use construct_prompt directly because we modified the messages list structure manually
    # But tokenizer.apply_chat_template handles list of dicts with string content fine (as we ensured content is string)
    prompt_nohist = tokenizer.apply_chat_template(msgs_nohist, tokenize=False, add_generation_prompt=True)
    
    # Full input: NoHistPrompt + <think> + CoT + </think>
    cot_only_text = prompt_nohist + "<think>" + cot_content + "</think>"
    h_ref_cot_only = get_hidden_state(model, tokenizer, cot_only_text, device)
    
    # Combine all for UMAP
    # Labels: 0=Subspace A (NoThink), 1=Subspace B (CoT), 2=Trajectory, 3=RefNoThink, 4=RefCoTOnly
    all_data = np.vstack([
        X_nothink, 
        X_cot_end, 
        X_trajectory, 
        h_ref_nothink.reshape(1, -1), 
        h_ref_cot_only.reshape(1, -1)
    ])
    
    print(f"Running UMAP on {len(all_data)} vectors...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(all_data)
    
    # Split back
    n_sub = len(X_nothink)
    n_traj = len(X_trajectory)
    
    emb_nothink = embedding[:n_sub]
    emb_cot = embedding[n_sub:2*n_sub]
    emb_traj = embedding[2*n_sub:2*n_sub+n_traj]
    emb_ref_nothink = embedding[-2]
    emb_ref_cot_only = embedding[-1]
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot Subspaces
    plt.scatter(emb_nothink[:, 0], emb_nothink[:, 1], c='blue', alpha=0.3, label='Intuitive Subspace (No Think)', s=30)
    plt.scatter(emb_cot[:, 0], emb_cot[:, 1], c='red', alpha=0.3, label='Reasoned Subspace (Full CoT)', s=30)
    
    # Plot Trajectory
    plt.plot(emb_traj[:, 0], emb_traj[:, 1], c='green', alpha=0.7, linewidth=2, label='CoT Trajectory')
    plt.scatter(emb_traj[:, 0], emb_traj[:, 1], c='green', s=10)
    
    # Fit Line to Trajectory
    reg = LinearRegression().fit(emb_traj[:, 0].reshape(-1, 1), emb_traj[:, 1])
    x_range = np.linspace(emb_traj[:, 0].min(), emb_traj[:, 0].max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_pred, 'k--', alpha=0.5, label='Fitted Line')
    
    # Plot Reference Points
    plt.scatter(emb_ref_nothink[0], emb_ref_nothink[1], c='cyan', marker='*', s=200, edgecolors='black', label='Ref: No Think (Hist Only)')
    plt.scatter(emb_ref_cot_only[0], emb_ref_cot_only[1], c='magenta', marker='*', s=200, edgecolors='black', label='Ref: CoT Only (No Hist)')
    
    # Connect References
    plt.plot([emb_ref_nothink[0], emb_ref_cot_only[0]], [emb_ref_nothink[1], emb_ref_cot_only[1]], 'k:', alpha=0.5, label='Ref Direction')

    plt.title('CoT Trajectory & Subspace Shift')
    plt.legend()
    plt.tight_layout()
    
    print(f"Saving plot to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
