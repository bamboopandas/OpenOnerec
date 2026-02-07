import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

def load_semantic_ids(model_path):
    added_tokens_path = os.path.join(model_path, "added_tokens.json")
    with open(added_tokens_path, 'r') as f:
        added_tokens = json.load(f)
    
    semantic_ids = []
    for token, token_id in added_tokens.items():
        if token.startswith("<s_c_"):
            semantic_ids.append(token_id)
    return sorted(list(set(semantic_ids)))

def convert_messages_format(messages):
    converted = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            converted.append({
                "role": msg.get("role"),
                "content": "".join(text_parts)
            })
        else:
            converted.append(msg)
    return converted

def main():
    model_path = "checkpoints/OneRec-1.7B"
    data_path = "raw_data/onerec_data/benchmark_data_1000/ad/ad_test_sample_1.parquet"
    template_path = "benchmarks/benchmark/tasks/v1_0/qwen3_soft_switch.jinja2"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    # Load custom chat template
    print(f"Loading custom chat template from {template_path}...")
    with open(template_path, 'r') as f:
        tokenizer.chat_template = f.read()

    # Load semantic IDs
    semantic_ids = set(load_semantic_ids(model_path))
    print(f"Loaded {len(semantic_ids)} semantic IDs.")

    # Load data
    data_path = "raw_data/onerec_data/benchmark_data_1000/ad/ad_test_sample_100.parquet"
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Limit to 50 samples
    num_samples = 50
    df = df.head(num_samples)

    captured_embeddings = {
        "Think On": [],
        "Think Off": []
    }

    # Helper to capture embeddings
    def run_inference(row, enable_thinking, label):
        messages_raw = row['messages']
        if isinstance(messages_raw, np.ndarray):
            messages_raw = messages_raw.tolist()
        if isinstance(messages_raw, str):
            try:
                messages_raw = json.loads(messages_raw)
            except json.JSONDecodeError:
                pass
        
        # Check if elements are strings that need parsing
        if isinstance(messages_raw, list) and len(messages_raw) > 0 and isinstance(messages_raw[0], str):
            try:
                messages_raw = [json.loads(m) if isinstance(m, str) else m for m in messages_raw]
            except:
                pass
        
        messages = convert_messages_format(messages_raw)

        # Prepare input
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            max_tokens = 2048 if enable_thinking else 512
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        generated_sequences = outputs.sequences[0]
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generated_sequences[input_len:]
        
        # Iterate through generated tokens
        for i, token_id in enumerate(generated_tokens):
            if token_id.item() in semantic_ids:
                if i >= len(outputs.hidden_states):
                    break

                last_layer_hidden = outputs.hidden_states[i][-1] # [batch, 1, hidden]
                
                # Apply final norm if it exists
                if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                    norm = model.model.norm
                elif hasattr(model, 'norm'):
                    norm = model.norm
                else:
                    norm = lambda x: x 
                
                normalized_hidden = norm(last_layer_hidden).squeeze(0).squeeze(0) # [hidden]
                
                captured_embeddings[label].append(normalized_hidden.detach().float().cpu().numpy())
                # Only capture the first semantic token
                break

    # Run for multiple samples
    print(f"Processing {len(df)} samples...")
    for idx, row in df.iterrows():
        print(f"Processing sample {idx}...")
        run_inference(row, True, "Think On")
        run_inference(row, False, "Think Off")


    # Visualization
    print("\nVisualizing...")
    
    # Load static embeddings
    embed_weight = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vocab_size = embed_weight.shape[0]
    
    semantic_indices = list(semantic_ids)
    all_indices = np.arange(vocab_size)
    is_semantic = np.isin(all_indices, semantic_indices)
    general_indices = all_indices[~is_semantic]
    
    # Sample general tokens for PCA fitting to keep it balanced? 
    # Or fit on all? Fitting on all is safer.
    
    print("Fitting PCA on static embeddings...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embed_weight)
    
    static_semantic_pca = pca_result[semantic_indices]
    
    # Sample general tokens for plotting
    if len(general_indices) > 20000:
        np.random.seed(42)
        sampled_general_indices = np.random.choice(general_indices, 20000, replace=False)
    else:
        sampled_general_indices = general_indices
    static_general_pca = pca_result[sampled_general_indices]
    
    plt.figure(figsize=(12, 10))
    
    # Plot static
    plt.scatter(static_general_pca[:, 0], static_general_pca[:, 1], alpha=0.1, s=1, c='lightgray', label='Static General')
    plt.scatter(static_semantic_pca[:, 0], static_semantic_pca[:, 1], alpha=0.3, s=5, c='blue', label='Static Semantic')
    
    # Plot dynamic
    colors = {"Think On": "red", "Think Off": "green"}
    for label, embeddings in captured_embeddings.items():
        if embeddings:
            embeddings_np = np.array(embeddings)
            # Transform using the same PCA
            projected = pca.transform(embeddings_np)
            plt.scatter(projected[:, 0], projected[:, 1], alpha=0.8, s=30, c=colors[label], label=f'Dynamic Semantic ({label})', marker='^')
        else:
            print(f"Warning: No embeddings captured for {label}")

    plt.title('Embedding Space: Static vs Inference (Think Mode)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    
    output_file = "inference_embedding_visualization.pdf"
    plt.savefig(output_file, format='pdf', dpi=300)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    main()
