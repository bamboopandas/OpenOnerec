import json
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os
import re

def main():
    model_path = "checkpoints/OneRec-1.7B"
    added_tokens_path = os.path.join(model_path, "added_tokens.json")
    safetensors_path = os.path.join(model_path, "model-00000-of-00001.safetensors")

    print(f"Loading added tokens from {added_tokens_path}...")
    with open(added_tokens_path, 'r') as f:
        added_tokens = json.load(f)

    # Identify semantic tokens
    # Assuming semantic tokens follow the pattern <s_c_...>
    semantic_ids = []
    for token, token_id in added_tokens.items():
        if token.startswith("<s_c_"):
            semantic_ids.append(token_id)
    
    semantic_ids = sorted(list(set(semantic_ids)))
    print(f"Found {len(semantic_ids)} semantic tokens.")

    print(f"Loading model weights from {safetensors_path}...")
    # Load only the embedding layer
    # Usually 'model.embed_tokens.weight' for Qwen/Llama
    try:
        weights = load_file(safetensors_path)
        if "model.embed_tokens.weight" in weights:
            embed_weight = weights["model.embed_tokens.weight"]
        elif "transformer.wte.weight" in weights: # potential alternative
            embed_weight = weights["transformer.wte.weight"]
        else:
            # List keys to debug
            print("Could not find embedding layer. Available keys:")
            for k in list(weights.keys())[:10]:
                print(k)
            return
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        return

    print(f"Embedding shape: {embed_weight.shape}")
    vocab_size, hidden_dim = embed_weight.shape

    # Convert to numpy
    embed_weight_np = embed_weight.float().numpy()

    # Separate semantic embeddings
    semantic_indices = np.array(semantic_ids)
    semantic_embeddings = embed_weight_np[semantic_indices]

    # For general tokens, we'll take the rest, or a sample
    # Let's exclude special tokens defined in added_tokens to be safe, 
    # but primarily we want the "original" vocab.
    # Usually original vocab is 0 to the start of added tokens, but let's just take everything that isn't a semantic token.
    
    all_indices = np.arange(vocab_size)
    # Create a boolean mask
    is_semantic = np.isin(all_indices, semantic_indices)
    general_indices = all_indices[~is_semantic]
    
    # If there are too many general tokens, we might want to sample them for visualization clarity
    # and performance (t-SNE is slow).
    # Let's verify counts.
    print(f"General tokens count: {len(general_indices)}")

    # Sampling for visualization
    # We want to show the distribution. If we have ~8000 semantic tokens, 
    # maybe we sample ~8000 general tokens or use all for PCA?
    # PCA is fast enough for 170k. t-SNE is not.
    # Let's do PCA first to see the global structure.
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    # Fit on a mix or all? Fitting on all is safer to see global variance.
    pca_result = pca.fit_transform(embed_weight_np)
    
    semantic_pca = pca_result[semantic_indices]
    
    # For general tokens, let's plot a subset to avoid overplotting if there are many
    # But for a paper-style figure, density matters.
    # Let's sample 10,000 general tokens for the plot if there are more than that.
    if len(general_indices) > 20000:
        np.random.seed(42)
        sampled_general_indices = np.random.choice(general_indices, 20000, replace=False)
    else:
        sampled_general_indices = general_indices
        
    general_pca = pca_result[sampled_general_indices]

    # Plotting
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Plot general tokens first (background)
    plt.scatter(general_pca[:, 0], general_pca[:, 1], alpha=0.3, s=1, c='gray', label='General Tokens')
    
    # Plot semantic tokens (foreground)
    plt.scatter(semantic_pca[:, 0], semantic_pca[:, 1], alpha=0.5, s=2, c='blue', label='Semantic Tokens')

    plt.title('Embedding Space Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    
    output_file = "embedding_visualization.pdf"
    plt.savefig(output_file, format='pdf', dpi=300)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    main()
