import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import random
import numpy as np
from tqdm import tqdm
import re
import argparse
import os
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader, Dataset

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Constants
SID_VOCAB_SIZE = 8192
SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"

class DriftDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        self.data = []
        df = pd.read_parquet(data_path)
        if max_samples:
            df = df.head(max_samples)
        
        for _, row in df.iterrows():
            try:
                messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
                prompt = messages[-1]['content'][0]['text']
                # Ensure prompt ends with proper trigger if not already
                if not prompt.strip().endswith("预测："):
                     pass # Assuming the prompt is already well-formed or we append later
                
                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                ground_truth = metadata.get('answer', '')
                
                if prompt and ground_truth:
                    self.data.append({
                        'id': str(row.get('uuid', _)),
                        'prompt': prompt,
                        'ground_truth': ground_truth
                    })
            except Exception as e:
                # print(f"Error parsing row: {e}")
                continue
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_sid_token_string(layer, index):
    if layer == 0:
        return f"<s_a_{index}>"
    elif layer == 1:
        return f"<s_b_{index}>"
    elif layer == 2:
        return f"<s_c_{index}>"
    return ""

def replace_sids_with_random(text, tokenizer, sid_ranges):
    # sid_ranges: list of token IDs for layer 0, 1, 2
    # We construct random strings.
    # Regex to find <|sid_begin|>...<|sid_end|>
    pattern = r"<\|sid_begin\|>(.*?)<\|sid_end\|>"
    
    def replacer(match):
        # We replace the content with random tokens
        # Assuming 3 layers structure <s_a_..><s_b_..><s_c_..>
        # We just generate 3 random indices
        idx_a = random.randint(0, SID_VOCAB_SIZE - 1)
        idx_b = random.randint(0, SID_VOCAB_SIZE - 1)
        idx_c = random.randint(0, SID_VOCAB_SIZE - 1)
        
        return f"{SID_BEGIN_TOKEN}{get_sid_token_string(0, idx_a)}{get_sid_token_string(1, idx_b)}{get_sid_token_string(2, idx_c)}{SID_END_TOKEN}"

    return re.sub(pattern, replacer, text)

def calculate_nll(model, tokenizer, prefix, target_sequence, device):
    # prefix: context
    # target_sequence: the sequence to score (the answer)
    # We want NLL(target | prefix)
    
    # Encode
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(device)
    target_ids = tokenizer.encode(target_sequence, add_special_tokens=False, return_tensors='pt').to(device)
    
    input_ids = torch.cat([prefix_ids, target_ids], dim=1)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # Shift logits and labels
    # We are interested in the log probs of target_ids given the history
    # The first token of target_ids is predicted by the last token of prefix_ids
    
    start_idx = prefix_ids.shape[1] - 1
    end_idx = input_ids.shape[1] - 1
    
    relevant_logits = logits[0, start_idx:end_idx, :]
    relevant_labels = input_ids[0, start_idx+1:]
    
    loss = F.cross_entropy(relevant_logits, relevant_labels, reduction='sum')
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1) # Inference is complex, batch=1 is safer for implementation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # Determine SID Vocabulary IDs (Layer 0)
    # Based on check, <s_a_0> is 151669, <s_a_8191> is 159860
    sid_start_id = tokenizer.convert_tokens_to_ids("<s_a_0>")
    sid_end_id = tokenizer.convert_tokens_to_ids(f"<s_a_{SID_VOCAB_SIZE-1}>")
    print(f"SID Vocab Range: {sid_start_id} - {sid_end_id}")
    
    sid_vocab_indices = torch.arange(sid_start_id, sid_end_id + 1).to(device)

    dataset = DriftDataset(args.data_file, args.max_samples)
    print(f"Loaded {len(dataset)} samples.")

    results = []
    
    # Sid Begin Token ID
    sid_begin_id = tokenizer.convert_tokens_to_ids(SID_BEGIN_TOKEN)

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = sample['prompt']
        ground_truth = sample['ground_truth']
        
        # 1. E-context: x
        # Ensure it prompts for prediction
        # Check if "预测：" is at the end. The user message from parquet seems to end with it.
        # But we need to feed "<|sid_begin|>" to get the next token.
        
        # Prepare inputs
        # x + <|sid_begin|>
        # Note: We need to be careful with tokenization.
        # It's better to tokenize x, then append sid_begin_id.
        
        input_ids_x = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        
        # 2. C-context: x + r
        # Generate r.
        # We assume "Think Mode" is triggered or we just generate until <|sid_begin|>
        # To be safe and consistent with "Think Mode", we can inject <think> if the model expects it, 
        # but the instructions say "Generate a free-text reasoning process r from x using the same model".
        # We'll just generate until <|sid_begin|>. 
        
        # Generate r
        with torch.no_grad():
            # Generate until <|sid_begin|> or max tokens
            gen_output = model.generate(
                input_ids_x, 
                max_new_tokens=512,
                do_sample=False, # Fixed decoding parameters (Greedy)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sid_begin_id, # Stop at sid_begin
                # We want to capture what comes BEFORE sid_begin.
                # Actually, if we set eos_token_id to sid_begin_id, it will stop *including* it or *before* it?
                # Usually it includes it.
            )
            
        generated_ids = gen_output[0]
        # Extract r. r is everything after input_ids_x until (and including?) the newly generated part.
        # Check if the last token is sid_begin_id
        has_sid_begin = False
        if generated_ids[-1].item() == sid_begin_id:
            has_sid_begin = True
            r_ids = generated_ids[len(input_ids_x[0]):] # Includes sid_begin at the end
        else:
            # Did not generate sid_begin within limit.
            # We treat the whole generation as r and append sid_begin manually for measurement.
            r_ids = torch.cat([generated_ids[len(input_ids_x[0]):], torch.tensor([sid_begin_id], device=device)])
        
        # r text for length calc
        r_text = tokenizer.decode(r_ids[:-1]) # Exclude sid_begin for text length
        cot_len = len(r_ids) - 1 # Exclude sid_begin
        
        # Context C input ids (for logits): x + r + <|sid_begin|>
        # If r_ids already has sid_begin at end, input is just x + r_ids
        input_ids_c = torch.cat([input_ids_x[0], r_ids]).unsqueeze(0)
        
        # Context E input ids (for logits): x + <|sid_begin|>
        input_ids_e = torch.cat([input_ids_x[0], torch.tensor([sid_begin_id], device=device)]).unsqueeze(0)
        
        # 3. T-context: x' + <|sid_begin|>
        # Replace SIDs in prompt
        prompt_t = replace_sids_with_random(prompt, tokenizer, None)
        input_ids_x_t = tokenizer.encode(prompt_t, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids_t = torch.cat([input_ids_x_t[0], torch.tensor([sid_begin_id], device=device)]).unsqueeze(0)
        
        # Forward Passes for Logits
        def get_sid_logits(input_ids):
            with torch.no_grad():
                outputs = model(input_ids)
                last_token_logits = outputs.logits[0, -1, :]
                # Restrict to SID vocab
                return last_token_logits[sid_vocab_indices]

        l_E = get_sid_logits(input_ids_e)
        l_C = get_sid_logits(input_ids_c)
        l_T = get_sid_logits(input_ids_t)
        
        # Measurements
        # delta = l_C - l_E
        delta = l_C - l_E
        # d = l_T - l_E
        d = l_T - l_E
        
        # Convert to numpy for calculations
        delta_np = delta.float().cpu().numpy()
        d_np = d.float().cpu().numpy()
        
        # A = cosine(delta, d)
        def cosine_sim(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
            
        A = cosine_sim(delta_np, d_np)
        
        # B = || proj_d(delta) || = | dot(delta, d) / ||d|| |
        # Wait, projection vector is (dot(delta, d) / dot(d, d)) * d
        # Magnitude is | dot(delta, d) / ||d|| |
        # = | dot(delta, d/||d||) | = | component of delta along d |
        norm_d = np.linalg.norm(d_np)
        B = abs(np.dot(delta_np, d_np) / norm_d) if norm_d > 0 else 0.0
        
        # Random Baseline
        # Random vector of same size
        d_rand = np.random.randn(len(d_np))
        norm_d_rand = np.linalg.norm(d_rand)
        B_rand = abs(np.dot(delta_np, d_rand) / norm_d_rand) if norm_d_rand > 0 else 0.0
        
        # JSD
        p_C = F.softmax(l_C.float(), dim=0).cpu().numpy()
        p_T = F.softmax(l_T.float(), dim=0).cpu().numpy()
        jsd_val = jensenshannon(p_C, p_T, base=2.0) ** 2 # JSD is square of metric usually, or direct. scipy returns distance (sqrt of JSD).
        # Definition varies. Usually JSD is in [0, 1]. Jensen-Shannon distance is sqrt(JSD).
        # "JSD = JSD(softmax(l_C), softmax(l_T))". Let's assume standard JSD (divergence).
        
        # 4. Correctness Proxy Metric
        # NLL(y* | c_C) - NLL(y* | c_E)
        # y* is ground_truth (sequence of SIDs)
        
        # We need to construct the full text for NLL calculation.
        # c_C text: we have input_ids_c (which ends with <|sid_begin|>)
        # But calculate_nll expects prefix string or ids.
        # We can pass IDs.
        
        # Prefix E: input_ids_e (ends with <|sid_begin|>)
        # Prefix C: input_ids_c (ends with <|sid_begin|>)
        
        # Wait, calculate_nll needs to handle the fact that input_ids already contain the prefix.
        # Let's adapt calculate_nll to take input_ids directly.
        
        def calculate_nll_ids(prefix_ids, target_ids):
            # prefix_ids: [1, seq_len]
            # target_ids: [1, target_len]
            
            # Combine
            input_ids = torch.cat([prefix_ids, target_ids], dim=1)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            # Labels
            # We predict tokens from prefix_len to end
            # Logits index: prefix_len - 1 (predicts first token of target)
            start_idx = prefix_ids.shape[1] - 1
            
            # logits: [1, seq_len, vocab]
            # We need logits for positions: start_idx, ..., end-1
            # Labels are: input_ids[0, start_idx+1 : ]
            
            relevant_logits = logits[0, start_idx : -1, :]
            relevant_labels = input_ids[0, start_idx + 1 :]
            
            loss = F.cross_entropy(relevant_logits, relevant_labels, reduction='sum')
            return loss.item()

        # Tokenize target y*
        # y* is in `ground_truth`. It is a string "<|sid_begin|>...".
        # But our prefix already ends with `<|sid_begin|>`. 
        # The ground_truth string from metadata might START with `<|sid_begin|>`. 
        # If so, we should remove the first `<|sid_begin|>` token from target because prefix has it.
        # OR, we construct prefix WITHOUT the last `<|sid_begin|>` and let target have it.
        
        # Let's see ground_truth format: "<|sid_begin|><s_a_...>"
        # input_ids_e ends with `<|sid_begin|>`. 
        # So we should strip the first `<|sid_begin|>` from ground_truth tokenization?
        # Or just tokenize ground_truth and check.
        
        target_ids = tokenizer.encode(ground_truth, return_tensors='pt', add_special_tokens=False).to(device)
        
        if target_ids[0, 0].item() == sid_begin_id:
            # Remove leading sid_begin because prefix has it
            target_ids = target_ids[:, 1:]
            
        nll_C = calculate_nll_ids(input_ids_c, target_ids)
        nll_E = calculate_nll_ids(input_ids_e, target_ids)
        
        delta_nll = nll_C - nll_E
        
        results.append({
            "sample_id": sample['id'],
            "cot_len": cot_len,
            "A": float(A),
            "B": float(B),
            "B_rand": float(B_rand),
            "JSD": float(jsd_val),
            "delta_nll": float(delta_nll)
        })
        
        if i % 10 == 0:
            # Intermediate save
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} samples. Results saved to {args.output_file}")
    
    # Correlation Summary
    df_res = pd.DataFrame(results)
    print("\nSpearman Correlations with Delta NLL:")
    for metric in ["cot_len", "A", "B", "B_rand", "JSD"]:
        corr = df_res[metric].corr(df_res['delta_nll'], method='spearman')
        print(f"{metric}: {corr:.4f}")

if __name__ == "__main__":
    main()
