import os
import sys
import torch
import json
import argparse
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Append path to access benchmark modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from benchmark.tasks.v1_0.registry import get_loader
    from benchmark.console import console
except ImportError:
    # Fallback if run from root
    sys.path.append(os.path.join(os.getcwd(), 'benchmarks'))
    from benchmark.tasks.v1_0.registry import get_loader
    from benchmark.console import console

class SidStoppingCriteria(StoppingCriteria):
    def __init__(self, sid_begin_id):
        self.sid_begin_id = sid_begin_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last generated token is sid_begin_id
        if input_ids.shape[1] > 0:
            return (input_ids[:, -1] == self.sid_begin_id).all()
        return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="../raw_data/onerec_data/benchmark_data_1000")
    parser.add_argument("--output_file", type=str, default="drift_analysis_results.jsonl")
    parser.add_argument("--task", type=str, default="video")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_sid_tokens(tokenizer):
    sid_pattern = re.compile(r"^<s_[a-z]_\d+>$")
    sid_tokens = []
    sid_ids = []
    for token, id in tokenizer.get_vocab().items():
        if sid_pattern.match(token):
            sid_tokens.append(token)
            sid_ids.append(id)
    return sid_tokens, sorted(sid_ids)

def replace_sids_in_text(text, sid_tokens, seed):
    # Pattern: <|sid_begin|>...<|sid_end|>
    rng = random.Random(seed)
    
    def replacer(match):
        content = match.group(0)
        inner = content[len("<|sid_begin|>"):-len("<|sid_end|>")]
        # Count approximate SIDs by finding <s_...
        current_sids = re.findall(r"<s_[a-z]_\d+>", inner)
        count = len(current_sids)
        if count == 0:
            # Fallback for empty or malformed
            return content
            
        new_sids = rng.choices(sid_tokens, k=count)
        return "<|sid_begin|>".join(new_sids) + "<|sid_end|>"

    return re.sub(r"<\|sid_begin\|>.*?<\|sid_end\|>", replacer, text, flags=re.DOTALL)

def main():
    args = parse_args()
    setup_seed(args.seed)
    
    # Resolve model path if relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model_path):
        # Try relative to current working directory first, then relative to script
        if os.path.exists(os.path.join(os.getcwd(), args.model_path)):
            args.model_path = os.path.abspath(os.path.join(os.getcwd(), args.model_path))
        else:
            args.model_path = os.path.abspath(os.path.join(script_dir, args.model_path))

    if not os.path.exists(args.model_path):
        console.print(f"[red]Error: Model path does not exist: {args.model_path}[/red]")
        sys.exit(1)

    console.print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.eval()
    
    sid_tokens, sid_ids = get_sid_tokens(tokenizer)
    sid_ids_tensor = torch.tensor(sid_ids, device=model.device)
    console.print(f"Found {len(sid_ids)} SID tokens.")
    
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    
    console.print(f"Loading data for task: {args.task}")
    try:
        loader = get_loader(
            task_name=args.task,
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            enable_thinking=True 
        )
        data = loader.load_data(split="test", sample_size=args.limit)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    results = []
    
    # Prepare batch processing for metrics? 
    # For now, sample by sample loop is safer for logic.
    
    stopping_criteria = StoppingCriteriaList([SidStoppingCriteria(sid_begin_id)])

    for sample_id, sample in tqdm(data.items(), desc="Processing"):
        x = sample["prompt"]
        y_star_str = sample["ground_truth"]
        
        # --- 1. Generate Reasoning (r) ---
        inputs = tokenizer(x, return_tensors="pt").to(model.device)
        console.print(f"Sample {sample_id} input length: {inputs.input_ids.shape[1]}")
        
        # We generate until <|sid_begin|>
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, 
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode
        full_output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Robustly extract generated part
        # input x might have special tokens that decode differently?
        # Usually safe to slice tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        # Check if we stopped at <|sid_begin|>
        if "<|sid_begin|>" in generated_text:
            r = generated_text.split("<|sid_begin|>")[0]
        else:
            r = generated_text
            
        cot_len_tokens = len(generated_tokens) # Approx, includes split parts
        # Refine cot_len: count tokens in r
        r_tokens = tokenizer.tokenize(r)
        cot_len_tokens = len(r_tokens)
        
        # --- 2. Construct Contexts ---
        c_E_text = x
        c_C_text = x + r
        c_T_text = replace_sids_in_text(x, sid_tokens, args.seed + int(sample_id))
        
        # --- 3. Forward Pass for Metrics (A, B, JSD) ---
        # Inputs: context + <|sid_begin|>
        # Target: Next token logits
        prompts = [
            c_E_text + "<|sid_begin|>",
            c_C_text + "<|sid_begin|>", 
            c_T_text + "<|sid_begin|>"
        ]
        
        logits_list = []
        for p in prompts:
            p_inputs = tokenizer(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**p_inputs)
            
            # Last token logits
            last_logits = out.logits[0, -1, :]
            # Slice SID
            sid_logits = last_logits[sid_ids].float().cpu().numpy()
            logits_list.append(sid_logits)
            
            del p_inputs, out, last_logits
            torch.cuda.empty_cache()
            
        l_E, l_C, l_T = logits_list
        
        # Metrics
        delta = l_C - l_E
        d = l_T - l_E
        
        # Random Baseline
        d_rand = np.random.randn(*d.shape)
        
        # Cosine A
        norm_delta = np.linalg.norm(delta)
        norm_d = np.linalg.norm(d)
        
        if norm_delta > 0 and norm_d > 0:
            A = 1 - cosine(delta, d)
        else:
            A = 0.0
            
        # Projection B
        if norm_d > 1e-9:
            B = abs(np.dot(delta, d)) / norm_d
        else:
            B = 0.0
            
        # Random B
        norm_d_rand = np.linalg.norm(d_rand)
        if norm_d_rand > 1e-9:
            B_rand = abs(np.dot(delta, d_rand)) / norm_d_rand
        else:
            B_rand = 0.0
            
        # JSD
        def safe_softmax(logits):
            # subtraction for stability
            logits = logits - np.max(logits)
            exp_l = np.exp(logits)
            return exp_l / np.sum(exp_l)
            
        p_C = safe_softmax(l_C)
        p_T = safe_softmax(l_T)
        
        jsd_val = jensenshannon(p_C, p_T) # Base e default. 
        # JSD is usually squared, but let's keep the distance (0 to 1).
        
        # --- 4. Delta NLL ---
        # We need to evaluate NLL of y* given context.
        # Target: y* (SIDs).
        # We assume y* is just the SIDs sequence.
        # We construct inputs: context + y*
        # But wait, we used context + <|sid_begin|> for logits.
        # y* usually follows <|sid_begin|>.
        # So we should append <|sid_begin|> + y*.
        
        # Prepare targets
        # Tokenize y* (SIDs)
        # We assume y_star_str is just "<s_...><s_...>"
        target_ids = tokenizer(y_star_str, add_special_tokens=False).input_ids
        target_tensor = torch.tensor(target_ids, device=model.device)
        
        def compute_nll(context_text):
            # Input: context + <|sid_begin|> + y*
            full_text = context_text + "<|sid_begin|>" # Start with prefix
            # We need to be careful with tokenization boundaries.
            # safe: tokenize prefix, cat target.
            prefix_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            
            full_ids = prefix_ids + target_ids
            input_tensor = torch.tensor([full_ids], device=model.device)
            labels = input_tensor.clone()
            
            # Mask prefix
            labels[:, :len(prefix_ids)] = -100
            
            with torch.no_grad():
                loss = model(input_tensor, labels=labels).loss
            return loss.item()
            
        nll_C = compute_nll(c_C_text)
        nll_E = compute_nll(c_E_text)
        delta_nll = nll_C - nll_E
        
        results.append({
            "sample_id": sample_id,
            "cot_len": cot_len_tokens,
            "A": float(A),
            "B": float(B),
            "B_rand": float(B_rand),
            "JSD": float(jsd_val),
            "delta_nll": delta_nll,
            # "debug_r": r[:100] # Optional
        })
        
    # --- Save & Summarize ---
    console.print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    df = pd.DataFrame(results)
    if len(df) > 1:
        console.print("\nSummary Spearman Correlations with Delta NLL:")
        metrics = ["A", "B", "B_rand", "JSD", "cot_len"]
        for metric in metrics:
            if metric in df.columns:
                corr, p = spearmanr(df[metric], df["delta_nll"])
                console.print(f"{metric}: {corr:.4f} (p={p:.4f})")
    else:
        console.print("Not enough samples for correlation.")

if __name__ == "__main__":
    main()
