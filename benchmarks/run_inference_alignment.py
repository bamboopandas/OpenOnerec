import os
import sys
import argparse
import json
import random
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional

# Add pretrain to path to load Qwen3
# benchmarks/run_inference_alignment.py -> ../pretrain
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pretrain"))
try:
    from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config
    MODEL_CLASS = Qwen3ForCausalLM
except ImportError:
    print("Warning: Could not import Qwen3 from source. Falling back to AutoModel (might fail).")
    MODEL_CLASS = AutoModelForCausalLM

# Constants based on inspection
SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"

class InferenceAligner:
    def __init__(self, model_path, device="cuda", dtype="bfloat16"):
        self.device = device
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = MODEL_CLASS.from_pretrained(
            model_path, 
            torch_dtype=self.dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Resolve token IDs
        self.sid_begin_id = self.tokenizer.convert_tokens_to_ids(SID_BEGIN_TOKEN)
        self.sid_end_id = self.tokenizer.convert_tokens_to_ids(SID_END_TOKEN)
        self.think_start_id = self.tokenizer.convert_tokens_to_ids(THINK_START_TOKEN)
        self.think_end_id = self.tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        
        print(f"Token IDs: sid_begin={self.sid_begin_id}, sid_end={self.sid_end_id}")

    def parse_sid_spans(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Identify (start, end) indices of history spans <|sid_begin|>...<|sid_end|>. 
        """
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.tolist()
        else:
            ids = input_ids
            
        spans = []
        start_idx = -1
        for i, token_id in enumerate(ids):
            if token_id == self.sid_begin_id:
                start_idx = i
            elif token_id == self.sid_end_id:
                if start_idx != -1:
                    spans.append((start_idx, i)) # inclusive
                    start_idx = -1
        return spans

    def mask_span_in_prompt(self, input_ids: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
        """
        Mask a span by removing it.
        """
        start, end = span
        # Keep everything before start and after end
        # Remove [start, end]
        prefix = input_ids[:start]
        suffix = input_ids[end+1:]
        return torch.cat([prefix, suffix])

    def mask_multiple_spans(self, input_ids: torch.Tensor, spans_to_remove: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Remove multiple spans from input_ids. 
        Assumes spans are non-overlapping and sorted by start index.
        """
        # We need to be careful with indices shifting if we remove one by one.
        # Strategy: Create a mask or rebuild the tensor.
        keep_mask = torch.ones(input_ids.shape[0], dtype=torch.bool, device=self.device)
        for start, end in spans_to_remove:
            keep_mask[start : end + 1] = False
        
        return input_ids[keep_mask]

    def generate_rationale(self, input_ids, max_tokens=512):
        """
        Generate CoT rationale.
        Force generation to start with <think> if not present, and stop at </think>.
        """
        curr_input = input_ids.clone()
        if curr_input[0, -1] != self.think_start_id:
             curr_input = torch.cat([curr_input, torch.tensor([[self.think_start_id]], device=self.device)], dim=1)
        
        with torch.no_grad():
            outputs = self.model.generate(
                curr_input,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.think_end_id
            )
        
        # Extract the new tokens
        new_tokens = outputs[0, input_ids.shape[1]:]
        
        # Decode
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return text, new_tokens

    def generate_items(self, input_ids, max_new_tokens=128):
        """
        Generate items (No-Think or after Think).
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id 
            )
        new_tokens = outputs[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return text

    def extract_items_from_text(self, text):
        """
        Parse <|sid_begin|>...<|sid_end|> blocks.
        Return list of item strings (including tags).
        """
        import re
        pattern = r"(<\|sid_begin\|>.*?<\|sid_end\|>)"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def encode_item(self, item_str):
        """
        Convert item string to token IDs.
        """
        return self.tokenizer.encode(item_str, add_special_tokens=False)

    def get_item_scores(self, context_ids: torch.Tensor, candidates: List[List[int]]) -> torch.Tensor:
        """
        Compute sum log p(item_tokens | context) for each candidate.
        """
        scores = []
        
        # We can batch this if needed, but sequential is safer for VRAM for now.
        for cand_tokens in candidates:
            cand_tensor = torch.tensor([cand_tokens], device=self.device)
            # Input: context + cand
            full_input = torch.cat([context_ids, cand_tensor], dim=1)
            
            with torch.no_grad():
                outputs = self.model(full_input)
                logits = outputs.logits # (1, L, V)
            
            # Shift logits and labels
            # logits[i] predicts input[i+1]
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = full_input[0, 1:].contiguous()
            
            # Identify where the candidate part starts in the shifted sequences
            # candidate starts at index len(context_ids) in full_input
            # In shift_labels, this corresponds to index len(context_ids) - 1.
            start_idx = context_ids.shape[1] - 1
            
            cand_logits = shift_logits[start_idx : start_idx + len(cand_tokens)]
            cand_labels = shift_labels[start_idx : start_idx + len(cand_tokens)]
            
            # Cross entropy (reduction=sum) gives -log_prob
            loss = F.cross_entropy(cand_logits, cand_labels, reduction='sum')
            scores.append(-loss.item())
            
        return torch.tensor(scores, device=self.device)

    def compute_jsd(self, scores_p, scores_q):
        """
        Compute Jensen-Shannon Divergence between two distributions over candidates.
        Input: Log-probs (scores).
        """
        p = F.softmax(scores_p, dim=0)
        q = F.softmax(scores_q, dim=0)
        m = 0.5 * (p + q)
        
        # KL(p||m)
        kl_p = F.kl_div(F.log_softmax(scores_p, dim=0), m, reduction='sum', log_target=False)
        # KL(q||m) = sum q * (log q - log m)
        # torch.kl_div expects input as log-probs and target as probs (if log_target=False)
        # But for q, we need to pass log_q as input
        kl_q = F.kl_div(F.log_softmax(scores_q, dim=0), m, reduction='sum', log_target=False)
        
        return 0.5 * (kl_p + kl_q)

    def asp_select_spans(self, input_ids: torch.Tensor, spans: List[Tuple[int, int]], 
                         candidate_tokens: List[List[int]], l_E: torch.Tensor, k: int = 5) -> List[Tuple[int, int]]:
        """
        ASP: Select top-k attributional spans.
        """
        if not spans:
            return []
            
        scores = []
        # Limit to checking last 10 spans for efficiency if many
        check_spans = spans[-10:] 
        
        for i, span in enumerate(check_spans):
            # Mask this span
            masked_ids = self.mask_span_in_prompt(input_ids, span).unsqueeze(0)
            # Compute scores
            l_masked = self.get_item_scores(masked_ids, candidate_tokens)
            # Compute JSD(l_E, l_masked)
            jsd = self.compute_jsd(l_E, l_masked).item()
            scores.append((jsd, span))
            
        # Add unchecked spans with 0 score (or handle differently)
        # For now, just consider checked ones.
        
        # Sort by JSD descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k spans
        selected = [s[1] for s in scores[:k]]
        return selected

    def run_instance(self, row, top_k=5, alpha=0.5, beta0=0.0, beta1=1.0, asp_k=3):
        """
        Run alignment for a single instance.
        """
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
            
        # Flatten messages if they are in the nested format
        flattened_messages = []
        for msg in messages:
            content = msg['content']
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if item.get('type') == 'text':
                        text_content += item.get('text', '')
                flattened_messages.append({"role": msg['role'], "content": text_content})
            else:
                flattened_messages.append(msg)

        prompt_text = self.tokenizer.apply_chat_template(flattened_messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)
        
        # 1. Generate No-Think Baseline
        with torch.no_grad():
            nt_outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                num_beams=1, # faster
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        no_think_text = self.tokenizer.decode(nt_outputs[0, input_ids.shape[1]:], skip_special_tokens=False)
        no_think_items = self.extract_items_from_text(no_think_text)
        
        # 2. Generate Think Baseline
        rationale_text, rationale_tokens = self.generate_rationale(input_ids, max_tokens=1024)
        input_with_rationale = torch.cat([input_ids, rationale_tokens.unsqueeze(0)], dim=1)
        
        with torch.no_grad():
            t_outputs = self.model.generate(
                input_with_rationale,
                max_new_tokens=512,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        think_items_text = self.tokenizer.decode(t_outputs[0, input_with_rationale.shape[1]:], skip_special_tokens=False)
        think_items = self.extract_items_from_text(think_items_text)
        
        # 3. Ground Truth & Candidates
        meta = json.loads(row['metadata'])
        gt_answer = meta['answer'] 
        gt_items = self.extract_items_from_text(gt_answer)
        
        # Parse history to get some negatives
        history_spans = self.parse_sid_spans(input_ids[0])
        history_items = []
        for start, end in history_spans:
            history_items.append(self.tokenizer.decode(input_ids[0, start:end+1]))
        
        # Candidate set: NT + Think + GT + some history (as negatives)
        candidate_set_str = list(set(no_think_items + think_items + gt_items + history_items[:20]))
        candidate_tokens = [self.encode_item(s) for s in candidate_set_str]
        
        if not candidate_tokens:
            return {
                "id": row.get("uuid", "unknown"),
                "error": "No candidates found"
            }

        # 4. Compute Scores
        
        # l_E: Evidence (Original Prompt)
        l_E = self.get_item_scores(input_ids, candidate_tokens)
        
        # l_C: CoT Context
        l_C = self.get_item_scores(input_with_rationale, candidate_tokens)
        
        # ASP & SCP
        spans = self.parse_sid_spans(input_ids[0])
        
        # Select important spans
        if spans:
            selected_spans = self.asp_select_spans(input_ids[0], spans, candidate_tokens, l_E, k=asp_k)
            # Identify spans to REMOVE (Unimportant ones) for c_S
            # c_S should keep selected_spans. So we remove (All - Selected).
            # Note: exact object identity matching might fail, use indices.
            selected_set = set(selected_spans)
            to_remove_S = [s for s in spans if s not in selected_set]
            
            if to_remove_S:
                c_S_ids = self.mask_multiple_spans(input_ids[0], to_remove_S).unsqueeze(0)
            else:
                c_S_ids = input_ids # Keep all
        else:
            c_S_ids = input_ids
            
        l_S = self.get_item_scores(c_S_ids, candidate_tokens)
        
        # c_P: Text Prior (Remove ALL history spans)
        if spans:
            c_P_ids = self.mask_multiple_spans(input_ids[0], spans).unsqueeze(0)
        else:
            c_P_ids = input_ids # Fallback
            
        l_P = self.get_item_scores(c_P_ids, candidate_tokens)
        
        # 5. Corrections & UG
        
        delta = l_C - l_S
        d = l_P - l_E 
        
        # Projection
        dot_prod = torch.dot(delta, d)
        norm_sq = torch.dot(d, d)
        if norm_sq > 1e-6:
            proj = (dot_prod / norm_sq) * d
        else:
            proj = torch.zeros_like(d)
            
        l_PDC = l_C - alpha * proj
        l_CDC = l_C - alpha * d
        
        # UG
        p_C = F.softmax(l_C, dim=0)
        p_S = F.softmax(l_S, dim=0)
        H_C = -torch.sum(p_C * torch.log(p_C + 1e-10))
        H_S = -torch.sum(p_S * torch.log(p_S + 1e-10))
        dH = H_C - H_S
        
        if norm_sq > 0 and torch.norm(delta) > 0:
            cos_align = F.cosine_similarity(delta.unsqueeze(0), d.unsqueeze(0)).item()
        else:
            cos_align = 0
        align = max(0, cos_align)
        
        alpha_ug = min(max(beta0 + beta1 * max(0, dH.item()) * align, 0.0), 1.0)
        
        l_Final = l_C - alpha_ug * proj
        
        def get_topk(scores, k=5):
            k = min(k, len(scores))
            indices = torch.argsort(scores, descending=True)[:k]
            return [candidate_set_str[i] for i in indices]

        return {
            "no_think_topk": no_think_items[:top_k],
            "think_topk": think_items[:top_k],
            "pdc_topk": get_topk(l_PDC, top_k),
            "cdc_topk": get_topk(l_CDC, top_k),
            "ours_topk": get_topk(l_Final, top_k),
            "metrics": {
                "alpha_ug": alpha_ug,
                "dH": dH.item(),
                "align": cos_align,
                "norm_d": math.sqrt(norm_sq.item()),
                "norm_proj": torch.norm(proj).item(),
                "jsd_asp_max": 0.0 # Placeholder
            },
            "gt_items": gt_items
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="checkpoints/OneRec-1.7B")
    parser.add_argument("--output_path", type=str, default="inference_results.jsonl")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta0", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print(f"Reading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    if args.num_samples > 0:
        df = df.head(args.num_samples)
    
    aligner = InferenceAligner(args.model_path)
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            res = aligner.run_instance(
                row, 
                top_k=args.top_k, 
                alpha=args.alpha,
                beta0=args.beta0,
                beta1=args.beta1
            )
            res["id"] = row.get("uuid", str(idx))
            results.append(res)
            
            # Flush periodically
            if len(results) % 5 == 0:
                with open(args.output_path, 'w') as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                        
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            import traceback
            traceback.print_exc()
            
    with open(args.output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {args.output_path}")

if __name__ == "__main__":
    main()
