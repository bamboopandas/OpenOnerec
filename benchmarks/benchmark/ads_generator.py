import os
import torch
import torch.nn.functional as F
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
from benchmark.base_generator import Generator
from benchmark.console import console, warning_style
import time

class ADSHuggingFaceGenerator(Generator):
    """
    Generator implementing Attention-driven Selection (ADS) for Interleaved-Modal CoT.
    Uses HuggingFace Transformers with proper KV caching for efficiency.
    Implements "Backfill" strategy: inserts placeholders during reasoning, fills actual SIDs for final answer.
    """
    def __init__(self, model_path: str, ads_top_k: int = 1, gpu_ids: Optional[List[int]] = None, **kwargs):
        self.model_name = model_path
        
        # Determine device
        if gpu_ids:
            self.device = f"cuda:{gpu_ids[0]}"
            console.print(f"Using specified GPU: {self.device}", style=warning_style)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.ads_top_k = ads_top_k
        
        console.print(f"Loading tokenizer from {model_path}...", style=warning_style)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        console.print(f"Loading model from {model_path}...", style=warning_style)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager"
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Identify special tokens for items
        self.sid_begin_token = "<|sid_begin|>"
        self.sid_end_token = "<|sid_end|>"
        self.sid_begin_id = self.tokenizer.convert_tokens_to_ids(self.sid_begin_token)
        self.sid_end_id = self.tokenizer.convert_tokens_to_ids(self.sid_end_token)
        
        # Identify </think> token ID for banning
        self.think_end_token = "</think>"
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids(self.think_end_token)
        
        # Trigger token for ADS (newline) 
        self.trigger_token = "\n" 
        self.trigger_token_id = self.tokenizer.convert_tokens_to_ids(self.trigger_token)
        
        console.print(f"ADS Generator initialized. Top-K: {self.ads_top_k} on {self.device}", style=warning_style)

    def _map_items_in_input(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """Identify ranges of item tokens in the input."""
        items = []
        in_item = False
        start_idx = -1
        ids = input_ids.tolist()
        for i, token_id in enumerate(ids):
            if token_id == self.sid_begin_id:
                in_item = True
                start_idx = i
            elif token_id == self.sid_end_id and in_item:
                in_item = False
                items.append((start_idx, i))
        return items

    def _backfill_placeholders(self, text: str) -> str:
        """Replace <Hidx> placeholders with actual SID sequences from the text history."""
        # Find all history items (SIDs) in the prompt text
        pattern = r"<\|sid_begin\|>.*?<\|sid_end|>"
        history_items = re.findall(pattern, text)
        
        def replace_func(match):
            idx = int(match.group(1))
            if 0 <= idx < len(history_items):
                return history_items[idx]
            return match.group(0) # Keep as is if index out of bounds
            
        new_text = re.sub(r"<H(\d+)>", replace_func, text)
        return new_text

    def _generate_standard(self, prompts: Dict[str, str], **kwargs) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Any]]:
        num_beams = kwargs.get("num_beams", 1)
        stop_strs = kwargs.get("stop", [])
        
        if num_beams > 1 or "</think>" not in stop_strs:
            return self._generate_with_hf_generate(prompts, **kwargs)
        else:
            return self._generate_with_ads_loop(prompts, **kwargs)

    def _generate_with_hf_generate(self, prompts: Dict[str, str], **kwargs) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Any]]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        results = {}
        mfu_stats = {}
        logprobs = {} 
        
        num_beams = kwargs.get("num_beams", 1)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        
        total_samples = len(prompts)
        for i, (sample_id, prompt) in enumerate(prompts.items()):
            start_time = time.time()
            
            # --- BACKFILL STEP ---
            backfilled_prompt = self._backfill_placeholders(prompt)
            
            inputs = self.tokenizer(backfilled_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            try:
                with torch.no_grad():
                    generated_outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                        early_stopping=True
                    )
                
                input_len = input_ids.shape[1]
                decoded_texts = []
                for seq in generated_outputs:
                    decoded = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                    decoded_texts.append(decoded)
                
                results[sample_id] = decoded_texts
                
                mfu_stats[sample_id] = {
                    "input_tokens": [input_len],
                    "output_tokens": [len(generated_outputs[0]) - input_len], 
                    "times": [time.time() - start_time]
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    console.print(f"OOM during sample {sample_id}. Clearing cache...", style="bold red")
                    torch.cuda.empty_cache()
                    results[sample_id] = [""] * num_return_sequences
                    mfu_stats[sample_id] = {"input_tokens": [0], "output_tokens": [0], "times": [0]}
                else:
                    raise e
            finally:
                del input_ids, attention_mask
                if 'generated_outputs' in locals(): del generated_outputs
                torch.cuda.empty_cache()
            
        return results, logprobs, mfu_stats

    def _generate_with_ads_loop(self, prompts: Dict[str, str], **kwargs) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Any]]:
        results = {}
        logprobs = {}
        mfu_stats = {}
        
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        stop_strs = kwargs.get("stop", [])
        temperature = kwargs.get("temperature", 0.0)
        
        # Pre-compute static tokens for reconstruction
        prefix_text = " 证据是用户曾经交互过 "
        suffix_text = "\n"
        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        suffix_tokens = self.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        
        total_samples = len(prompts)
        for i, (sample_id, prompt) in enumerate(prompts.items()):
            start_time = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            item_ranges = self._map_items_in_input(input_ids[0])
            
            all_ids = input_ids
            curr_input_ids = input_ids
            past_key_values = None
            
            generated_start_idx = input_ids.shape[1]
            last_ads_trigger_step = 0
            
            # Store insertions: list of (insertion_index_relative_to_gen, item_idx)
            # We store item_idx to resolve to actual tokens later
            insertions = []
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    outputs = self.model(
                        input_ids=curr_input_ids,
                        past_key_values=past_key_values,
                        output_attentions=True,
                        use_cache=True
                    )
                    
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # --- CONSTRAINTS ---
                    # 1. STRICTLY BAN raw SID generation during thinking
                    next_token_logits[:, self.sid_begin_id] = -float("inf")
                    
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    
                    next_token_id = next_token.item()
                    tokens_to_append = next_token
                    decoded_token = self.tokenizer.decode(next_token_id)
                    
                    should_stop = False
                    if next_token_id == self.tokenizer.eos_token_id or any(s in decoded_token for s in stop_strs):
                        should_stop = True

                    # --- ADS LOGIC (Pure Recording) ---
                    # Trigger check
                    if not should_stop and \
                       (next_token_id == self.trigger_token_id or "\n" in decoded_token) and \
                       item_ranges and \
                       (step - last_ads_trigger_step > 10):
                       
                        all_layers_attn = torch.stack(outputs.attentions) 
                        avg_attn = all_layers_attn[:, 0, :, -1, :].mean(dim=[0, 1]) 
                        
                        item_scores = []
                        for idx, (start, end) in enumerate(item_ranges):
                            if end < avg_attn.shape[0]:
                                score = avg_attn[start:end+1].mean().item()
                                item_scores.append((score, idx))
                        
                        if item_scores:
                            item_scores.sort(key=lambda x: x[0], reverse=True)
                            top_items = item_scores[:self.ads_top_k]
                            top_items.sort(key=lambda x: x[1])
                            
                            # Record insertion for the first top item (Top-1 for now based on previous code)
                            # Or loop if Top-K > 1? The previous code concatenated them.
                            # "The evidence is that users like XX" for each selected item.
                            
                            insertion_point = all_ids.shape[1] - generated_start_idx + 1
                            
                            current_insertions = []
                            for _, item_idx in top_items:
                                current_insertions.append(item_idx)
                                
                            if current_insertions:
                                insertions.append((insertion_point, current_insertions))
                                last_ads_trigger_step = step
                    
                    # Update context with ONLY the next token (Standard Autoregressive)
                    all_ids = torch.cat([all_ids, tokens_to_append], dim=1)
                    
                    if past_key_values is None:
                        curr_input_ids = all_ids
                    else:
                        curr_input_ids = tokens_to_append
                    
                    if all_ids.shape[1] - generated_start_idx >= max_new_tokens:
                        break
                    
                    if should_stop:
                        break
            
            # --- RECONSTRUCTION (Resolving SIDs) ---
            generated_only = all_ids[0, generated_start_idx:]
            final_tokens_list = []
            current_idx = 0
            
            # insertions is sorted by insertion_point (ascending)
            for point, item_indices in insertions:
                # Append segment from last point to current point
                if point > current_idx:
                    final_tokens_list.append(generated_only[current_idx:point])
                
                # Construct and append full evidence text for each selected item
                for item_idx in item_indices:
                    # Retrieve actual item tokens from input_ids
                    start, end = item_ranges[item_idx]
                    # Ensure indices are within bounds (they should be, as they come from input_ids)
                    item_tokens_tensor = input_ids[0, start:end+1]
                    
                    # [Prefix] + [Item Tokens] + [Suffix/Newline]
                    # Note: prefix_tokens shape is [1, L], item_tokens_tensor is [L]
                    full_insertion = torch.cat([prefix_tokens[0], item_tokens_tensor, suffix_tokens[0]])
                    final_tokens_list.append(full_insertion)
                
                current_idx = point
            
            # Append remaining
            if current_idx < len(generated_only):
                final_tokens_list.append(generated_only[current_idx:])
                
            if final_tokens_list:
                final_token_tensor = torch.cat(final_tokens_list)
            else:
                final_token_tensor = torch.tensor([], dtype=torch.long, device=self.device)

            decoded_text = self.tokenizer.decode(final_token_tensor, skip_special_tokens=True)
            results[sample_id] = [decoded_text]
            
            mfu_stats[sample_id] = {
                "input_tokens": [generated_start_idx],
                "output_tokens": [len(final_token_tensor)],
                "times": [time.time() - start_time]
            }
            
        return results, logprobs, mfu_stats

    def cleanup(self):
        import gc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
