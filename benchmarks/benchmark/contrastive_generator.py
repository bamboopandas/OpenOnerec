import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Dict, List, Any, Optional, Tuple
import math
import copy
import time
import re

from benchmark.base_generator import Generator, HfTransformersMixin
from benchmark.console import *

class ContrastiveGenerator(HfTransformersMixin, Generator):
    """
    Generator implementing Batched Contrastive Decoding.
    Provides standard generate() and specialized generate_contrastive().
    """
    def __init__(
        self,
        model_name_or_path: str,
        alpha: float = 0.5,
        beta: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        console.print(f"Loading model from {model_name_or_path}...", style=subhead_style_2)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code
        )
        if max_model_len:
            self.tokenizer.model_max_length = max_model_len
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else (torch.float16 if dtype == "float16" else "auto")
        
        attn_implementation = "eager"
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            if dtype in ["bfloat16", "float16"]:
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                    console.print("Enable Flash Attention 2", style=success_style)
                except ImportError:
                    pass
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation
        )
        self.model.eval()
        
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id:
             if isinstance(self.model.config.eos_token_id, list):
                 self.eos_token_ids.extend(self.model.config.eos_token_id)
             else:
                 self.eos_token_ids.append(self.model.config.eos_token_id)
        self.eos_token_ids = list(set(self.eos_token_ids))

    def _generate_standard(self, prompts: Dict[str, str], **kwargs) -> tuple:
        """Standard generation (Phase 1 Thinking)"""
        # ... Reuse standard transformers generation for thinking ...
        # This is essentially what the base class or standard generator does.
        # But we need batching here too for speed.
        
        batch_size = kwargs.get("worker_batch_size", 4)
        if batch_size > 128:
            batch_size = 128
            
        results = {}
        mfu_stats = {}
        
        prompt_items = list(prompts.items())
        
        for i in range(0, len(prompt_items), batch_size):
            batch_items = prompt_items[i : i + batch_size]
            batch_ids = [item[0] for item in batch_items]
            batch_prompts = [item[1] for item in batch_items]
            
            start_time = time.time()
            
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_thinking_tokens", 1024),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_ids
                )
                
            new_tokens = outputs[:, input_len:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
            
            for idx, sample_id in enumerate(batch_ids):
                results[sample_id] = [decoded[idx]]
                mfu_stats[sample_id] = {
                    "input_tokens": [input_len],
                    "output_tokens": [len(new_tokens[idx])],
                    "times": [time.time() - start_time]
                }
                
        return results, {}, mfu_stats

    def generate_contrastive(
        self,
        expert_prompts: Dict[str, str],
        amateur_prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Phase 2: Contrastive Decoding
        Requires aligned expert and amateur prompts.
        """
        results = {}
        mfu_stats = {}
        
        batch_size = kwargs.get("worker_batch_size", 4)
        if batch_size > 128:
            batch_size = 128
            
        sample_ids = list(expert_prompts.keys())
        
        for i in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[i : i + batch_size]
            batch_expert = [expert_prompts[uid] for uid in batch_ids]
            batch_amateur = [amateur_prompts[uid] for uid in batch_ids]
            
            start_time = time.time()
            
            # Tokenize both
            self.tokenizer.padding_side = "left"
            expert_inputs = self.tokenizer(batch_expert, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            amateur_inputs = self.tokenizer(batch_amateur, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            
            # Init caches
            expert_kv = DynamicCache()
            amateur_kv = DynamicCache()
            
            with torch.no_grad():
                expert_out = self.model(input_ids=expert_inputs.input_ids, attention_mask=expert_inputs.attention_mask, past_key_values=expert_kv, use_cache=True)
                amateur_out = self.model(input_ids=amateur_inputs.input_ids, attention_mask=amateur_inputs.attention_mask, past_key_values=amateur_kv, use_cache=True)
                
            # Logits from last valid token
            expert_logits = expert_out.logits[:, -1, :]
            amateur_logits = amateur_out.logits[:, -1, :]
            
            generated_ids = [[] for _ in range(len(batch_ids))]
            finished_mask = [False] * len(batch_ids)
            
            for step in range(kwargs.get("max_new_tokens", 128)):
                if all(finished_mask): break
                
                cd_logits = expert_logits - self.alpha * amateur_logits
                
                temp = kwargs.get("temperature", 0.6)
                if temp < 1e-5:
                    next_tokens = torch.argmax(cd_logits, dim=-1)
                else:
                    probs = F.softmax(cd_logits / temp, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                next_input_ids = next_tokens.unsqueeze(-1)
                
                with torch.no_grad():
                    expert_out = self.model(input_ids=next_input_ids, past_key_values=expert_kv, use_cache=True)
                    amateur_out = self.model(input_ids=next_input_ids, past_key_values=amateur_kv, use_cache=True)
                    
                expert_logits = expert_out.logits[:, -1, :]
                amateur_logits = amateur_out.logits[:, -1, :]
                
                current_tokens = next_tokens.cpu().tolist()
                for idx, tid in enumerate(current_tokens):
                    if not finished_mask[idx]:
                        if tid in self.eos_token_ids:
                            finished_mask[idx] = True
                        else:
                            generated_ids[idx].append(tid)
                            
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for idx, sample_id in enumerate(batch_ids):
                results[sample_id] = [decoded[idx]]
                mfu_stats[sample_id] = {
                    "input_tokens": [expert_inputs.input_ids.shape[1]],
                    "output_tokens": [len(generated_ids[idx])],
                    "times": [time.time() - start_time]
                }
                
        return results, {}, mfu_stats