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
    Generator implementing Contrastive Decoding for Recommendation.
    
    Process:
    1. Generate CoT (Thinking) using the full history (Expert).
    2. Construct Amateur context (CoT only).
    3. Generate Recommendation using Contrastive Decoding:
       Logits = Expert_Logits - alpha * Amateur_Logits
    """
    def __init__(
        self,
        model_name_or_path: str,
        alpha: float = 0.5,
        beta: float = 0.1,  # Adaptive plausibility threshold (optional, strict mode)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        console.print(f"Loading model from {model_name_or_path}...", style=subhead_style_2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Determine EOS tokens
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id:
             if isinstance(self.model.config.eos_token_id, list):
                 self.eos_token_ids.extend(self.model.config.eos_token_id)
             else:
                 self.eos_token_ids.append(self.model.config.eos_token_id)
        self.eos_token_ids = list(set(self.eos_token_ids))
        
    def _generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Implementation of Contrastive Decoding loop.
        """
        results = {}
        logprobs = {}
        mfu_stats = {}
        
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        
        console.print(f"Starting Contrastive Decoding generation for {len(prompts)} prompts...", style=subhead_style_2)
        
        for sample_id, prompt_text in prompts.items():
            start_time = time.time()
            
            input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            prompt_len = input_ids.shape[1]
            
            # Phase 1: Thinking
            # Generate until </think> or max limit
            with torch.no_grad():
                think_outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=kwargs.get("max_new_thinking_tokens", 1024),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_ids
                )
            
            # Decode the full output including prompt to find tags reliably
            full_output_ids = think_outputs[0]
            full_output_text = self.tokenizer.decode(full_output_ids, skip_special_tokens=False)
            
            # Extract CoT
            cot_content = ""
            expert_text = full_output_text # Default to full output if extraction fails
            
            if "<think>" in full_output_text:
                parts = full_output_text.split("<think>")
                if len(parts) > 1:
                    after_think = parts[-1]
                    if "</think>" in after_think:
                        cot_content = after_think.split("</think>")[0]
                        # Reconstruct Expert Text: Prompt + <think> + CoT + </think>
                        # Use the text up to </think> and include </think>
                        expert_text = full_output_text.split("</think>")[0] + "</think>"
                    else:
                        cot_content = after_think
                        expert_text = full_output_text
            
            # Clean expert text: ensure we are at the end of generated tokens
            # We need to tokenize expert_text to get ids
            expert_input_ids = self.tokenizer.encode(expert_text, return_tensors="pt").to(self.device)
            
            # Prepare Amateur Input: <|im_start|>assistant\n<think>CoT</think>\n\n
            amateur_text = f"<|im_start|>assistant\n<think>{cot_content}</think>\n\n"
            amateur_input_ids = self.tokenizer.encode(amateur_text, return_tensors="pt").to(self.device)
            
            generated_tokens = []
            
            # Initialize KV caches
            expert_past_key_values = DynamicCache()
            amateur_past_key_values = DynamicCache()
            
            # Prefill
            with torch.no_grad():
                expert_out = self.model(
                    input_ids=expert_input_ids,
                    past_key_values=expert_past_key_values,
                    use_cache=True
                )
                expert_logits = expert_out.logits[:, -1, :]
                
                amateur_out = self.model(
                    input_ids=amateur_input_ids,
                    past_key_values=amateur_past_key_values,
                    use_cache=True
                )
                amateur_logits = amateur_out.logits[:, -1, :]
            
            next_token = torch.argmax(expert_logits, dim=-1).unsqueeze(0)
            
            # Decoding Loop
            for i in range(max_new_tokens):
                cd_logits = expert_logits - self.alpha * amateur_logits
                
                temp = kwargs.get("temperature", 0.6)
                if temp < 1e-5:
                    next_token = torch.argmax(cd_logits, dim=-1).unsqueeze(0)
                else:
                    probs = F.softmax(cd_logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                if next_token.item() in self.eos_token_ids:
                    break
                    
                with torch.no_grad():
                    expert_out = self.model(
                        input_ids=next_token,
                        past_key_values=expert_past_key_values,
                        use_cache=True
                    )
                    expert_logits = expert_out.logits[:, -1, :]
                    
                    amateur_out = self.model(
                        input_ids=next_token,
                        past_key_values=amateur_past_key_values,
                        use_cache=True
                    )
                    amateur_logits = amateur_out.logits[:, -1, :]
            
            end_time = time.time()
            
            final_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Return generated text (CoT + Answer)
            # Remove prompt from expert_text
            generated_cot_part = expert_text[len(prompt_text):] if len(expert_text) > len(prompt_text) else ""
            full_generated = generated_cot_part + final_answer
            
            results[sample_id] = [full_generated]
            
            mfu_stats[sample_id] = {
                "input_tokens": [prompt_len],
                "output_tokens": [len(generated_tokens) + (len(expert_input_ids[0]) - prompt_len)],
                "times": [end_time - start_time]
            }
            
        return results, logprobs, mfu_stats
