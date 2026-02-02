import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Dict, List, Any, Optional, Tuple
import math
import copy
import time

from benchmark.base_generator import Generator, HfTransformersMixin
from benchmark.console import *

class PowerDecodingGenerator(HfTransformersMixin, Generator):
    """
    Generator implementing Future-Aware Power Decoding (Algorithm 1 & 2)
    """
    def __init__(
        self,
        model_name_or_path: str,
        alpha: float = 2.0,
        top_k_candidates: int = 5,
        max_rollouts: int = 5,
        max_lookahead: int = 3,
        crit_threshold: float = 0.5, # Entropy threshold for Crit
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.alpha = alpha
        self.top_k_candidates = top_k_candidates
        self.max_rollouts = max_rollouts
        self.max_lookahead = max_lookahead
        self.crit_threshold = crit_threshold
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
        Implementation of Future-Aware Power Decoding loop.
        """
        results = {}
        logprobs = {}
        mfu_stats = {}
        
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        
        console.print(f"Starting Power Decoding generation for {len(prompts)} prompts...", style=subhead_style_2)
        
        for sample_id, prompt_text in prompts.items():
            input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            prompt_len = input_ids.shape[1]
            
            past_key_values = DynamicCache()
            
            generated_tokens = []
            
            start_time = time.time()
            
            # Initial forward to fill cache
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            for t in range(max_new_tokens):
                # Probs
                p_t = F.softmax(next_token_logits, dim=-1)
                
                # 2) Crit & Budget
                is_critical = self._crit(t, p_t)
                M_t, H_t = self._budget(t, p_t)
                
                next_token = None
                
                if not is_critical or M_t == 0 or H_t == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                else:
                    # Power Decoding Path
                    top_k_probs, top_k_ids = torch.topk(p_t, self.top_k_candidates, dim=-1)
                    C_t = top_k_ids[0].tolist() 
                    
                    log_w_t = []
                    
                    for c in C_t:
                        log_zeta = self._estimate_zeta_log(
                            past_key_values=past_key_values,
                            curr_token_id=c,
                            M=M_t,
                            H=H_t
                        )
                        p_c = p_t[0, c].item()
                        log_p_c = math.log(p_c + 1e-10)
                        log_w = self.alpha * log_p_c + log_zeta
                        log_w_t.append(log_w)
                    
                    log_w_tensor = torch.tensor(log_w_t, device=self.device)
                    w_probs = F.softmax(log_w_tensor, dim=0)
                    sample_idx = torch.multinomial(w_probs, num_samples=1).item()
                    next_token_id = C_t[sample_idx]
                    next_token = torch.tensor([[next_token_id]], device=self.device)
                
                # Append
                generated_tokens.append(next_token.item())
                
                # Advance model for NEXT step
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                next_token_logits = outputs.logits[:, -1, :]
                
                # Check EOS
                if next_token.item() in self.eos_token_ids:
                    break
            
            end_time = time.time()
            decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results[sample_id] = [decoded_text]
            
            mfu_stats[sample_id] = {
                "input_tokens": [prompt_len],
                "output_tokens": [len(generated_tokens)],
                "times": [end_time - start_time]
            }

        return results, logprobs, mfu_stats

    def _crit(self, t: int, p_t: torch.Tensor) -> bool:
        entropy = -torch.sum(p_t * torch.log(p_t + 1e-10), dim=-1).item()
        return entropy > self.crit_threshold

    def _budget(self, t: int, p_t: torch.Tensor) -> Tuple[int, int]:
        return self.max_rollouts, self.max_lookahead

    def _estimate_zeta_log(
        self,
        past_key_values: DynamicCache,
        curr_token_id: int,
        M: int,
        H: int
    ) -> float:
        """
        Algorithm 2: Monte-Carlo Lookahead for log zeta_t(c)
        """
        s_r_list = []
        
        next_input = torch.tensor([[curr_token_id]], device=self.device)
        
        # Advance model by one step with candidate token c
        # We need to clone the cache first because feeding 'c' is part of all rollouts for this candidate
        cand_cache = self._clone_dynamic_cache(past_key_values)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=next_input,
                past_key_values=cand_cache,
                use_cache=True
            )
            base_rollout_kv = cand_cache
            next_logits_after_c = outputs.logits[:, -1, :]
        
        for r in range(M):
            log_p_r = 0.0
            
            curr_kv = self._clone_dynamic_cache(base_rollout_kv)
            next_logits = next_logits_after_c
            
            for h in range(H):
                probs = F.softmax(next_logits, dim=-1)
                y_idx = torch.multinomial(probs, num_samples=1)
                y_token = y_idx
                
                p_y = probs[0, y_idx.item()].item()
                log_p_r += math.log(p_y + 1e-10)
                
                if h < H - 1:
                    with torch.no_grad():
                        step_out = self.model(
                            input_ids=y_token,
                            past_key_values=curr_kv,
                            use_cache=True
                        )
                        next_logits = step_out.logits[:, -1, :]
            
            s_r = (self.alpha - 1) * log_p_r
            s_r_list.append(s_r)
            
        s_r_tensor = torch.tensor(s_r_list, device=self.device)
        log_zeta = torch.logsumexp(s_r_tensor, dim=0).item() - math.log(M)
        
        return log_zeta

    def _clone_dynamic_cache(self, cache: DynamicCache) -> DynamicCache:
        """
        Clone DynamicCache.
        """
        return DynamicCache.from_legacy_cache(cache.to_legacy_cache())