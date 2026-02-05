from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.console import *
from utils.generator import RayVllmGenerator

class CompressedCoTGenerator(RayVllmGenerator):
    """
    Generator that implements a 3-stage process:
    1. Generate Chain-of-Thought (CoT) using Main Model (OneRec).
    2. Compress the CoT into a summary using a Secondary Model (generic Qwen).
    3. Generate final answer using Beam Search (Main Model) based on the compressed thought.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract summarizer_model_path before calling super().__init__
        self.summarizer_model_path = kwargs.pop('summarizer_model_path', None)
        
        super().__init__(*args, **kwargs)
        
        # Initialize Summarizer Model
        if self.summarizer_model_path:
            console.print(f"\n[CompressedCoT] Loading Summarizer Model from: {self.summarizer_model_path}", style=warning_style)
            try:
                # Load on the same GPU as the main process (usually GPU 0 or specified)
                # If using Ray, this process is the driver. The workers hold the vLLM model.
                # So loading a small model here is fine if there is VRAM.
                # Ideally, we should check available devices. 
                # Assuming single-node, we use "cuda:0" or similar. 
                # Check gpu_ids from kwargs if passed to RayVllmGenerator, but it consumes them.
                # We will try to find a free GPU or just use cuda:0. 
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.summ_tokenizer = AutoTokenizer.from_pretrained(self.summarizer_model_path, trust_remote_code=True)
                self.summ_model = AutoModelForCausalLM.from_pretrained(
                    self.summarizer_model_path, 
                    trust_remote_code=True, 
                    torch_dtype=torch.bfloat16,
                    device_map="auto" # Let HF decide placement
                )
                self.summ_model.eval()
                console.print("[CompressedCoT] Summarizer Model Loaded Successfully.", style=success_style)
            except Exception as e:
                console.print(f"[CompressedCoT] Failed to load summarizer model: {e}", style=err_style)
                self.summ_model = None
        else:
            console.print("[CompressedCoT] No summarizer_model_path provided. Will use Heuristic Fallback.", style=warning_style)
            self.summ_model = None

    def _extract_conclusion_heuristic(self, thought: str) -> str:
        """Fallback heuristic if model fails or is not provided"""
        text = thought.replace("<think>", "").replace("</think>", "").strip()
        text = re.sub(r"<\|sid_begin\|>.*?<\|sid_end\|>", "", text, flags=re.DOTALL)
        text = text.replace("<|sid_begin|>", "").replace("<|sid_end|>", "")
        sentences = re.split(r'([.。?!;；？！\n])', text)
        clean_sentences = []
        current = ""
        for part in sentences:
            if re.match(r'[.。?!;；？！\n]', part):
                current += part
                if current.strip():
                    clean_sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            clean_sentences.append(current.strip())
        if not clean_sentences:
            return "Based on user history."
        conclusion = clean_sentences[-1]
        if len(conclusion) < 15 and len(clean_sentences) > 1:
            conclusion = clean_sentences[-2] + " " + conclusion
        return conclusion

    def _generate_summary_with_model(self, thoughts: List[str]) -> List[str]:
        """Generate summaries using the secondary Qwen model"""
        summaries = []
        
        # Batching could be implemented, but doing sequential for simplicity/safety first
        for thought in thoughts:
            # Clean thought
            clean_thought = thought.replace("<think>", "").replace("</think>", "").strip()
            clean_thought = re.sub(r"<\|sid_begin\|>.*?<\|sid_end\|>", "", clean_thought, flags=re.DOTALL)
            
            # Construct Prompt (Chinese)
            # Qwen chat template usually: <|im_start|>system\n...<|im_end|><|im_start|>user\n...<|im_end|><|im_start|>assistant\n
            # We will use apply_chat_template if available, or raw string
            
            messages = [
                {"role": "system", "content": "你是一个严格的总结助手。请直接输出结果，不要包含“好的”、“明白”、“以下是总结”等任何废话。"},
                # {"role": "user", "content": f"以下是一段关于推荐任务的推理过程：\n\n{clean_thought}\n\n请提取核心偏好和推荐理由。\n\n总结："}
                {"role": "user", "content": f"以下是一段关于推荐任务的推理过程：\n\n{clean_thought}\n\n请提取核心偏好和推荐理由，控制在50字以内。\n\n总结："}
            ]
            

            try:
                text = self.summ_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback format
                text = f"System: 你是一个严格的总结助手。\nUser: 以下是一段推理：{clean_thought}\n请提取核心偏好。\nAssistant:"

            inputs = self.summ_tokenizer([text], return_tensors="pt").to(self.summ_model.device)
            
            with torch.no_grad():
                generated_ids = self.summ_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, # Deterministic summary
                    temperature=0.1
                )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            summary = self.summ_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Heuristic Cleaning of Meta-talk
            # Remove common prefixes like "Okay, I will...", "Here is the summary:", etc.
            prefixes_to_clean = [
                "Okay,", "Okay.", "Sure,", "Sure.", "Certainly,", "Certainly.",
                "Here is", "Here's", "Based on", "To summarize", "In summary",
                "好的，", "好的。", "明白，", "没问题，", "以下是", "根据", "总结如下"
            ]
            
            # Simple check: if starts with prefix, try to find the actual start
            cleaned_summary = summary
            for p in prefixes_to_clean:
                if cleaned_summary.lower().startswith(p.lower()):
                    # Try to split by punctuation (colon, comma, newline)
                    parts = re.split(r'[:：\n]', cleaned_summary, 1)
                    if len(parts) > 1:
                        cleaned_summary = parts[1].strip()
                    else:
                        # Just remove the prefix
                        cleaned_summary = cleaned_summary[len(p):].strip()
            
            summaries.append(cleaned_summary)
            
        return summaries

    def generate(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        
        console.print(
            "\n[CompressedCoT] Starting 3-Stage Generation Process (Dual-Model Mode)",
            style=head_style_2,
        )

        # --- Stage 1: Generate Thinking (CoT) ---
        console.print(
            "Stage 1/3: Generating thinking content (Main Model)...",
            style=warning_style,
        )
        
        kwargs_stage1 = kwargs.copy()
        kwargs_stage1["stop"] = ["</think>"]
        kwargs_stage1["max_new_tokens"] = kwargs.get("max_new_thinking_tokens", 1024)
        kwargs_stage1["num_beams"] = 1
        kwargs_stage1["num_return_sequences"] = 1
        kwargs_stage1["repetition_penalty"] = 1.1
        
        stage1_results, _, stage1_mfu = self._generate_standard(prompts, **kwargs_stage1)
        
        # --- Stage 2: Compress Thinking (Secondary Model) ---
        console.print(
            "Stage 2/3: Compressing thinking content...",
            style=warning_style,
        )
        
        stage2_results = {}
        summary_id_map = {}
        
        # Collect all thoughts
        all_sample_ids = []
        all_thoughts = []
        
        for sample_id, thoughts in stage1_results.items():
            all_sample_ids.append(sample_id)
            all_thoughts.append(thoughts[0])
            
        if self.summ_model:
            # Use Model
            summaries = self._generate_summary_with_model(all_thoughts)
        else:
            # Use Heuristic Fallback
            console.print("[CompressedCoT] Using Heuristic Fallback for summarization.", style=warning_style)
            summaries = [self._extract_conclusion_heuristic(t) for t in all_thoughts]
            
        for i, sample_id in enumerate(all_sample_ids):
            summary_id = f"{sample_id}_summary"
            stage2_results[summary_id] = [summaries[i]]
            summary_id_map[summary_id] = sample_id

        stage2_mfu = {} # No MFU tracking for local secondary model for now

        # --- Stage 3: Beam Search with Compressed Thought ---
        console.print(
            "Stage 3/3: Generating final answer with Beam Search (Main Model)...",
            style=warning_style,
        )
        
        final_prompts = {}
        prompt_token = kwargs.get("prompt_token", "")
        
        for summary_id, summaries in stage2_results.items():
            sample_id = summary_id_map[summary_id]
            summary = summaries[0].strip()
            original_prompt = prompts[sample_id]
            
            final_prompt = (
                f"{original_prompt}"
                f"<think>{summary}</think>\n"
                f"{prompt_token}"
            )
            final_prompts[sample_id] = final_prompt

        kwargs_stage3 = kwargs.copy()
        kwargs_stage3["num_beams"] = kwargs.get("num_beams", 4) 
        kwargs_stage3["num_return_sequences"] = kwargs.get("num_return_sequences", kwargs_stage3["num_beams"])
        kwargs_stage3["max_new_tokens"] = kwargs.get("max_new_tokens", 128)
        
        stage3_results, stage3_logprobs, stage3_mfu = self._generate_standard(final_prompts, **kwargs_stage3)
        
        # --- Prepend Compressed Thought to Results ---
        final_results = {}
        
        for sample_id, answers in stage3_results.items():
            summary_id = f"{sample_id}_summary"
            summary = stage2_results[summary_id][0].strip()
            
            prefix = f"<think>{summary}</think>\n{prompt_token}"
            
            final_answers = []
            for ans in answers:
                final_answers.append(prefix + ans)
            
            final_results[sample_id] = final_answers
        
        # --- Aggregate Results & MFU ---
        final_mfu_stats = defaultdict(lambda: {"input_tokens": [], "output_tokens": [], "times": []})
        
        for stage_stats in [stage1_mfu, stage3_mfu]:
            for sid, stats in stage_stats.items():
                real_sid = sid
                if "_summary" in sid:
                    real_sid = summary_id_map.get(sid, sid)
                
                if real_sid in final_mfu_stats:
                     final_mfu_stats[real_sid]["input_tokens"].extend(stats.get("input_tokens", []))
                     final_mfu_stats[real_sid]["output_tokens"].extend(stats.get("output_tokens", []))
                     final_mfu_stats[real_sid]["times"].extend(stats.get("times", []))
                else:
                     final_mfu_stats[real_sid] = stats.copy()
        
        self.mfu_stats = final_mfu_stats
        
        return final_results, stage3_logprobs