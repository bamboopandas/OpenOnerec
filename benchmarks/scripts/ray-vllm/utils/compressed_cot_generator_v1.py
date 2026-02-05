from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from benchmark.console import *
from utils.generator import RayVllmGenerator

class CompressedCoTGenerator(RayVllmGenerator):
    """
    Generator that implements a 3-stage process:
    1. Generate Chain-of-Thought (CoT).
    2. Compress the CoT into a summary.
    3. Generate final answer using Beam Search based on the compressed thought.
    """

    def generate(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Three-stage generation:
        1. Generate thinking content.
        2. Compress thinking content.
        3. Generate final answer with beam search using compressed thinking.
        """
        
        console.print(
            "\n[CompressedCoT] Starting 3-Stage Generation Process",
            style=head_style_2,
        )

        # --- Stage 1: Generate Thinking (CoT) ---
        console.print(
            "Stage 1/3: Generating thinking content (CoT)...",
            style=warning_style,
        )
        
        # Configure kwargs for Stage 1
        kwargs_stage1 = kwargs.copy()
        kwargs_stage1["stop"] = ["</think>"]
        kwargs_stage1["max_new_tokens"] = kwargs.get("max_new_thinking_tokens", 1024)
        kwargs_stage1["num_beams"] = 1 # Use sampling for diversity in thinking
        kwargs_stage1["num_return_sequences"] = 1 # One thought per prompt for now
        
        # Call standard generation for Stage 1
        # We use _generate_standard which distributes to workers
        stage1_results, _, stage1_mfu = self._generate_standard(prompts, **kwargs_stage1)
        
        # --- Stage 2: Compress Thinking ---
        console.print(
            "Stage 2/3: Compressing thinking content...",
            style=warning_style,
        )
        
        # Prepare prompts for summarization
        # We ask the model to summarize the thought it just generated
        summary_prompts = {}
        # Map summary_id back to original_sample_id
        summary_id_map = {} 
        
        debug_printed = False
        for sample_id, thoughts in stage1_results.items():
            thought = thoughts[0] # Take the first thought
            # Strip <think> tags if present (though stop might have handled end, start might be missing)
            thought = thought.replace("<think>", "").replace("</think>", "").strip()
            
            if not thought:
                thought = "No reasoning provided."

            # Construct summarization prompt
            # Note: This prompt format depends on the model's instruction tuning. 
            # We use a generic one, assuming the model can handle it.
            # Ideally, we should wrap this in a chat template if the model expects it.
            # For now, we append instructions.
            summary_prompt = (
                f"以下是针对推荐任务的推理过程：\n\n"
                f"{thought}\n\n"
                f"请提炼出用户的核心意图以及最终推荐的关键理由。"
                f"将其浓缩为一句极其简练的话（30字以内）。"
                f"去除冗余背景，直击核心逻辑。\n"
                f"简要总结："
            )
            
            if not debug_printed:
                console.print("\n[DEBUG] Example Summary Prompt:", style=warning_style)
                console.print(summary_prompt)
                console.print("-" * 50)
                debug_printed = True
            
            summary_id = f"{sample_id}_summary"
            summary_prompts[summary_id] = summary_prompt
            summary_id_map[summary_id] = sample_id

        # Configure kwargs for Stage 2 (Summarization)
        kwargs_stage2 = kwargs.copy()
        kwargs_stage2.pop("prompt_token", None) # Remove prompt_token to avoid triggering ID generation
        kwargs_stage2["max_new_tokens"] = 256 # Short summary
        kwargs_stage2["stop"] = ["<|sid_begin|>"] # Strictly stop if it tries to generate SIDs
        kwargs_stage2["num_beams"] = 1
        kwargs_stage2["num_return_sequences"] = 1
        kwargs_stage2["temperature"] = 0.2 # Low temp for deterministic summary
        
        stage2_results, _, stage2_mfu = self._generate_standard(summary_prompts, **kwargs_stage2)
        
        # --- Stage 3: Beam Search with Compressed Thought ---
        console.print(
            "Stage 3/3: Generating final answer with Beam Search...",
            style=warning_style,
        )
        
        final_prompts = {}
        
        for summary_id, summaries in stage2_results.items():
            sample_id = summary_id_map[summary_id]
            summary = summaries[0].strip()
            original_prompt = prompts[sample_id]
            
            # Construct final prompt
            # We inject the compressed thought as if it was a <think> block or context
            # We use a specific marker so we can potentially strip it later or just leave it
            # Using <compressed_thought> tag for clarity
            
            # Check if we should use the prompt_token (e.g., "Response:")
            prompt_token = kwargs.get("prompt_token", "")
            
            final_prompt = (
                f"{original_prompt}"
                f"<think>Summary of reasoning: {summary}</think>\n"
                f"{prompt_token}"
            )
            final_prompts[sample_id] = final_prompt

        # Configure kwargs for Stage 3 (Beam Search)
        kwargs_stage3 = kwargs.copy()
        # Restore original beam settings
        # The user requested "multiple beam searches", implies using beam search here
        kwargs_stage3["num_beams"] = kwargs.get("num_beams", 4) 
        kwargs_stage3["num_return_sequences"] = kwargs.get("num_return_sequences", kwargs_stage3["num_beams"])
        kwargs_stage3["max_new_tokens"] = kwargs.get("max_new_tokens", 128)
        
        stage3_results, stage3_logprobs, stage3_mfu = self._generate_standard(final_prompts, **kwargs_stage3)
        
        # --- Prepend Compressed Thought to Results ---
        # stage3_results currently contains ONLY the answer (because the prompt is stripped).
        # We need to prepend "<think>Summary...</think>\n{prompt_token}" to it.
        
        final_results = {}
        prompt_token = kwargs.get("prompt_token", "")
        
        for sample_id, answers in stage3_results.items():
            # Get the summary for this sample
            # Find summary_id for this sample_id
            # Invert summary_id_map or just reconstruct ID
            summary_id = f"{sample_id}_summary"
            summary = stage2_results[summary_id][0].strip()
            
            # Clean up the summary by removing common prefixes
            for prefix_to_strip in ["简要总结：", "简要总结:", "Summary:", "Summary of reasoning:", "总结：", "总结:"]:
                if summary.startswith(prefix_to_strip):
                    summary = summary[len(prefix_to_strip):].strip()
            
            # Construct the prefix - removed the hardcoded "Summary of reasoning:"
            prefix = f"<think>{summary}</think>\n{prompt_token}"
            
            # Prepend to all beam answers
            final_answers = []
            for ans in answers:
                final_answers.append(prefix + ans)
            
            final_results[sample_id] = final_answers
        
        # --- Aggregate Results & MFU ---
        # Combine MFU stats
        final_mfu_stats = defaultdict(lambda: {"input_tokens": [], "output_tokens": [], "times": []})
        
        for stage_stats in [stage1_mfu, stage2_mfu, stage3_mfu]:
            for sid, stats in stage_stats.items():
                # Map summary IDs back to original sample IDs
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
