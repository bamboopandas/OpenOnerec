from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import re
from benchmark.console import *
from utils.generator import RayVllmGenerator

class CompressedCoTGenerator(RayVllmGenerator):
    """
    Generator that implements a 3-stage process:
    1. Generate Chain-of-Thought (CoT).
    2. Compress the CoT into a summary (Heuristic Extraction).
    3. Generate final answer using Beam Search based on the compressed thought.
    """

    def _extract_conclusion_heuristic(self, thought: str) -> str:
        """
        Extract the conclusion from the thought process using heuristics.
        Logic:
        1. Clean up tags.
        2. Split into sentences.
        3. Take the last non-empty sentence (conclusion).
        4. If too short, append the one before it.
        5. Strictly remove any Semantic ID markers.
        6. Deduplicate keywords/phrases if present.
        """
        # 1. Clean up tags
        text = thought.replace("<think>", "").replace("</think>", "").strip()
        
        # 2. Strictly remove Semantic ID patterns (just in case they exist in source thought)
        # Pattern: <|sid_begin|>...<|sid_end|>
        text = re.sub(r"<\|sid_begin\|>.*?<\|sid_end\|>", "", text, flags=re.DOTALL)
        # Also remove standalone tags
        text = text.replace("<|sid_begin|>", "").replace("<|sid_end|>", "")
        
        # 3. Split into sentences (simple regex for Chinese/English punctuation)
        # Split by . ? ! ; 。 ？ ！ ； 

        sentences = re.split(r'([.。?!;；？！\n])', text)
        
        # Reconstruct sentences (delimiter is kept in odd positions)
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

        # 4. Extract Conclusion (Last 1-2 sentences)
        conclusion = clean_sentences[-1]
        
        # If conclusion is too short (e.g., just "Therefore."), prepend previous sentence
        if len(conclusion) < 15 and len(clean_sentences) > 1:
            conclusion = clean_sentences[-2] + " " + conclusion
            
        # 5. Intra-sentence deduplication (for repetitive keywords)
        # Split by comma (Eng/Chi), semicolon, enumeration comma, or multiple spaces
        # This handles: "A, A", "A A", "A; A", "A、A"
        parts = re.split(r'[,，;；、\s]+', conclusion)
        
        seen = set()
        deduped_parts = []
        for p in parts:
            p_clean = p.strip()
            # Remove common list noise
            p_clean = re.sub(r'^(and|or|with|和|以及|与|的)\s*$', '', p_clean)
            # Remove punctuation at ends
            p_clean = p_clean.strip('.。')
            
            if p_clean and len(p_clean) > 1 and p_clean.lower() not in [x.lower() for x in seen]:
                seen.add(p_clean)
                deduped_parts.append(p_clean)
        
        # Reconstruct if we actually had parts
        if deduped_parts:
            conclusion = ", ".join(deduped_parts)
            
        return conclusion

    def generate(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Three-stage generation:
        1. Generate thinking content.
        2. Compress thinking content (Heuristic).
        3. Generate final answer with beam search using compressed thinking.
        """
        
        console.print(
            "\n[CompressedCoT] Starting 3-Stage Generation Process (Heuristic Mode)",
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
        
        # Add repetition_penalty to discourage loops in source thought
        kwargs_stage1["repetition_penalty"] = 1.1
        
        # Call standard generation for Stage 1
        stage1_results, _, stage1_mfu = self._generate_standard(prompts, **kwargs_stage1)
        
        # --- Stage 2: Compress Thinking (Heuristic) ---
        console.print(
            "Stage 2/3: Compressing thinking content (Heuristic Extraction)...",
            style=warning_style,
        )
        
        # We do NOT use the model here. We use Python logic.
        stage2_results = {} # Map summary_id -> [summary_text]
        summary_id_map = {}
        
        for sample_id, thoughts in stage1_results.items():
            thought = thoughts[0]
            
            # Apply Heuristic
            summary = self._extract_conclusion_heuristic(thought)
            
            summary_id = f"{sample_id}_summary"
            stage2_results[summary_id] = [summary]
            summary_id_map[summary_id] = sample_id

        # No MFU for Stage 2 since it's CPU logic
        stage2_mfu = {}

        # --- Stage 3: Beam Search with Compressed Thought ---
        console.print(
            "Stage 3/3: Generating final answer with Beam Search...",
            style=warning_style,
        )
        
        final_prompts = {}
        prompt_token = kwargs.get("prompt_token", "")
        
        for summary_id, summaries in stage2_results.items():
            sample_id = summary_id_map[summary_id]
            summary = summaries[0].strip()
            original_prompt = prompts[sample_id]
            
            # Construct final prompt
            # Injecting compressed thought
            final_prompt = (
                f"{original_prompt}"
                f"<think>{summary}</think>\n"
                f"{prompt_token}"
            )
            final_prompts[sample_id] = final_prompt

        # Configure kwargs for Stage 3 (Beam Search)
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
            
            # Construct the prefix
            prefix = f"<think>{summary}</think>\n{prompt_token}"
            
            # Prepend to all beam answers
            final_answers = []
            for ans in answers:
                final_answers.append(prefix + ans)
            
            final_results[sample_id] = final_answers
        
        # --- Aggregate Results & MFU ---
        final_mfu_stats = defaultdict(lambda: {"input_tokens": [], "output_tokens": [], "times": []})
        
        for stage_stats in [stage1_mfu, stage3_mfu]: # Skip stage2
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
