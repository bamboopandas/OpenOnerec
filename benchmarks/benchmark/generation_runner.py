"""
Generation Runner

Responsible for:
1. Loading test data via data loader
2. Calling Generator to produce model outputs  
3. Saving generation results to JSON files

Note: Does NOT compute evaluation metrics (handled by task-specific evaluators)
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from benchmark.console import *
from benchmark.base_generator import Generator
from benchmark.tasks.v1_0.base_loader import BaseLoader


class GenerationRunner:
    """
    Generation task runner
    
    Orchestrates the generation phase of evaluation:
    - Loads test data via data loader
    - Calls generator to produce model outputs
    - Saves generation results to disk
    
    Evaluation metrics are computed separately by task-specific evaluators.
    """
    
    def __init__(
        self,
        data_loader: BaseLoader,
        overwrite: bool = False
    ):
        """
        Args:
            data_loader: Data loader (any object with load_data method)
            overwrite: Whether to overwrite existing results
        """
        self.data_loader = data_loader
        self.overwrite = overwrite
        self.benchmark_version = data_loader.benchmark_version
    
    def __call__(
        self,
        task_name: str,
        split: str,
        results_save_dir: str,
        generator: Generator,
        **kwargs
    ) -> None:
        """
        Execute generation pipeline
        
        This method is responsible for generation and saving only,
        NOT for computing evaluation metrics.
        
        Args:
            task_name: Task name
            split: Dataset split
            results_save_dir: Results save directory
            generator: Generator instance
            **kwargs: Generation parameters
        
        Returns:
            None
        """
        model_name = str(generator)
        results_dir = os.path.join(
            results_save_dir,
            model_name,
            task_name
        )
        os.makedirs(results_dir, exist_ok=True)
        
        generation_file = os.path.join(results_dir, f"{split}_generated.json")
        
        # Check if generation results already exist
        if os.path.exists(generation_file) and not self.overwrite:
            console.print(f"Generation results already exist, skipping: {generation_file}")
            console.print("To regenerate, please set overwrite=True")
            
            return None
        
        start_time = time.time()

        # Extract sample_size parameter (don't pass to generator)
        sample_size_param = kwargs.pop('sample_size', None)

        # 1. Load data
        test_data = self.data_loader.load_data(task_name=task_name, split=split, sample_size=sample_size_param)

        # 2. Extract prompts and references
        prompts = {id: data["prompt"] for id, data in test_data.items()}
        references = {id: data["ground_truth"] for id, data in test_data.items()}

        # 3. Generate text (unified entry point)
        if hasattr(generator, 'generate_contrastive') and kwargs.get('enable_thinking', False):
            console.print("[Contrastive] Starting Phase 1: Thinking Generation", style=subhead_style_2)
            # Phase 1: Thinking
            # We use standard generate() for this, assuming it handles batching
            think_generations, think_logprobs, think_mfu = generator.generate(prompts, **kwargs)
            
            expert_prompts = {}
            amateur_prompts = {}
            cot_parts = {}
            
            console.print("[Contrastive] Processing CoT and preparing Phase 2", style=subhead_style_2)
            for sample_id, gens in think_generations.items():
                gen_text = gens[0] # Take first candidate
                
                # Robust CoT extraction
                cot_content = ""
                if "<think>" in gen_text and "</think>" in gen_text:
                    cot_content = gen_text.split("<think>")[1].split("</think>")[0]
                elif "<think>" in gen_text:
                    cot_content = gen_text.split("<think>")[1]
                elif "</think>" in gen_text:
                    cot_content = gen_text.split("</think>")[0]
                else:
                    cot_content = gen_text
                
                # Expert: Prompt + <think> + CoT + </think>
                # Note: prompts[sample_id] ends with .../think<|im_end|>\n<|im_start|>assistant\n
                # We append the extracted/generated thinking.
                expert_prompts[sample_id] = prompts[sample_id] + "<think>" + cot_content + "</think>"
                
                # Amateur: Assistant + <think> + CoT + </think> + \n\n
                amateur_prompts[sample_id] = f"<|im_start|>assistant\n<think>{cot_content}</think>\n\n"
                
                # Store CoT part for reconstruction: <think>...
                cot_parts[sample_id] = "<think>" + cot_content + "</think>"

            console.print("[Contrastive] Starting Phase 2: Contrastive Decoding", style=subhead_style_2)
            # Phase 2: Contrastive
            # Pass expert and amateur prompts
            contrast_gens, _, contrast_mfu = generator.generate_contrastive(
                expert_prompts, 
                amateur_prompts, 
                **kwargs
            )
            
            # Combine results
            generations = {}
            for sample_id, ans_list in contrast_gens.items():
                # Reconstruct: CoT + Answer
                full_gen = cot_parts[sample_id] + ans_list[0]
                generations[sample_id] = [full_gen]
                
            # Merge MFU stats if possible (simplified: just take Phase 2 or sum?)
            # For simplicity, we just use Phase 2 stats or the generator's internal tracking if refined
            logprobs = {} # Contrastive doesn't easily return logprobs of the difference
            
            # Update MFU stats to include Phase 1
            # (Optional: complex merge logic omitted for brevity, keeping Phase 2 stats which are critical for throughput)
            
        else:
            # Standard generation
            generations, logprobs = generator.generate(prompts, **kwargs)
        
        end_time = time.time()

        total_time = end_time - start_time
        num_samples = len(test_data)
        avg_time_per_sample = total_time / num_samples if num_samples > 0 else 0
        console.print(f"Total time: {total_time:.2f}s, Average per sample: {avg_time_per_sample:.4f}s")

        # 4. Collect hardware info and MFU statistics (for MFU calculation)
        console.print("[MFU DEBUG] Starting MFU data collection...")
        
        hardware_info = None
        mfu_stats = None
        
        try:
            # Check if generator has get_hardware_info method
            if not hasattr(generator, 'get_hardware_info'):
                console.print("[MFU ERROR] generator does NOT have get_hardware_info() method!")
                console.print(f"[MFU ERROR] Generator type: {type(generator)}")
                console.print(f"[MFU ERROR] Generator class: {generator.__class__.__name__}")
            else:
                hardware_info = generator.get_hardware_info()
                if hardware_info:
                    console.print(f"[MFU DEBUG] GPU Model: {hardware_info.get('gpu_model')}")
                    console.print(f"[MFU DEBUG] GPU Count: {hardware_info.get('gpu_count')}")
                    console.print(f"[MFU DEBUG] GPU TFLOPs: {hardware_info.get('gpu_tflops')}")
                else:
                    console.print("[MFU WARNING] hardware_info is None!")
            
            # Check if generator has mfu_stats attribute
            if not hasattr(generator, 'mfu_stats'):
                console.print("[MFU WARNING] generator does NOT have 'mfu_stats' attribute!")
            else:
                mfu_stats = getattr(generator, 'mfu_stats', None)
                if mfu_stats:
                    console.print(f"[MFU DEBUG] mfu_stats sample count: {len(mfu_stats)}")
                    if len(mfu_stats) > 0:
                        first_key = list(mfu_stats.keys())[0]
                        first_stats = mfu_stats[first_key]
                        console.print(f"[MFU DEBUG] First sample: {first_key}")
                        console.print(f"[MFU DEBUG]   input_tokens: {first_stats.get('input_tokens', 'MISSING')}")
                        console.print(f"[MFU DEBUG]   output_tokens: {first_stats.get('output_tokens', 'MISSING')}")
                        console.print(f"[MFU DEBUG]   times: {first_stats.get('times', 'MISSING')}")
                else:
                    console.print("[MFU WARNING] mfu_stats is None!")
                    
        except Exception as e:
            console.print(f"Warning: Failed to collect hardware info or MFU stats: {e}", style=warning_style)
        
        num_params_value = getattr(generator, 'num_params', None)
        console.print(f"[MFU DEBUG] num_params value: {num_params_value}")

        # 5. Save generation results
        self.save_generations(
            model_name=model_name,
            task_name=task_name,
            split=split,
            generations=generations,
            references=references,
            logprobs=logprobs,
            test_data=test_data,
            output_path=generation_file,
            total_time=total_time,
            avg_time_per_sample=avg_time_per_sample,
            hardware_info=hardware_info,
            mfu_stats=mfu_stats,
            num_params=getattr(generator, 'num_params', None),
        )
        
        console.print(f"Generation results saved to: {generation_file}")
        
        return None
    
    @staticmethod
    def save_generations(
        model_name: str,
        task_name: str,
        split: str,
        generations: Dict[str, List[str]],
        references: Dict[str, str],
        logprobs: Dict[str, List[float]],
        test_data: Dict[str, Dict[str, Any]],
        output_path: str,
        total_time: float,
        avg_time_per_sample: float,
        hardware_info: Optional[Dict[str, Any]] = None,
        mfu_stats: Optional[Dict[str, Dict[str, List[int]]]] = None,
        num_params: Optional[float] = None,
    ):
        """
        Save generation results (excluding evaluation metrics)
        
        Result format:
        {
            "model_name": "...",
            "task_name": "...",
            "split": "...",
            "total_time": "...",
            "avg_time_per_sample": "...",
            "samples": {
                "<sample_id>": {
                    "prompt": "...",
                    "generations": ["...", "..."],
                    "ground_truth": "...",
                    "metadata": {...}  # Contains metadata from original data
                },
                ...
            }
        }
        """
        # Check if this is a classification task (label_pred)
        is_classification_task = task_name == "label_pred"
        
        samples: Dict[str, Any] = {}
        for id, gens in generations.items():
            sample_data = {
                "prompt": test_data.get(id, {}).get("prompt", ""),
                "generations": gens,
                "ground_truth": references.get(id, ""),
            }

            if id in logprobs and logprobs[id]:
                sample_data["logprobs"] = logprobs[id]

            # Add MFU statistics for this sample (for MFU calculation)
            if mfu_stats and id in mfu_stats:
                sample_data["input_tokens"] = mfu_stats[id].get("input_tokens", [])
                sample_data["output_tokens"] = mfu_stats[id].get("output_tokens", [])
                sample_data["times"] = mfu_stats[id].get("times", [])

            if is_classification_task and id in test_data:
                metadata = test_data[id].get("metadata", {})
                if "uid" in metadata:
                    sample_data["user_id"] = metadata["uid"]

            if id in test_data and "metadata" in test_data[id]:
                sample_data["metadata"] = test_data[id]["metadata"]

            samples[id] = sample_data

        data = {
            "model_name": model_name,
            "task_name": task_name,
            "split": split,
            "total_time": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "samples": samples,
        }

        # Add hardware info and token statistics (for MFU calculation)
        if hardware_info:
            data["hardware_info"] = hardware_info
        else:
            console.print("[MFU DEBUG] ❌ Skipping hardware_info (None or empty)")

        if num_params:
            data["num_params"] = num_params
        else:
            console.print("[MFU DEBUG] ❌ Skipping num_params (None or 0)")

        # Save mfu_stats_aggregate for multi-stage MFU calculation
        # Compute aggregate statistics from per-sample mfu_stats
        if mfu_stats:
            # Determine number of stages from first sample
            num_stages = 0
            for sample_stats in mfu_stats.values():
                num_stages = len(sample_stats.get("input_tokens", []))
                console.print(f"[MFU DEBUG] Determined num_stages: {num_stages}")
                break

            # New structure: dict with lists instead of array of dicts
            data["mfu_stats_aggregate"] = {
                "total_input_tokens": [],
                "total_output_tokens": [],
                "total_time": []
            }

            for stage_idx in range(num_stages):
                total_input_tokens = 0
                total_output_tokens = 0

                # Aggregate token stats across all samples for this stage
                for sample_stats in mfu_stats.values():
                    input_tokens_list = sample_stats.get("input_tokens", [])
                    output_tokens_list = sample_stats.get("output_tokens", [])

                    if stage_idx < len(input_tokens_list):
                        total_input_tokens += input_tokens_list[stage_idx]
                    if stage_idx < len(output_tokens_list):
                        total_output_tokens += output_tokens_list[stage_idx]

                # Calculate stage time as max across all samples
                # Ray workers run in parallel, so stage time = slowest worker time
                stage_times = []
                for sample_stats in mfu_stats.values():
                    times_list = sample_stats.get("times", [])
                    if stage_idx < len(times_list):
                        stage_times.append(times_list[stage_idx])

                # Use max time if available, otherwise 0.0
                stage_time = max(stage_times) if stage_times else 0.0

                data["mfu_stats_aggregate"]["total_input_tokens"].append(total_input_tokens)
                data["mfu_stats_aggregate"]["total_output_tokens"].append(total_output_tokens)
                data["mfu_stats_aggregate"]["total_time"].append(stage_time)
                
        else:
            console.print("[MFU DEBUG] ❌ Skipping mfu_stats processing (None or empty)")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)