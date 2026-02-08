import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from collections import defaultdict
from benchmark.console import *


# Global configuration: tasks that should disable optimizations (long prompts may cause issues)
# Used by vLLM-based generators to control chunked_prefill and prefix_caching
DISABLE_OPTIMIZATIONS_FOR_TASKS = ["rec_reason", "interactive"]


class Generator(ABC):
    """
    Abstract base class for generation models

    All generation models should inherit from this class.
    Subclasses must implement _generate_standard() to support the generate() method.
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Args:
            num_return_sequences: Number of candidates to generate per prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Other generation parameters
        """
        pass

    def __str__(self) -> str:
        """
        Return model name (for directory naming, remove path separators)

        This method is shared across all generator implementations.
        Subclasses must set self.model_name for this method to work.

        Returns:
            str: Model name
        """
        return os.path.basename(self.model_name.rstrip('/'))

    def generate(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Batch text generation

        Supports two-stage generation for recommendation tasks:
        - Stage 1: Generate thinking content with top_p/top_k sampling (if thinking enabled)
        - Stage 2: Generate SID sequences with beam search and prompt_token

        This method is shared across all generator implementations to reduce code duplication.
        Subclasses must implement _generate_standard() for this method to work.

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters (will override initialization parameters)

        Returns:
            Tuple of two dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [cum_logprob_1, cum_logprob_2, ...]} (only for beam search)
        """
        prompt_token = kwargs.get("prompt_token", None)
        enable_thinking = kwargs.get("enable_thinking", False)
        max_new_thinking_tokens = kwargs.get("max_new_thinking_tokens", None)
        target_tokens = kwargs.get("target_tokens", None)

        # Check if this is a classification task (has target_tokens parameter)
        is_classification = target_tokens is not None

        # Generation logic based on task type:
        # A: has max_new_thinking_tokens + has prompt_token (recommendation tasks)
        # B: has max_new_thinking_tokens + no prompt_token (caption tasks)
        # C: no max_new_thinking_tokens (standard tasks)
        # D: classification task + no think
        # E: classification task + think
        if is_classification:
            # Classification task scenarios (D & E)
            if enable_thinking:
                # E: Classification with thinking
                console.print(
                    f"Two-stage classification with thinking enabled: thinking (max_new_thinking_tokens={max_new_thinking_tokens}) + logprobs extraction for {target_tokens}",
                    style=warning_style,
                )
                return self._generate_two_stage_classification_with_thinking(prompts, **kwargs)
            else:
                # D: Classification without thinking
                console.print(
                    f"Classification task: extracting logprobs for tokens {target_tokens}",
                    style=warning_style,
                )
                # Remove target_tokens from kwargs to avoid passing it twice
                kwargs_classification = kwargs.copy()
                kwargs_classification.pop("target_tokens", None)
                results, _, mfu_stats = self.extract_token_logprobs(prompts, target_tokens, **kwargs_classification)

                self.mfu_stats = mfu_stats
                return results, {}, mfu_stats
        elif max_new_thinking_tokens:
            if enable_thinking:
                # A & B with thinking: two-stage generation
                console.print(
                    f"Two-stage generation enabled: thinking (max_new_thinking_tokens={max_new_thinking_tokens}) + prompt_token ({prompt_token})",
                    style=warning_style,
                )
                results, logprobs, mfu_stats = self._generate_two_stage_with_thinking(prompts, **kwargs)
                self.mfu_stats = mfu_stats
                return results, logprobs, mfu_stats
            else:
                # A & B without thinking
                if prompt_token:
                    # A without thinking: single-stage with prompt_token (beam search)
                    console.print(
                        f"Single-stage generation with prompt_token ({prompt_token})",
                        style=warning_style,
                    )
                    prompts_with_token = {
                        sample_id: prompt + prompt_token
                        for sample_id, prompt in prompts.items()
                    }
                    results, logprobs, mfu_stats = self._generate_standard(prompts_with_token, **kwargs)
                    self.mfu_stats = mfu_stats
                    return results, logprobs
                else:
                    # B without thinking: single-stage sampling
                    console.print(
                        f"Warning: max_new_thinking_tokens={max_new_thinking_tokens} is set but "
                        f"enable_thinking=False and prompt_token=None. The max_new_thinking_tokens parameter will be ignored.",
                        style=warning_style,
                    )
                    results, logprobs, mfu_stats = self._generate_standard(prompts, **kwargs)
                    self.mfu_stats = mfu_stats
                    return results, logprobs
        else:
            # C: standard single-stage sampling
            results, logprobs, mfu_stats = self._generate_standard(prompts, **kwargs)
            self.mfu_stats = mfu_stats
            return results, logprobs


    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get GPU hardware information for MFU calculation

        Default implementation that works for all generators.
        Handles both single-machine and Ray-based multi-machine setups.

        Returns:
            Dictionary containing:
            - gpu_model: str, GPU model name
            - gpu_count: int, total number of GPUs used
            - gpu_tflops: float, theoretical peak TFLOPS for BF16/FP16
            - tensor_parallel_size: int, tensor parallelism size
            - gpu_memory_total_gb: float, total GPU memory in GB
        """
        from benchmark.gpu_utils import get_gpu_info

        gpu_info = get_gpu_info()

        # Calculate total GPU count
        tensor_parallel_size = getattr(self, 'tensor_parallel_size', 1)

        # For Ray-based generators, multiply by number of workers
        if hasattr(self, 'workers') and self.workers:
            num_workers = len(self.workers)
            total_gpus = num_workers * tensor_parallel_size
        else:
            # For single-machine generators
            total_gpus = tensor_parallel_size

        gpu_info["gpu_count"] = total_gpus
        gpu_info["tensor_parallel_size"] = tensor_parallel_size

        # Add worker info for Ray-based generators
        if hasattr(self, 'workers'):
            gpu_info["num_workers"] = len(self.workers) if self.workers else 0

        return gpu_info

    def _generate_two_stage_with_thinking(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Two-stage generation with thinking mode

        Stage 1: Generate thinking content with top_p/top_k sampling until </think>
        Stage 2: Continue generation (with prompt_token if provided, beam search or sampling)

        This method is shared across all generator implementations to reduce code duplication.
        Subclasses must implement _generate_standard() for this method to work.

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters

        Returns:
            Tuple of two dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [cum_logprob_1, cum_logprob_2, ...]} (only for beam search)
        """
        prompt_token = kwargs.get("prompt_token", None)
        console.print(
            "Stage 1/2: Generating thinking content with top_p/top_k sampling...",
            style=warning_style,
        )

        # Stage 1: Build kwargs for thinking generation (remove beam search, add stop)
        kwargs_stage1 = kwargs.copy()
        kwargs_stage1.pop("num_beams", None)  # Remove beam search to force sampling mode
        kwargs_stage1["stop"] = ["</think>"]  # Stop at </think> tag

        # Use num_return_thinking_sequences for stage 1 if specified
        num_return_thinking = kwargs.get("num_return_thinking_sequences", 1)
        kwargs_stage1["num_return_sequences"] = num_return_thinking

        # Use max_new_thinking_tokens for stage 1 if specified
        max_new_thinking_tokens = kwargs.get("max_new_thinking_tokens", 1000)
        kwargs_stage1["max_new_tokens"] = max_new_thinking_tokens

        # Call _generate_standard for stage 1 (ignoring logprobs as they're not used)
        stage1_results, _, stage1_mfu_stats = self._generate_standard(prompts, **kwargs_stage1)

        # Prepare prompts for stage 2 by appending thinking + prompt_token
        # Each sample will have multiple thinking candidates
        stage2_prompts = {}
        sample_to_thinking_count = {}  # Track how many thinking candidates each sample has

        for sample_id, thinking_list in stage1_results.items():
            # Use ALL thinking candidates (not just the first one)
            sample_to_thinking_count[sample_id] = len(thinking_list)

            for idx, thinking_text in enumerate(thinking_list):
                # Create unique ID for each thinking candidate
                thinking_sample_id = f"{sample_id}_thinking_{idx}"

                # Append </think> + prompt_token (if provided)
                # If model didn't generate </think>, treat entire output as thinking
                if prompt_token:
                    full_thinking = thinking_text + "</think>\n" + prompt_token
                else:
                    full_thinking = thinking_text + "</think>\n"
                stage2_prompt = prompts[sample_id] + full_thinking
                stage2_prompts[thinking_sample_id] = stage2_prompt

        # Stage 2: Determine generation mode based on num_beams
        kwargs_stage2 = kwargs.copy()
        original_num_sequences = kwargs.get("num_return_sequences", 1)
        original_num_beams = kwargs.get("num_beams", None)

        # Determine if stage 2 uses beam search or sampling
        use_beam_search_stage2 = original_num_beams is not None

        if use_beam_search_stage2:
            # Beam search mode: num_beams is directly used per thinking candidate
            beams_per_thinking = original_num_beams

            # Validate configuration: total sequences should match
            if original_num_sequences != beams_per_thinking * num_return_thinking:
                raise ValueError(
                    f"Configuration error: num_return_sequences ({original_num_sequences}) must equal "
                    f"num_beams ({beams_per_thinking}) * num_return_thinking_sequences ({num_return_thinking}) = "
                    f"{beams_per_thinking * num_return_thinking}. "
                    f"Please adjust your parameters accordingly."
                )

            kwargs_stage2["num_return_sequences"] = beams_per_thinking
            kwargs_stage2["num_beams"] = beams_per_thinking

            console.print(
                f"Stage 2/2: Generating sequences with beam search for {len(stage2_prompts)} thinking candidates...",
                style=warning_style,
            )
            console.print(
                f"Each thinking candidate will use beam_width={beams_per_thinking}, return {beams_per_thinking} sequences "
                f"({num_return_thinking} thinking × {beams_per_thinking} = {num_return_thinking * beams_per_thinking} total per sample)",
                style=warning_style,
            )
        else:
            # Sampling mode: each thinking generates 1 result
            kwargs_stage2["num_return_sequences"] = 1
            kwargs_stage2.pop("num_beams", None)  # Remove num_beams to use sampling

            console.print(
                f"Stage 2/2: Generating sequences with sampling for {len(stage2_prompts)} thinking candidates...",
                style=warning_style,
            )
            console.print(
                f"Each thinking candidate will generate 1 sequence "
                f"({num_return_thinking} thinking × 1 = {num_return_thinking} total per sample)",
                style=warning_style,
            )

        # Call _generate_standard for stage 2
        stage2_results, stage2_logprobs, stage2_mfu_stats = self._generate_standard(stage2_prompts, **kwargs_stage2)

        # Merge mfu_stats from both stages
        self.mfu_stats = {}
        for sample_id, stats in stage1_mfu_stats.items():
            self.mfu_stats[sample_id] = {
                "input_tokens": stats["input_tokens"].copy(),
                "output_tokens": stats["output_tokens"].copy(),
                "times": stats["times"].copy()
            }

        # Group stage2 stats by original_id first
        stage2_by_original = defaultdict(lambda: {"input_tokens": [], "output_tokens": [], "times": []})
        for thinking_id, stats in stage2_mfu_stats.items():
            original_id = thinking_id.rsplit("_thinking_", 1)[0]
            stage2_by_original[original_id]["input_tokens"].extend(stats["input_tokens"])
            stage2_by_original[original_id]["output_tokens"].extend(stats["output_tokens"])
            stage2_by_original[original_id]["times"].extend(stats["times"])

        # Aggregate: sum tokens, max time
        for original_id, stats in stage2_by_original.items():
            self.mfu_stats[original_id]["input_tokens"].append(sum(stats["input_tokens"]))
            self.mfu_stats[original_id]["output_tokens"].append(sum(stats["output_tokens"]))
            self.mfu_stats[original_id]["times"].append(max(stats["times"]))

        # Merge results back by original sample_id
        # Combine thinking + prompt_token + SID into final generation
        final_results = defaultdict(list)
        final_logprobs = defaultdict(list)

        for thinking_sample_id, sid_sequences in stage2_results.items():
            # Extract original sample_id and thinking index
            # Format: "sampleID_thinking_N"
            parts = thinking_sample_id.rsplit("_thinking_", 1)
            original_sample_id = parts[0]
            thinking_idx = int(parts[1])

            # Get the corresponding thinking text from stage 1
            thinking_text = stage1_results[original_sample_id][thinking_idx]

            # Combine thinking + prompt_token + SID for each sequence
            for sid_seq in sid_sequences:
                # Format: <think>thinking_text</think>\n<|sid_begin|>sid_sequence
                combined = f"{thinking_text}</think>\n{prompt_token or ''}{sid_seq}"
                final_results[original_sample_id].append(combined)

            # Also merge logprobs if available (from stage 2 beam search)
            if thinking_sample_id in stage2_logprobs:
                final_logprobs[original_sample_id].extend(stage2_logprobs[thinking_sample_id])

        return (dict(final_results), dict(final_logprobs), self.mfu_stats)

    def _generate_two_stage_classification_with_thinking(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Two-stage generation for classification tasks with thinking mode

        Stage 1: Generate thinking content with top_p/top_k sampling until </think>
        Stage 2: Extract logprobs for target tokens for each thinking candidate

        This method is shared across all generator implementations to reduce code duplication.
        Subclasses must implement _generate_standard() and extract_token_logprobs() for this method to work.

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters

        Returns:
            Tuple of two dicts:
            - First dict: {sample_id: ["<think>thinking_1</think>\n{'是': 0.8, '否': 0.2}", ...]}
            - Second dict: {} (empty, no logprobs for classification)
        """
        # target_tokens is guaranteed to be in kwargs (checked in generate() method)
        target_tokens = kwargs["target_tokens"]

        console.print(
            "Stage 1/2: Generating thinking content with top_p/top_k sampling...",
            style=warning_style,
        )

        # Stage 1: Build kwargs for thinking generation (remove beam search, add stop)
        kwargs_stage1 = kwargs.copy()
        kwargs_stage1.pop("num_beams", None)  # Remove beam search to force sampling mode
        kwargs_stage1.pop("target_tokens", None)  # Remove target_tokens for stage 1
        kwargs_stage1["stop"] = ["</think>"]  # Stop at </think> tag

        # Use num_return_thinking_sequences for stage 1 if specified
        num_return_thinking = kwargs.get("num_return_thinking_sequences", 1)
        kwargs_stage1["num_return_sequences"] = num_return_thinking

        # Use max_new_thinking_tokens for stage 1 if specified
        max_new_thinking_tokens = kwargs.get("max_new_thinking_tokens", 1000)
        kwargs_stage1["max_new_tokens"] = max_new_thinking_tokens

        # Call _generate_standard for stage 1 (ignoring logprobs as they're not used)
        stage1_results, _, stage1_mfu_stats = self._generate_standard(prompts, **kwargs_stage1)

        # Prepare prompts for stage 2 by appending thinking + </think>
        # Each sample will have multiple thinking candidates
        stage2_prompts = {}
        sample_to_thinking_count = {}  # Track how many thinking candidates each sample has

        for sample_id, thinking_list in stage1_results.items():
            # Use ALL thinking candidates (not just the first one)
            sample_to_thinking_count[sample_id] = len(thinking_list)

            for idx, thinking_text in enumerate(thinking_list):
                # Create unique ID for each thinking candidate
                thinking_sample_id = f"{sample_id}_thinking_{idx}"

                # Append </think> to complete the thinking tag
                full_thinking = thinking_text + f"</think>\n"
                stage2_prompt = prompts[sample_id] + full_thinking
                stage2_prompts[thinking_sample_id] = stage2_prompt

        console.print(
            f"Stage 2/2: Extracting logprobs for {len(stage2_prompts)} thinking candidates...",
            style=warning_style,
        )
        console.print(
            f"Each thinking candidate will extract logprobs for tokens {target_tokens} "
            f"({num_return_thinking} thinking total per sample)",
            style=warning_style,
        )

        # Build kwargs for stage 2 (remove target_tokens to avoid duplication)
        kwargs_stage2 = kwargs.copy()
        kwargs_stage2.pop("target_tokens", None)

        # Call extract_token_logprobs for stage 2
        stage2_probs, _, stage2_mfu_stats = self.extract_token_logprobs(stage2_prompts, target_tokens, **kwargs_stage2)

        # Merge mfu_stats from both stages
        self.mfu_stats = {}
        for sample_id, stats in stage1_mfu_stats.items():
            self.mfu_stats[sample_id] = {
                "input_tokens": stats["input_tokens"].copy(),
                "output_tokens": stats["output_tokens"].copy(),
                "times": stats["times"].copy()
            }

        # Group stage2 stats by original_id first
        stage2_by_original = defaultdict(lambda: {"input_tokens": [], "output_tokens": [], "times": []})
        for thinking_id, stats in stage2_mfu_stats.items():
            original_id = thinking_id.rsplit("_thinking_", 1)[0]
            stage2_by_original[original_id]["input_tokens"].extend(stats["input_tokens"])
            stage2_by_original[original_id]["output_tokens"].extend(stats["output_tokens"])
            stage2_by_original[original_id]["times"].extend(stats["times"])

        # Aggregate: sum tokens, max time
        for original_id, stats in stage2_by_original.items():
            self.mfu_stats[original_id]["input_tokens"].append(sum(stats["input_tokens"]))
            self.mfu_stats[original_id]["output_tokens"].append(sum(stats["output_tokens"]))
            self.mfu_stats[original_id]["times"].append(max(stats["times"]))

        # Merge results back by original sample_id
        # Combine thinking + probabilities into final generation
        final_results = defaultdict(list)

        for thinking_sample_id, json_str_list in stage2_probs.items():
            # Extract original sample_id and thinking index
            # Format: "sampleID_thinking_N"
            parts = thinking_sample_id.rsplit("_thinking_", 1)
            original_sample_id = parts[0]
            thinking_idx = int(parts[1])

            # Get the corresponding thinking text from stage 1
            thinking_text = stage1_results[original_sample_id][thinking_idx]

            # Extract JSON string from list (extract_token_logprobs returns [json_str])
            json_str = json_str_list[0]

            # Combine thinking + probabilities (json_str is already formatted)
            # Format: "<think>thinking_text</think>\n{\"是\": 0.8, \"否\": 0.2}"
            combined = f"{thinking_text}</think>\n{json_str}"
            final_results[original_sample_id].append(combined)

        return (dict(final_results), {}, self.mfu_stats)


class HfTransformersMixin:
    """
    Mixin for HuggingFace Transformers functionality
    
    Provides common parameter building logic for HuggingFace Transformers generate() API.
    This mixin can be combined with Generator or RayMixin to create HuggingFace-based generators.
    """
    
    def _build_sampling_params(self, **kwargs) -> tuple:
        """
        Build HuggingFace sampling/generation parameters
        
        Args:
            **kwargs: Optional parameters to override default values
        
        Returns:
            Tuple of (gen_kwargs dict, stop_sequences list)
        """
        n = kwargs.get("num_return_sequences")
        max_tokens = kwargs.get("max_new_tokens")
        num_beams = kwargs.get("num_beams", None)
        use_beam_search = num_beams is not None
        
        stop_sequences = kwargs.get("stop", [])
        
        if use_beam_search:
            # Beam search mode
            if n and n > num_beams:
                raise ValueError(
                    f"num_return_sequences ({n}) cannot be greater than num_beams ({num_beams}). "
                    f"Beam search can only return at most {num_beams} sequences. "
                    f"Please set num_return_sequences <= num_beams or increase num_beams."
                )

            gen_kwargs = {
                "num_beams": num_beams,
                "num_return_sequences": n if n else num_beams,
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            if "repetition_penalty" in kwargs:
                gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        else:
            # Sampling mode
            gen_kwargs = {
                "num_return_sequences": n,
                "max_new_tokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", -1),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "do_sample": kwargs.get("do_sample", True),
            }
        
        return gen_kwargs, stop_sequences


class VllmMixin:
    """
    Mixin for vLLM functionality
    
    Provides common parameter building logic for vLLM generate() API.
    This mixin can be combined with Generator or RayMixin to create vLLM-based generators.
    """
    
    def _build_sampling_params(self, **kwargs):
        """
        Build vLLM sampling parameters
        
        Args:
            **kwargs: Optional parameters to override default values
        
        Returns:
            SamplingParams or BeamSearchParams object
        """
        from vllm import SamplingParams
        from vllm.sampling_params import BeamSearchParams
        
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", -1)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        max_tokens = kwargs.get("max_new_tokens")
        n = kwargs.get("num_return_sequences", 1)
        stop = kwargs.get("stop", None)
        
        num_beams = kwargs.get("num_beams", None)
        use_beam_search = num_beams  is not None
        
        if use_beam_search:
            # Beam search: set beam_width to max(num_beams, n)
            actual_beam_width = max(num_beams, n)
            params = BeamSearchParams(
                beam_width=actual_beam_width,
                max_tokens=max_tokens,
            )
        else:
            # Sampling mode
            params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                stop=stop,
            )
        
        return params

    
    def _should_enable_optimizations(self) -> bool:
        """
        Determine whether to enable optimizations based on task types and force flags
        
        This method is primarily used by vLLM-based generators to control
        chunked_prefill and prefix_caching optimizations.
        
        Returns:
            True if should enable optimizations, False otherwise
        """
        # Priority 1: Force flags
        if self.force_enable_optimizations:
            return True
        if self.force_disable_optimizations:
            return False
        
        # Priority 2: Check if any task in task_types requires disabling optimizations
        if hasattr(self, 'task_types') and self.task_types:
            for task_type in self.task_types:
                if task_type in DISABLE_OPTIMIZATIONS_FOR_TASKS:
                    return False
        
        # Default: enable optimizations
        return True


class RayMixin:
    """
    Mixin for Ray distributed computing functionality
    
    Provides Ray cluster management, GPU allocation, and resource cleanup
    for distributed generators. This is a mixin class designed to be combined
    with other generator classes using multiple inheritance.
    """
    
    def _initialize_ray_cluster(self):
        """Initialize Ray cluster connection"""
        import ray

        if ray.is_initialized():
            console.print(
                "  ✓ Ray already initialized",
                style=success_style,
            )
            return

        console.print(
            "  Initializing Ray cluster connection...",
            style=subhead_style_2,
        )

        # Determine connection mode
        if self.ray_address == "local":
            # Local mode (single machine)
            ray.init(ignore_reinit_error=True)
            console.print(
                "  ✓ Ray initialized in local mode",
                style=success_style,
            )
        elif self.ray_address == "auto":
            # Auto-detect mode
            try:
                ray.init(address="auto", ignore_reinit_error=True)
                console.print(
                    "  ✓ Ray connected to existing cluster (auto-detected)",
                    style=success_style,
                )
            except:
                # Fallback to local mode
                console.print(
                    "  [yellow]No existing cluster found, initializing local mode...[/yellow]",
                    style=warning_style,
                )
                ray.init(ignore_reinit_error=True)
                console.print(
                    "  ✓ Ray initialized in local mode",
                    style=success_style,
                )
        else:
            # Specific address
            ray.init(address=self.ray_address, ignore_reinit_error=True)
            console.print(
                f"  ✓ Ray connected to cluster at {self.ray_address}",
                style=success_style,
            )
    
    def _determine_gpu_ids_from_cluster(self) -> List[Dict[str, Any]]:
        """
        Determine GPU resources from Ray cluster
        
        Returns:
            List of GPU info dicts: [{"node_id": str, "gpu_index": int}, ...]
        """
        import ray
        
        # Get all nodes in cluster
        nodes = ray.nodes()
        
        # Collect GPU information from all nodes
        gpu_list = []
        
        for node in nodes:
            if not node['Alive']:
                continue
            
            node_id = node['NodeID']
            node_resources = node.get('Resources', {})
            
            # Count GPUs on this node
            num_gpus_on_node = int(node_resources.get('GPU', 0))
            
            if num_gpus_on_node > 0:
                # Add GPU entries for this node
                for gpu_idx in range(num_gpus_on_node):
                    gpu_list.append({
                        "node_id": node_id,
                        "node_ip": node.get('NodeManagerAddress', 'unknown'),
                        "gpu_index": gpu_idx,
                        "global_index": len(gpu_list)  # Global GPU index across cluster
                    })
        
        if not gpu_list:
            raise RuntimeError("No GPUs detected in Ray cluster")
        
        # Apply user filters if specified
        if self.gpu_ids is not None:
            # In cluster mode, gpu_ids refers to global indices
            filtered_list = []
            for idx in self.gpu_ids:
                if idx < len(gpu_list):
                    filtered_list.append(gpu_list[idx])
                else:
                    console.print(
                        f"  [yellow]Warning:[/yellow] GPU index {idx} out of range (max: {len(gpu_list)-1}), skipping",
                        style=warning_style,
                    )
            gpu_list = filtered_list
        elif self.num_gpus is not None:
            # Limit to first num_gpus
            if self.num_gpus < len(gpu_list):
                gpu_list = gpu_list[:self.num_gpus]
            elif self.num_gpus > len(gpu_list):
                console.print(
                    f"  [yellow]Warning:[/yellow] Requested {self.num_gpus} GPUs, but only {len(gpu_list)} available in cluster",
                    style=warning_style,
                )
        
        return gpu_list
    
    def _group_gpus_for_workers(
        self,
        gpu_list: List[Dict[str, Any]],
        tensor_parallel_size: int
    ) -> tuple:
        """
        Group GPUs for workers, ensuring same-node constraint for tensor parallelism
        
        Args:
            gpu_list: List of GPU info dicts
            tensor_parallel_size: Number of GPUs per worker
        
        Returns:
            (worker_gpu_groups, worker_node_assignments)
            - worker_gpu_groups: List of GPU index lists for each worker
            - worker_node_assignments: List of node IDs for each worker
        """
        if len(gpu_list) % tensor_parallel_size != 0:
            raise ValueError(
                f"Number of GPUs ({len(gpu_list)}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
            )
        
        num_workers = len(gpu_list) // tensor_parallel_size
        worker_gpu_groups = []
        worker_node_assignments = []
        
        if tensor_parallel_size == 1:
            # Simple case: one GPU per worker
            for gpu_info in gpu_list:
                worker_gpu_groups.append([gpu_info["gpu_index"]])
                worker_node_assignments.append(gpu_info["node_id"])
        else:
            # Complex case: multiple GPUs per worker
            # Need to ensure all GPUs in a group are on the same node
            
            if not self.allow_cross_node_tensor_parallel:
                # Group by node first
                node_to_gpus = {}
                for gpu_info in gpu_list:
                    node_id = gpu_info["node_id"]
                    if node_id not in node_to_gpus:
                        node_to_gpus[node_id] = []
                    node_to_gpus[node_id].append(gpu_info)
                
                # Create workers from each node
                for node_id, node_gpus in node_to_gpus.items():
                    # Group GPUs on this node
                    for i in range(0, len(node_gpus), tensor_parallel_size):
                        if i + tensor_parallel_size <= len(node_gpus):
                            gpu_group = [gpu["gpu_index"] for gpu in node_gpus[i:i+tensor_parallel_size]]
                            worker_gpu_groups.append(gpu_group)
                            worker_node_assignments.append(node_id)
                
                if len(worker_gpu_groups) != num_workers:
                    raise ValueError(
                        f"Cannot create {num_workers} workers with tensor_parallel_size={tensor_parallel_size} "
                        f"while ensuring same-node constraint. Got {len(worker_gpu_groups)} workers instead. "
                        f"Try setting --allow_cross_node_tensor_parallel or adjust tensor_parallel_size."
                    )
            else:
                # Allow cross-node tensor parallel (not recommended)
                console.print(
                    "  [yellow]Warning: Cross-node tensor parallelism enabled. This may cause performance degradation.[/yellow]",
                    style=warning_style,
                )
                for i in range(num_workers):
                    start_idx = i * tensor_parallel_size
                    end_idx = start_idx + tensor_parallel_size
                    gpu_group = [gpu_list[j]["gpu_index"] for j in range(start_idx, end_idx)]
                    worker_gpu_groups.append(gpu_group)
                    # Use first GPU's node as primary node
                    worker_node_assignments.append(gpu_list[start_idx]["node_id"])
        
        return worker_gpu_groups, worker_node_assignments
    
    def _display_cluster_info(self, gpu_list: List[Dict[str, Any]], num_workers: int):
        """Display cluster and GPU information"""
        import ray

        # Get cluster info
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n['Alive']]

        console.print(
            f"  Cluster nodes: [green]{len(alive_nodes)}[/green]",
            style=subhead_style_2,
        )

        # Group GPUs by node
        node_gpu_count = {}
        for gpu_info in gpu_list:
            node_ip = gpu_info["node_ip"]
            node_gpu_count[node_ip] = node_gpu_count.get(node_ip, 0) + 1

        for node_ip, count in node_gpu_count.items():
            console.print(
                f"    - Node {node_ip}: {count} GPU(s)",
                style=subhead_style_2,
            )

        console.print(
            f"  Total GPUs: [green]{len(gpu_list)}[/green]",
            style=subhead_style_2,
        )
        console.print(
            f"  Tensor Parallel Size: [green]{self.tensor_parallel_size}[/green]",
            style=subhead_style_2,
        )
        console.print(
            f"  Worker count: [green]{num_workers}[/green]",
            style=subhead_style_2,
        )

        # Display worker assignments
        console.print(
            f"  Worker GPU assignments:",
            style=subhead_style_2,
        )
        for i, (gpu_group, node_id) in enumerate(zip(self.worker_gpu_groups, self.worker_node_assignments)):
            # Find node IP for this node_id
            node_ip = "unknown"
            for gpu_info in gpu_list:
                if gpu_info["node_id"] == node_id:
                    node_ip = gpu_info["node_ip"]
                    break
            console.print(
                f"    - Worker {i}: GPUs {gpu_group} on node {node_ip}",
                style=subhead_style_2,
            )
    
    def cleanup(self):
        """
        Explicitly cleanup resources and release GPU memory
        
        Called after generation tasks complete to release GPU memory occupied by Ray Workers.
        This is useful for avoiding OOM errors during subsequent metric calculations.
        """
        import ray
        
        console.print(
            "\nReleasing Ray Workers and resources...",
            style=warning_style,
        )
        
        try:
            # 1. Cleanup all Workers
            if hasattr(self, 'workers') and self.workers:
                for i, worker in enumerate(self.workers):
                    try:
                        ray.kill(worker)
                        console.print(
                            f"  ✓ Worker {i} terminated",
                            style=success_style,
                        )
                    except Exception as e:
                        console.print(
                            f"  ⚠ Worker {i} cleanup failed: {e}",
                            style=err_style,
                        )
                self.workers = []

            # 2. Shut down Ray (optional)
            if ray.is_initialized():
                console.print(
                    "  Shutting down Ray...",
                    style=subhead_style_2,
                )
                ray.shutdown()
                console.print(
                    "  ✓ Ray shut down",
                    style=subhead_style_2,
                )

            console.print(
                "✓ Resource cleanup completed\n",
                style=success_style,
            )

        except Exception as e:
            console.print(
                f"✗ Cleanup process error: {e}",
                style=err_style,
            )