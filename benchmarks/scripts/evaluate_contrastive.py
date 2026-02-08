from transformers import HfArgumentParser
import torch
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

# Add benchmark directory to sys.path to allow importing benchmark modules
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir) # benchmarks/ 
sys.path.append(benchmark_dir)

# Add ray-vllm to sys.path to allow importing from utils.arguments
ray_vllm_dir = os.path.join(current_dir, "ray-vllm")
sys.path.append(ray_vllm_dir)

from benchmark import Benchmark
from benchmark.console import *
from benchmark.contrastive_generator import ContrastiveGenerator
from utils.arguments import (
    ModelConfig,
    InfrastructureConfig,
    InferenceConfig,
    GenerationConfig,
    PromptConfig,
    BenchmarkConfig
)

@dataclass
class ContrastiveConfig:
    """Configuration for Contrastive Decoding"""
    alpha: float = field(
        default=0.5,
        metadata={"help": "Contrastive penalty alpha (Logits = Expert - alpha * Amateur)"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "Adaptive plausibility threshold"}
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new tokens to generate"}
    )

def main():
    parser = HfArgumentParser([
        ModelConfig,
        InfrastructureConfig,
        InferenceConfig,
        GenerationConfig,
        PromptConfig,
        BenchmarkConfig,
        ContrastiveConfig
    ])
    
    # We use parse_args_into_dataclasses to handle all args
    # Note: We import config classes from ray_vllm.utils.arguments to match the shell script args exactly.
    
    (model_config, infra_config, inference_config, generation_config, 
     prompt_config, benchmark_config, cd_config) = parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
    )
    
    # 2. Initialize ContrastiveGenerator
    # We pass relevant args from the configs to the generator
    generator = ContrastiveGenerator(
        model_name_or_path=model_config.model_path,
        dtype=model_config.dtype,
        alpha=cd_config.alpha,
        beta=cd_config.beta,
        max_new_tokens=cd_config.max_new_tokens,
        max_model_len=model_config.max_model_len,
        trust_remote_code=model_config.trust_remote_code,
        # We pass ignored args just in case generator wants to log them or strict check
        gpu_memory_utilization=infra_config.gpu_memory_utilization,
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        # Generation parameters
        enable_thinking=prompt_config.enable_thinking,
        sample_size=benchmark_config.sample_size,
        max_new_tokens=cd_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        num_return_sequences=generation_config.num_return_sequences,
    )
    
    # 4. Cleanup (optional for local script)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Calculate evaluation metrics
    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=benchmark_config.task_types
    )

if __name__ == "__main__":
    main()