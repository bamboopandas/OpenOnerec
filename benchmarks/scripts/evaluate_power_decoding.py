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

from benchmark import Benchmark
from benchmark.console import *
from benchmark.power_decoding_generator import PowerDecodingGenerator

@dataclass
class ModelConfig:
    """Model loading and initialization parameters"""
    model_path: str = field(
        metadata={"help": "Model path or HuggingFace model name (e.g., Qwen/Qwen2-7B)", "required": True}
    )
    dtype: str = field(
        default='bfloat16',
        metadata={"help": "Model data type: auto, half, float16, bfloat16, float, float32"}
    )
    
@dataclass
class BenchmarkConfig:
    """Benchmark execution and evaluation parameters"""
    task_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Task name list (e.g., item_understand rec_reason)"}
    )
    sample_size: Optional[str] = field(
        default=None,
        metadata={"help": "Sample size for evaluation (e.g., 'full' for all data, or a number like '100')"}
    )
    splits: List[str] = field(
        default_factory=lambda: ['test'],
        metadata={"help": "Dataset split list"}
    )
    data_dir: str = field(
        default='./data',
        metadata={"help": "Data directory path"}
    )
    output_dir: str = field(
        default='./results',
        metadata={"help": "Output directory for results"}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing results"}
    )

@dataclass
class PromptConfig:
    """Prompt formatting and template parameters"""
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for apply_chat_template (overrides task config if set)"}
    )

@dataclass
class PowerDecodingConfig:
    """Configuration for Future-Aware Power Decoding"""
    alpha: float = field(
        default=2.0,
        metadata={"help": "Sharpening exponent alpha"}
    )
    top_k_candidates: int = field(
        default=5,
        metadata={"help": "Number of candidates for reweighting"}
    )
    max_rollouts: int = field(
        default=5,
        metadata={"help": "Max number of rollouts per candidate"}
    )
    max_lookahead: int = field(
        default=3,
        metadata={"help": "Max lookahead steps"}
    )
    crit_threshold: float = field(
        default=0.5,
        metadata={"help": "Entropy threshold for critical step detection"}
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new tokens to generate"}
    )

def main():
    parser = HfArgumentParser([
        ModelConfig,
        BenchmarkConfig,
        PromptConfig,
        PowerDecodingConfig
    ])
    model_config, benchmark_config, prompt_config, pd_config = \
        parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
    )
    
    # 2. Initialize PowerDecodingGenerator
    generator = PowerDecodingGenerator(
        model_name_or_path=model_config.model_path,
        dtype=model_config.dtype,
        alpha=pd_config.alpha,
        top_k_candidates=pd_config.top_k_candidates,
        max_rollouts=pd_config.max_rollouts,
        max_lookahead=pd_config.max_lookahead,
        crit_threshold=pd_config.crit_threshold,
        max_new_tokens=pd_config.max_new_tokens
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        # Generation parameters
        enable_thinking=prompt_config.enable_thinking,
        sample_size=benchmark_config.sample_size,
        max_new_tokens=pd_config.max_new_tokens
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
