from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from transformers import HfArgumentParser

from benchmark import Benchmark
from benchmark.console import console, success_style
from benchmark.lead_decoding_generator import LeadDecodingGenerator


@dataclass
class ModelConfig:
    model_path: str = field(metadata={"help": "Model path or HuggingFace model name", "required": True})
    dtype: str = field(default="bfloat16", metadata={"help": "Model dtype"})


@dataclass
class BenchmarkConfig:
    task_types: Optional[List[str]] = field(default=None, metadata={"help": "Task name list"})
    sample_size: Optional[str] = field(default=None, metadata={"help": "Sample size or 'full'"})
    splits: List[str] = field(default_factory=lambda: ["test"], metadata={"help": "Dataset split list"})
    data_dir: str = field(default="./data", metadata={"help": "Benchmark data directory"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    overwrite: bool = field(default=False, metadata={"help": "Overwrite outputs"})


@dataclass
class PromptConfig:
    enable_thinking: bool = field(default=False, metadata={"help": "Enable benchmark thinking template"})


@dataclass
class LeadDecodingConfig:
    decode_mode: str = field(default="lead", metadata={"help": "vanilla, lead, or lead_step0_rerank"})
    latent_topk: int = field(default=64, metadata={"help": "Top-k support for latent mixture"})
    persistence_window: int = field(default=3, metadata={"help": "Minimum discrete residency before latent switch"})
    max_switches: int = field(default=5, metadata={"help": "Maximum number of mode switches"})
    num_return_sequences: int = field(default=32, metadata={"help": "Number of sequences to return"})
    max_new_tokens: int = field(default=3, metadata={"help": "Fixed itemic decoding length"})


def main() -> None:
    parser = HfArgumentParser([ModelConfig, BenchmarkConfig, PromptConfig, LeadDecodingConfig])
    model_config, benchmark_config, prompt_config, lead_config = parser.parse_args_into_dataclasses()

    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
    )

    generator = LeadDecodingGenerator(
        model_name_or_path=model_config.model_path,
        dtype=model_config.dtype,
        decode_mode=lead_config.decode_mode,
        latent_topk=lead_config.latent_topk,
        persistence_window=lead_config.persistence_window,
        max_switches=lead_config.max_switches,
        num_return_sequences=lead_config.num_return_sequences,
        max_new_tokens=lead_config.max_new_tokens,
    )

    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        enable_thinking=prompt_config.enable_thinking,
        num_return_sequences=lead_config.num_return_sequences,
        max_new_tokens=lead_config.max_new_tokens,
        sample_size=benchmark_config.sample_size,
    )

    task_name = benchmark_config.task_types[0] if benchmark_config.task_types else "unknown"
    generator.write_decode_stats(output_dir=Path(benchmark_config.output_dir), task_name=task_name)

    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=benchmark_config.task_types,
    )

    console.print(
        f"LEAD decoding evaluation finished: task={task_name}, mode={lead_config.decode_mode}",
        style=success_style,
    )


if __name__ == "__main__":
    main()
