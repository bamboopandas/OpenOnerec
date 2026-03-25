from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from transformers import HfArgumentParser

current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)
sys.path.append(benchmark_dir)

from benchmark import Benchmark
from benchmark.console import console, success_style
from benchmark.thinking_lead_generator import ThinkingLeadGenerator, resolve_candidate_budget


@dataclass
class ModelConfig:
    model_path: str = field(metadata={"help": "Model path or HuggingFace model name", "required": True})
    dtype: str = field(default="bfloat16", metadata={"help": "Model dtype"})


@dataclass
class BenchmarkConfig:
    task_types: Optional[List[str]] = field(default=None, metadata={"help": "Recommendation task names"})
    sample_size: Optional[str] = field(default=None, metadata={"help": "Sample size or 'full'"})
    splits: List[str] = field(default_factory=lambda: ["test"], metadata={"help": "Dataset split list"})
    data_dir: str = field(default="./data", metadata={"help": "Benchmark data directory"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    overwrite: bool = field(default=False, metadata={"help": "Overwrite outputs"})


@dataclass
class ThinkingLeadConfig:
    stage1_decode_mode: str = field(default="lead", metadata={"help": "vanilla or lead"})
    latent_topk: int = field(default=64, metadata={"help": "Top-k support for latent mixture"})
    persistence_window: int = field(default=3, metadata={"help": "Discrete residency before switching to latent"})
    max_switches: int = field(default=5, metadata={"help": "Maximum number of stage-1 mode switches"})
    initial_discrete_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Initial discrete residency before the first possible latent switch"},
    )
    initial_entropy_ratio_threshold: float = field(
        default=0.64,
        metadata={"help": "Initial entropy-ratio threshold for the first latent switch"},
    )
    discrete_to_latent_margin: float = field(
        default=0.0,
        metadata={"help": "Required entropy margin above reference to switch from discrete to latent"},
    )
    latent_to_discrete_margin: float = field(
        default=0.0,
        metadata={"help": "Required entropy drop below reference to switch from latent back to discrete"},
    )
    candidate_budget: int = field(default=32, metadata={"help": "Candidate budget: 32 or 128"})
    max_new_thinking_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Optional runtime override for stage-1 max_new_thinking_tokens"},
    )


def main() -> None:
    parser = HfArgumentParser([ModelConfig, BenchmarkConfig, ThinkingLeadConfig])
    model_config, benchmark_config, lead_config = parser.parse_args_into_dataclasses()

    selected_tasks = benchmark_config.task_types or ["ad", "product", "video"]
    candidate_config = resolve_candidate_budget(lead_config.candidate_budget)

    generator = ThinkingLeadGenerator(
        model_name_or_path=model_config.model_path,
        dtype=model_config.dtype,
        stage1_decode_mode=lead_config.stage1_decode_mode,
        latent_topk=lead_config.latent_topk,
        persistence_window=lead_config.persistence_window,
        max_switches=lead_config.max_switches,
        initial_discrete_steps=lead_config.initial_discrete_steps,
        initial_entropy_ratio_threshold=lead_config.initial_entropy_ratio_threshold,
        discrete_to_latent_margin=lead_config.discrete_to_latent_margin,
        latent_to_discrete_margin=lead_config.latent_to_discrete_margin,
    )

    for task_name in selected_tasks:
        benchmark = Benchmark(
            model_path=model_config.model_path,
            task_types=[task_name],
            splits=benchmark_config.splits,
            data_dir=benchmark_config.data_dir,
            enable_thinking=True,
        )

        benchmark.run(
            generator=generator,
            output_dir=benchmark_config.output_dir,
            overwrite=benchmark_config.overwrite,
            enable_thinking=True,
            sample_size=benchmark_config.sample_size,
            max_new_thinking_tokens=lead_config.max_new_thinking_tokens,
            **candidate_config,
        )

        decode_stats_dir = Path(benchmark_config.output_dir) / str(generator) / task_name
        generator.write_decode_stats(
            output_dir=decode_stats_dir,
            task_name=task_name,
            candidate_budget=lead_config.candidate_budget,
        )

    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=selected_tasks,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print(
        f"Thinking-only LEAD evaluation finished: tasks={selected_tasks}, stage1_mode={lead_config.stage1_decode_mode}, budget={lead_config.candidate_budget}",
        style=success_style,
    )


if __name__ == "__main__":
    main()
