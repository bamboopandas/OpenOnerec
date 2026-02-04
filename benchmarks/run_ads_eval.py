import argparse
import os
import sys

# Add the current directory to sys.path to allow importing from 'benchmark'
sys.path.append(os.getcwd())

from benchmark import Benchmark
from benchmark.ads_generator import ADSHuggingFaceGenerator
from benchmark.console import console, success_style, warning_style

def main():
    parser = argparse.ArgumentParser(description="Run ADS Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--task_types", type=str, nargs="+", default=["video"], help="Tasks to run")
    parser.add_argument("--splits", type=str, nargs="+", default=["test"], help="Splits to run")
    parser.add_argument("--ads_top_k", type=int, default=1, help="Number of items to select via ADS")
    parser.add_argument("--sample_size", type=str, default="10", help="Number of samples to run (or 'full')")
    
    args = parser.parse_args()
    
    # Handle sample_size
    sample_size = args.sample_size
    if sample_size != "full":
        sample_size = int(sample_size)
    else:
        sample_size = None 
    
    console.print(f"Initializing Benchmark with data: {args.data_dir}", style=warning_style)
    benchmark = Benchmark(
        model_path=args.model_path,
        task_types=args.task_types,
        splits=args.splits,
        data_dir=args.data_dir,
        enable_thinking=True 
    )
    
    console.print(f"Initializing ADS Generator (Top-K={args.ads_top_k})...", style=warning_style)
    generator = ADSHuggingFaceGenerator(
        model_path=args.model_path,
        ads_top_k=args.ads_top_k
    )
    
    console.print(f"Starting Generation...", style=warning_style)
    benchmark.run(
        generator=generator,
        output_dir=args.output_dir,
        overwrite=True,
        enable_thinking=True,
        sample_size=sample_size,
        # Generation params
        max_new_tokens=512,
        temperature=0.0,
        stop=["</think>"]
    )
    
    console.print("Calculating Metrics...", style=warning_style)
    eval_results_path = os.path.join(args.output_dir, "eval_results.json")
    Benchmark.evaluate_dev(
        generation_results_dir=args.output_dir,
        output_path=eval_results_path,
        data_dir=args.data_dir,
        overwrite=True,
        task_types=args.task_types
    )
    
    console.print("Evaluation Done!", style=success_style)

if __name__ == "__main__":
    main()
