
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "benchmarks"))

from benchmark.ads_generator import ADSHuggingFaceGenerator

model_path = "checkpoints/OneRec-1.7B"
generator = ADSHuggingFaceGenerator(model_path, ads_top_k=1, gpu_ids=[3])

# Create a dummy prompt with some history
prompt = "User History: <|sid_begin|><s_a_1><|sid_end|><|sid_begin|><s_a_2><|sid_end|>\nAnalyze user preference."
prompts = {"0": prompt}

print("Starting generation...")
results, _ = generator.generate(prompts, max_new_tokens=50, stop=["</think>"])
print("Generation result:", results)

