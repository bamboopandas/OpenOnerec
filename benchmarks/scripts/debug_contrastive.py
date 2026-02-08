import torch
import sys
import os

# Add path to find benchmark modules
sys.path.append(os.getcwd())

from benchmark.contrastive_generator import ContrastiveGenerator

PROMPT = """<|im_start|>system
你是一个个性化广告推荐系统，能够基于用户的视频观看历史和广告点击记录，预测用户未来可能点击的广告。<|im_end|>
<|im_start|>user
用户的历史行为如下：
用户浏览过的视频内容：<|sid_begin|><s_a_6063><s_b_4308><s_c_4548><|sid_end|>
用户点击过的广告视频：<|sid_begin|><s_a_4162><s_b_2932><s_c_3396><|sid_end|>
基于以上记录，推荐用户 可能感兴趣并点击的广告视频。/think<|im_end|>
<|im_start|>assistant
"""

def main():
    model_path = "../checkpoints/OneRec-1.7B"
    print(f"Initializing ContrastiveGenerator with model: {model_path}")
    
    # Init generator
    generator = ContrastiveGenerator(
        model_name_or_path=model_path,
        alpha=0.5,
        device="cuda",
        dtype="bfloat16",
        worker_batch_size=1
    )
    
    print("Running generate...")
    prompts = {"debug_1": PROMPT}
    
    # Run generation
    results, _ = generator.generate(prompts, max_new_tokens=20)
    
    print("\n=== FINAL RESULT ===")
    print(results["debug_1"][0])
    print("====================")

if __name__ == "__main__":
    main()

