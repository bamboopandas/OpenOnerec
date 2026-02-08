import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Prompt from Sample 7
PROMPT = """<|im_start|>system
你是一个个性化广告推荐系统，能够基于用户的视频观看历史和广告点击记录，预测用户未来可能点击的广告。<|im_end|>
<|im_start|>user
用户的历史行为如下：
用户浏览过的视频内容：<|sid_begin|><s_a_6063><s_b_4308><s_c_4548><|sid_end|><|sid_begin|><s_a_2585><s_b_4975><s_c_6507><|sid_end|>
... (truncated for brevity, just simulating the end) ...
<|sid_begin|><s_a_4162><s_b_2932><s_c_3396><|sid_end|>
基于以上记录，推荐用户 可能感兴趣并点击的广告视频。/think<|im_end|>
<|im_start|>assistant
"""

# Shortened prompt for testing (the model might need context, but let's try to keep the structure)
# I will use the full prompt from the json file to be sure, but truncated middle.
FULL_PROMPT = """<|im_start|>system
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
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    
    inputs = tokenizer(FULL_PROMPT, return_tensors="pt").to("cuda")
    
    print("Generating...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=True,
        temperature=0.7
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\n=== GENERATED OUTPUT ===")
    print(generated_text)
    print("========================")

if __name__ == "__main__":
    main()
