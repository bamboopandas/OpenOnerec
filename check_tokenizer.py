from transformers import AutoTokenizer
import os

model_path = "checkpoints/OneRec-1.7B"
if not os.path.exists(model_path):
    # Try to find where the model is. The file structure showed checkpoints/OneRec-1.7B
    # but maybe I need to use absolute path or check if it's empty.
    # The list_dir showed it exists.
    pass

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded from {model_path}")
    
    # Check special tokens
    print(f"PAD: {tokenizer.pad_token}, EOS: {tokenizer.eos_token}")
    
    # Check specific SID tokens
    test_tokens = ["<|sid_begin|>", "<|sid_end|>", "<s_a_0>", "<s_a_100>", "<s_b_0>", "<s_c_0>"]
    for t in test_tokens:
        ids = tokenizer.convert_tokens_to_ids(t)
        # If unknown, it might return unk_token_id or be split
        decoded = tokenizer.decode([ids])
        print(f"Token: '{t}' -> ID: {ids} -> Decoded: '{decoded}'")
        
    # Check if they are single tokens
    tokens = tokenizer.tokenize("<s_a_123>")
    print(f"Tokenize '<s_a_123>': {tokens}")

    # Find range of s_a tokens
    sa_ids = []
    for i in range(8192):
        t = f"<s_a_{i}>"
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid == tokenizer.unk_token_id:
            break
        sa_ids.append(tid)
    
    print(f"Found {len(sa_ids)} <s_a_*> tokens.")
    if len(sa_ids) > 0:
        print(f"Range: {min(sa_ids)} - {max(sa_ids)}")

except Exception as e:
    print(f"Error: {e}")
