from transformers import AutoTokenizer
import json

model_path = "checkpoints/OneRec-1.7B"

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Len tokenizer: {len(tokenizer)}")

special_tokens = tokenizer.all_special_tokens
print(f"Special tokens: {special_tokens}")

# Check for specific tokens
tokens_to_check = ["<|sid_begin|>", "<|sid_end|>", "<|item_start|>", "<|item_end|>", "<think>", "</think>"]
for t in tokens_to_check:
    if t in tokenizer.get_vocab():
        print(f"Found '{t}': ID {tokenizer.convert_tokens_to_ids(t)}")
    else:
        print(f"'{t}' not found in vocab.")

# Inspect added tokens
try:
    with open(f"{model_path}/added_tokens.json") as f:
        added = json.load(f)
        print(f"Added tokens count: {len(added)}")
        # Print first few added tokens
        first_few = list(added.items())[:10]
        print(f"First few added tokens: {first_few}")
        
        # Check if they look like item IDs
        # Usually item tokens might be like <item_0>, <item_1> etc. or just numbers if using the KMeans approach?
        # The tokenizer/README.md said it uses "codes".
        
except FileNotFoundError:
    print("added_tokens.json not found.")

# Try encoding a sample with history
text = "User history: <|sid_begin|> item_123 item_456 <|sid_end|>. Recommend me something."
encoded = tokenizer.encode(text)
print(f"Encoded '{text}': {encoded}")
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
