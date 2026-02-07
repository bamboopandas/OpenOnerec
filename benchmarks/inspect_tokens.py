from transformers import AutoTokenizer
import re
import os

model_path = os.path.abspath("../checkpoints/OneRec-1.7B")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

sid_pattern = re.compile(r"^<s_[a-z]_\d+>$")
sid_tokens = []
sid_ids = []

for token, id in tokenizer.get_vocab().items():
    if sid_pattern.match(token):
        sid_tokens.append(token)
        sid_ids.append(id)

print(f"Found {len(sid_tokens)} SID tokens.")
print(f"Example SIDs: {sid_tokens[:10]}")
print(f"Min ID: {min(sid_ids) if sid_ids else 'N/A'}")
print(f"Max ID: {max(sid_ids) if sid_ids else 'N/A'}")

# Also check for start/end tokens
print(f"sid_begin ID: {tokenizer.convert_tokens_to_ids('<|sid_begin|>')}")
print(f"sid_end ID: {tokenizer.convert_tokens_to_ids('<|sid_end|>')}")
