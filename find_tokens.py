from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/OneRec-1.7B", trust_remote_code=True)
vocab = tokenizer.get_vocab()

sid_tokens = [t for t in vocab.keys() if t.startswith("<s_") and t.endswith(">")]
print(f"Number of SID tokens found: {len(sid_tokens)}")
if sid_tokens:
    print(f"Examples: {sid_tokens[:10]}")

other_tokens = [t for t in vocab.keys() if not (t.startswith("<s_") and t.endswith(">"))]
print(f"Number of other tokens: {len(other_tokens)}")
