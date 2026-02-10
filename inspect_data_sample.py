import pandas as pd
import json

path = "raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet"
df = pd.read_parquet(path)
row = df.iloc[0]
print("Columns:", df.columns)
print("Messages type:", type(row['messages']))
print("Messages content:")
try:
    msgs = json.loads(row['messages'])
    for m in msgs:
        print(f"Role: {m['role']}")
        print(f"Content: {m['content'][:200]}...") # Truncate
except:
    print(row['messages'])
