import pandas as pd
import json

FILE = "raw_data/onerec_data/benchmark_data/ad/ad_test.parquet" # Check ORIGINAL file

try:
    df = pd.read_parquet(FILE)
    print(f"Loaded {FILE}")
    for i in range(5):
        meta = df.iloc[i]['metadata']
        print(f"Row {i} Metadata: {meta}")
        try:
            parsed = json.loads(meta)
            print(f"  Parsed UID: {parsed.get('uid')}")
        except:
            print("  Parse Failed")
except Exception as e:
    print(e)
