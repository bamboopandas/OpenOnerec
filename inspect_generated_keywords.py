import pandas as pd
import json
import os

files = [
    'raw_data/onerec_data/benchmark_data_1000_test_raganswersummary_v2/video/video_test.parquet',
    'raw_data/onerec_data/benchmark_data_1000_test_raganswersummary_v2/product/product_test.parquet',
    'raw_data/onerec_data/benchmark_data_1000_test_raganswersummary_v2/ad/ad_test.parquet'
]

for f in files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
        
    print(f"\n=== Inspecting {f} ===")
    try:
        df = pd.read_parquet(f)
        # Sample 5 rows
        for i in range(min(5, len(df))):
            msgs = df.iloc[i]['messages']
            try:
                if isinstance(msgs, str):
                    msgs_list = json.loads(msgs)
                else:
                    msgs_list = msgs # Already list/dict?
                
                # Look for user message with summary
                # The script appends summary to the last user message
                if isinstance(msgs_list, list):
                    last_msg = msgs_list[-1]
                    content = last_msg.get('content', '')
                    print(f"--- Row {i} ---")
                    print(f"Content: {content[-200:]}") # Print last 200 chars to see the summary
            except Exception as e:
                print(f"Error parsing row {i}: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
