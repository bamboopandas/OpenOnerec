import pandas as pd
import json

file_path = "raw_data/onerec_data/benchmark_data_rag/video/video_test.parquet"

try:
    df = pd.read_parquet(file_path)
    print(f"Loaded {file_path}")
    
    # Check random rows
    for i in range(20):
        row = df.sample(1).iloc[0]
        messages_json = row['messages']
        messages = json.loads(messages_json)
        
        last_msg = messages[-1]
        content = ""
        if isinstance(last_msg['content'], str):
            content = last_msg['content']
        elif isinstance(last_msg['content'], list):
            for part in last_msg['content']:
                if part['type'] == 'text':
                    content = part['text']
                    break
        
        if "以下是相似用户的参考示例" in content:
            print(f"\n[Found RAG in sample {i}]")
            print(content[:500])
            break
        else:
            print(f".", end="", flush=True)
    else:
        print("\nNo RAG text found in 20 samples.")
    
except Exception as e:
    print(f"Error: {e}")
