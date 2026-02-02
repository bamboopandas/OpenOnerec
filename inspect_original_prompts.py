import pandas as pd
import json
import os

TASKS = ['ad', 'video', 'product']
BASE_DIR = "raw_data/onerec_data/benchmark_data"

for task in TASKS:
    file_path = os.path.join(BASE_DIR, task, f"{task}_test.parquet")
    if not os.path.exists(file_path):
        print(f"Skipping {task} (not found)")
        continue
        
    print(f"\n=== Task: {task} ===")
    df = pd.read_parquet(file_path)
    
    # Inspect first 3 rows
    for i in range(3):
        row = df.iloc[i]
        msgs = json.loads(row['messages'])
        # Find user message
        for m in msgs:
            if m['role'] == 'user':
                content = m['content']
                if isinstance(content, list):
                    text = ""
                    for part in content:
                        if part['type'] == 'text':
                            text += part['text']
                else:
                    text = content
                
                print(f"--- Sample {i} User Message ---")
                print(text[:500] + "..." if len(text) > 500 else text)
                break
