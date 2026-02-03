
import pandas as pd
import os
import json

def inspect_processed_ad_data(directory):
    print(f"--- Inspecting Processed Ad Data in {directory} ---")
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # List all parquet files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    files.sort()

    for filename in files:
        file_path = os.path.join(directory, filename)
        print(f"\n>>> Reading {filename}...")
        try:
            df = pd.read_parquet(file_path)
            # Iterate through all rows
            for index, row in df.iterrows():
                try:
                    msgs = json.loads(row['messages'])
                    if msgs and isinstance(msgs, list):
                        last_msg = msgs[-1]
                        if last_msg.get('role') == 'user':
                            content = last_msg.get('content')
                            text = ""
                            if isinstance(content, str):
                                text = content
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get('type') == 'text':
                                        text += part.get('text', '')
                            
                            # Extract the last line (the summary)
                            lines = text.strip().split('\n')
                            if lines:
                                print(lines[-1])
                except Exception as e:
                    print(f"Error parsing row {index}: {e}")
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    # Target the processed ad directory
    ad_processed_dir = "raw_data/onerec_data/benchmark_data_1000_test/ad"
    inspect_processed_ad_data(ad_processed_dir)
