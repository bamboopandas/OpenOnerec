import os
import json
import pandas as pd
from tqdm import tqdm

SOURCE_DIR = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000"
TARGET_DIR = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000_test"

def process_messages(messages_json, answer_summary):
    try:
        messages = json.loads(messages_json)
        if not messages:
            return messages_json
        
        # Find the last user message
        # Usually it's the last one, or second to last if there's a system prompt or empty assistant prompt?
        # Typically in these datasets, the last message is the user query.
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                last_msg['content'] = content + "\n" + answer_summary
            elif isinstance(content, list):
                # Handle multi-modal or list content
                text_part_found = False
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        part['text'] = part['text'] + "\n" + answer_summary
                        text_part_found = True
                        break
                if not text_part_found:
                    content.append({'type': 'text', 'text': "\n" + answer_summary})
        
        return json.dumps(messages, ensure_ascii=False)
    except Exception as e:
        print(f"Error processing messages: {e}")
        return messages_json

def process_file(file_path, target_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    new_messages = []
    
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
        orig_msgs = row.messages
        metadata_str = getattr(row, 'metadata', '{}')
        
        answer_summary = ""
        try:
            meta = json.loads(metadata_str)
            if 'answer' in meta:
                answer = meta['answer']
                answer_summary = f"正确答案是：{answer}"
        except:
            pass
        
        if answer_summary:
            new_msg = process_messages(orig_msgs, answer_summary)
            new_messages.append(new_msg)
        else:
            new_messages.append(orig_msgs)
            
    df['messages'] = new_messages
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    df.to_parquet(target_path)
    print(f"Saved to {target_path}")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    # Walk through the directory
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Determine relative path to mirror structure
        rel_path = os.path.relpath(root, SOURCE_DIR)
        target_root = os.path.join(TARGET_DIR, rel_path)
        
        for file in files:
            if file.endswith('.parquet'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                process_file(source_file, target_file)
            elif file.endswith('.json'):
                # Copy JSON files (like mappings) just in case
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                os.system(f"cp {source_file} {target_file}")
                print(f"Copied {file}")

if __name__ == "__main__":
    main()
