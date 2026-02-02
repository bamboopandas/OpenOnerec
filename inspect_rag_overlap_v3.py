import pandas as pd
import json
import re
import random
import os
import sys

# Configuration
DATA_FILE = "raw_data/onerec_data/benchmark_data_id_v4/ad/ad_test.parquet"
MAPPING_FILE = "raw_data/onerec_data/video_ad_pid2sid.parquet"

def load_mapping():
    print("Loading PID-to-SID mapping...")
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Mapping file {MAPPING_FILE} not found.")
        return {}
    df = pd.read_parquet(MAPPING_FILE)
    pid2sid = {}
    sid_fmt = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
    
    for row in df.itertuples():
        try:
            code = row.sid
            formatted_sid = sid_fmt.format(c0=code[0], c1=code[1], c2=code[2])
            pid2sid[row.pid] = formatted_sid
        except:
            continue
    print(f"Loaded {len(pid2sid)} mappings.")
    return pid2sid

def extract_sids_from_text(text):
    """Extracts all SID strings from a text block."""
    matches = re.findall(r'<\|sid_begin\|>.*?<\|sid_end\|>', text)
    return set(matches)

def get_current_user_sids(row, pid2sid):
    """Converts current user's history PIDs to SIDs."""
    sids = set()
    if row.get('hist_ad') is not None:
        for pid in row['hist_ad']: 
            if pid in pid2sid: sids.add(pid2sid[pid])
    if row.get('hist_longview') is not None:
        for pid in row['hist_longview']: 
            if pid in pid2sid: sids.add(pid2sid[pid])
    return sids

def parse_rag_examples(messages):
    """Parses the RAG section from messages to get example SIDs and UIDs."""
    try:
        msgs = json.loads(messages)
    except:
        return []
    
    rag_text = ""
    for m in msgs:
        content = m.get('content')
        if isinstance(content, str):
            if "[Reference Examples]" in content or "以下是相似用户的参考示例" in content:
                rag_text = content
                break
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'text' and ("Reference Examples" in part.get('text', '') or "以下是相似用户的参考示例" in part.get('text', '')):
                    rag_text = part['text']
                    break
    
    if not rag_text:
        return []

    # Truncate at [示例结束]
    if "[示例结束]" in rag_text:
        rag_text = rag_text.split("[示例结束]")[0]

    # Find headers with UIDs: [示例 X (UID: 123)]
    parts = re.split(r'\[示例 \d+(?: \(UID: .*?\))?\]', rag_text)
    headers = re.findall(r'\[示例 \d+(?: \(UID: (.*?)\))?\]', rag_text)
    
    examples = []
    # parts[0] is intro. parts[1] matches headers[0].
    for i, part in enumerate(parts[1:]):
        ex_sids = extract_sids_from_text(part)
        if ex_sids:
            uid_str = "Unknown"
            if i < len(headers):
                uid_str = headers[i]
            examples.append({'sids': ex_sids, 'uid': uid_str})
            
    return examples

def main():
    pid2sid = load_mapping()
    
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return
        
    df = pd.read_parquet(DATA_FILE)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows.")

    while True:
        idx = random.randint(0, total_rows - 1)
        row = df.iloc[idx]
        
        print("\n" + "="*80)
        print(f"Sample Index: {idx}")
        
        # Current User Metadata Check
        try:
            meta = json.loads(row['metadata'])
            print(f"Current User UID: {meta.get('uid', 'N/A')}")
        except:
            print("Current User UID: Parse Error")

        # 1. Get Current User SIDs
        curr_sids = get_current_user_sids(row, pid2sid)
        print(f"Current User Unique SIDs: {len(curr_sids)}")
        
        # 2. Get RAG Examples
        rag_examples = parse_rag_examples(row['messages'])
        
        if not rag_examples:
            print("No RAG examples found in this sample.")
        else:
            print(f"Found {len(rag_examples)} RAG examples.")
            
            for i, ex in enumerate(rag_examples):
                ex_sids = ex['sids']
                ex_uid = ex['uid']
                overlap = curr_sids.intersection(ex_sids)
                count = len(overlap)
                
                print(f"\n--- Example {i+1} (UID: {ex_uid}) ---")
                print(f"  Example SID Count: {len(ex_sids)}")
                print(f"  Overlap Count: {count}")
                if count > 0:
                    print(f"  Overlapping SIDs (first 3): {list(overlap)[:3]}")

        print("="*80)
        cmd = input("\nPress Enter to sample again, or 'q' to quit: ").strip().lower()
        if cmd == 'q':
            break

if __name__ == "__main__":
    main()