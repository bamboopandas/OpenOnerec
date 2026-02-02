import pandas as pd
import json
import random
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_list(data, limit=8):
    """Formats a list for display, limiting the number of items shown."""
    if data is None: return "None"
    # Convert numpy arrays/pandas series to list if necessary
    try:
        val_list = list(data)
    except:
        return str(data)
        
    if len(val_list) <= limit: 
        return str(val_list)
    return str(val_list[:limit])[:-1] + f", ... (+{len(val_list)-limit} more)]"

def print_raw_row(row):
    print("\n" + "="*80)
    print(f" User ID: {row.get('uid')} | Split: {'Train' if row.get('split')==0 else 'Test'}")
    print("="*80)

    # 1. Video History
    print("\n[Short Video History]")
    print(f"  Video PIDs:  {format_list(row.get('hist_video_pid'))}")
    print(f"  Longview:    {format_list(row.get('hist_video_longview'))}")
    print(f"  Likes:       {format_list(row.get('hist_video_like'))}")
    print(f"  Follows:     {format_list(row.get('hist_video_follow'))}")
    
    # 2. Targets
    print("\n[Target Interactions (Labels)]")
    print(f"  Target PIDs: {format_list(row.get('target_video_pid'))}")
    print(f"  Longview:    {format_list(row.get('target_video_longview'))}")

    # 3. Ads & Goods
    print("\n[Multi-Domain History]")
    print(f"  Ad PIDs:     {format_list(row.get('hist_ad_pid'))}")
    print(f"  Goods PIDs:  {format_list(row.get('hist_goods_pid'))}")

    # 4. JSON Fields (Interactive / Profile)
    for field in ['inter_user_profile_with_pid', 'inter_keyword_to_items']:
        val = row.get(field)
        if val and isinstance(val, str):
            try:
                parsed = json.loads(val)
                print(f"\n[{field.replace('_', ' ').title()}]")
                # Print a snippet of the JSON
                dumped = json.dumps(parsed, indent=2, ensure_ascii=False)
                if len(dumped) > 800:
                    print(dumped[:800] + "\n... (truncated)")
                else:
                    print(dumped)
            except:
                pass

    # 5. COT / Captions
    if row.get('reco_cot') and str(row.get('reco_cot')) != 'None':
        print("\n[Reasoning (CoT)]")
        print(row.get('reco_cot'))

    print("\n" + "="*80)

def main():
    file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/split_data_pretrain/part-27761-of-33947.parquet'
    # file_path = 'raw_data/onerec_data/onerec_bench_release.parquet'
    
    if not os.path.exists(file_path):
        # Try relative path if run from scripts folder
        alt_path = '../raw_data/onerec_data/onerec_bench_release.parquet'
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"Error: File not found at {file_path}")
            return

    print(f"Loading {file_path} (2.1GB, please wait)...")
    try:
        df = pd.read_parquet(file_path)
        total_rows = len(df)
        print(f"Success! Total rows: {total_rows}")
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    while True:
        idx = random.randint(0, total_rows - 1)
        row = df.iloc[idx]
 
        clear_screen()
        print(f"Current Index: {idx} / {total_rows}")
        print_raw_row(row)
        
        user_input = input("\n[Enter] Next Sample, [q] Quit: ").strip().lower()
        if user_input == 'q':
            break

if __name__ == "__main__":
    main()
