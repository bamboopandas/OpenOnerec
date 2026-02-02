import argparse
import os
import sys
import json
import numpy as np

# Check for pandas dependency
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Error: 'pandas' library is required but not found.")
    print("Please run this script in an environment with pandas installed.")

# --- Field Descriptions ---
# A dictionary to explain what each field means in the context of OpenOneRec
FIELD_DESCRIPTIONS = {
    # Raw Data Fields
    "uid": "User Unique Identifier (用户ID)",
    "split": "Data Split ID (e.g., 0 for train)",
    "hist_video_pid": "History: Sequence of Watched Video IDs (用户历史观看序列)",
    "hist_video_like": "History: Sequence of Like Actions (0/1) (用户历史点赞行为)",
    "hist_video_longview": "History: Sequence of Long-View Actions (0/1) (用户历史长播行为)",
    "target_video_pid": "Target: Next Video IDs to Predict (预测目标视频ID)",
    "inter_user_profile_with_pid": "User Profile with Raw PIDs (包含原始ID的用户画像)",
    "inter_user_profile_with_sid": "User Profile with Tokenized IDs (包含Token ID的用户画像 - 模型输入用)",
    "inter_keyword_to_items": "User Search Keywords & Associated Items (搜索词及其关联Item)",
    "pid": "Product/Video ID (物品ID)",
    "dense_caption": "Detailed Text Description of the Item (物品详细文本描述)",
    "sid": "Semantic ID / Tokenized ID (语义ID/Token化ID)",
    
    # Processed/Training Data Fields
    "source": "Dataset Source Name (数据来源)",
    "uuid": "Unique Sample ID (样本唯一ID)",
    "messages": "LLM Chat Format Data (LLM对话格式数据 - 核心训练输入)",
    "metadata": "Extra Meta Info (Json) - e.g. original UID, task type (元数据)",
    "images": "Image placeholders (Not used in text-only training)",
    "videos": "Video placeholders (Not used in text-only training)",
    "input": "Input Prompt (模型输入Prompt)",
    "output": "Expected Output (期望输出)",
}

def format_value(key, value):
    """
    Intelligently formats a value for display:
    - Parses and pretty-prints JSON strings.
    - Truncates long lists/arrays.
    - Truncates long strings.
    """
    if value is None:
        return "None"

    # 1. Handle JSON strings (common in this dataset)
    if isinstance(value, str) and (value.strip().startswith('{') or value.strip().startswith('[')):
        try:
            parsed = json.loads(value)
            # If it's a list inside JSON, check length
            if isinstance(parsed, list) and len(parsed) > 3:
                return json.dumps(parsed[:3], indent=2, ensure_ascii=False) + f"\n... (+ {len(parsed)-3} more items)"
            # If it's a dict, use pretty print
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass # Not valid JSON, treat as string

    # 2. Handle Numpy Arrays / Lists
    if isinstance(value, (list, np.ndarray)):
        # Convert numpy array to list for easier handling
        if isinstance(value, np.ndarray):
            value = value.tolist()
        
        total_len = len(value)
        if total_len > 10:
            preview = str(value[:10]).rstrip(']')
            return f"{preview}, ... ] (Total Length: {total_len})"
        return str(value)

    # 3. Handle Long Strings (that are not JSON)
    if isinstance(value, str):
        if len(value) > 200:
            return f"{value[:200]} ... [Truncated, Total Chars: {len(value)}]"
    
    return str(value)

def print_section(title):
    print(f"\n{'-'*100}")
    print(f"|  {title.upper()}")
    print(f"{'-'*100}")

def is_empty_or_nan(val):
    """Safely check if a value is None, empty, or NaN."""
    if val is None:
        return True
    
    # Handle Numpy Arrays
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return True
        # Check if all elements are NaN (if numeric)
        if np.issubdtype(val.dtype, np.number):
            return np.all(np.isnan(val))
        return False # Non-empty array is considered "not empty"

    # Handle Lists/Strings
    if isinstance(val, (str, list)):
        return len(val) == 0

    # Handle Floats/Numbers (NaN check)
    if isinstance(val, float) and np.isnan(val):
        return True
        
    # Handle Pandas/Numpy Scalars
    try:
        if pd.isna(val):
            return True
    except:
        pass # If pd.isna fails or returns array, ignore here

    return False

def inspect_file(file_path, description, file_type='parquet', show_columns=False):
    print(f"\n📂 FILE: {os.path.basename(file_path)}")
    print(f"   Path: {file_path}")
    print(f"   Info: {description}")
    
    if not os.path.exists(file_path):
        print("   ⚠️  File not found. Skipping.")
        return

    try:
        if file_type == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
        else:
            print(f"   [!] Unsupported file type: {file_type}")
            return

        print(f"   Shape: {df.shape} (Rows, Columns)")
        
        if df.empty:
            print("   [Empty DataFrame]")
            return

        # Sample Extraction (First Row)
        sample = df.iloc[0].to_dict()
        
        print(f"\n   🔍 SAMPLE DATA (Row 0):")
        print(f"   {'-'*90}")

        # Iterate through columns
        for col, val in sample.items():
            if is_empty_or_nan(val):
                continue
            
            # Get Description
            desc = FIELD_DESCRIPTIONS.get(col, "")
            
            # Format Value
            formatted_val = format_value(col, val)
            
            # Print cleanly
            print(f"   👉 [{col}]")
            if desc:
                print(f"      Explain: {desc}")
            
            # Indent the value for better structure
            val_lines = formatted_val.split('\n')
            for line in val_lines:
                print(f"      {line}")
            print("")

    except Exception as e:
        print(f"   ❌ Error reading file: {e}")

def find_first_parquet(directory):
    """Helper to find the first parquet file in a directory for sampling."""
    if not os.path.exists(directory):
        return None
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                return os.path.join(root, file)
    return None

def main():
    parser = argparse.ArgumentParser(description="Inspect OpenOneRec data pipeline with enhanced readability.")
    parser.add_argument('stage', choices=['raw', 'sft', 'rl', 'all'], default='all', nargs='?',
                        help="The data stage to inspect.")
    args = parser.parse_args()

    base_dir = os.getcwd()
    
    # --- Phase 1: Raw Data ---
    if args.stage in ['raw', 'all']:
        print_section("Phase 1: Raw Business Data (Before LLM Processing)")
        
        inspect_file(os.path.join(base_dir, "raw_data/onerec_data/onerec_bench_release.parquet"),
                     "User Behavior Logs - The raw input for recommendation.")
        
        inspect_file(os.path.join(base_dir, "raw_data/onerec_data/pid2caption.parquet"),
                     "Item Knowledge - Maps Item IDs to text descriptions.")

    # --- Phase 2: SFT / Split Data ---
    if args.stage in ['sft', 'all']:
        print_section("Phase 2: SFT Prepared Data (Tokenized & Split)")
        
        sft_sample = find_first_parquet(os.path.join(base_dir, "output/split_data_sft"))
        if sft_sample:
            inspect_file(sft_sample, "SFT Data Chunk - Ready for Instruction Tuning.")
        else:
            print("   (No SFT split data found)")

    # --- Phase 3: RL Data ---
    if args.stage in ['rl', 'all']:
        print_section("Phase 3: RL Training Data (Chat Format)")
        
        inspect_file(os.path.join(base_dir, "output/rl_data/train.parquet"),
                     "RL Train Set - 'messages' field contains the core prompt/response.")

if __name__ == "__main__":
    if not HAS_PANDAS:
        sys.exit(1)
    main()