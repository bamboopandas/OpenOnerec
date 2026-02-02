
import pandas as pd
import os

def inspect_parquet(file_path):
    print(f"--- Inspecting {os.path.basename(file_path)} ---")
    try:
        df = pd.read_parquet(file_path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("\nFirst 1 Row Sample (dict format):")
        print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n")

# Check typical OneRec files
base_dir = "raw_data/onerec_data"
files_to_check = [
    "onerec_bench_release.parquet",
    "pid2caption.parquet" 
]

for f in files_to_check:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        inspect_parquet(path)
    else:
        print(f"File not found: {path}")

# Check for processed RL files if they exist
rl_base = "output/rl_data"
if os.path.exists(rl_base):
    for f in os.listdir(rl_base):
        if f.endswith(".parquet"):
             inspect_parquet(os.path.join(rl_base, f))
             break # Just check one
