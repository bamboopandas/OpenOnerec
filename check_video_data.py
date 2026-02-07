import pandas as pd
import os

file_path = "raw_data/onerec_data/benchmark_data_1000/video/video_test.parquet"
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print(f"Columns: {list(df.columns)}")
    print(df.head(1).to_string())
else:
    print("File not found")
