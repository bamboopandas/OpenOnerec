import pandas as pd

file_path = 'raw_data/onerec_data/benchmark_data/ad/ad_test.parquet'

try:
    df = pd.read_parquet(file_path)
    print(f"File: {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst row raw data:")
    print(df.iloc[0])
except Exception as e:
    print(f"Error: {e}")

