import pandas as pd

file_path = 'raw_data/onerec_data/onerec_bench_release.parquet'

try:
    df = pd.read_parquet(file_path)
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst row raw data:")
    print(df.iloc[0])
    
    print("\nSample 5 rows:")
    print(df.sample(5))

except Exception as e:
    print(f"Error: {e}")

