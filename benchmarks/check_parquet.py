import pandas as pd
try:
    df = pd.read_parquet("../raw_data/onerec_data/benchmark_data_1000/ad/ad_test.parquet")
    msg = df.iloc[0]['messages']
    print(type(msg))
    print(msg)
except Exception as e:
    print(e)