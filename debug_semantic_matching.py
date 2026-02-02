import pandas as pd
from generate_rag_benchmark import SidConverter, InvertedIndex, SOURCE_DATA_PATH, VIDEO_PID2SID_PATH, PRODUCT_PID2SID_PATH

def debug_matching():
    # 1. Load Maps
    print("Loading maps...")
    converter = SidConverter()
    converter.load_maps()
    
    # 2. Load Train Data (Small subset for speed)
    print("Loading source data...")
    df_source = pd.read_parquet(SOURCE_DATA_PATH)
    df_train = df_source[df_source['split'] == 0].head(5000) # Index 5000 users
    
    # 3. Build Index
    print("Building Index...")
    index = InvertedIndex("DebugVideo", converter)
    index.add_users(df_train, 'hist_video_pid', 'target_video_pid', 'video')
    
    # 4. Load Test Data
    print("Loading test data...")
    df_test = pd.read_parquet("raw_data/onerec_data/benchmark_data/video/video_test.parquet").head(10)
    
    # 5. Check Matches
    print("\n--- Checking Matches ---")
    for row in df_test.itertuples():
        hist = list(row.hist_pid) if row.hist_pid is not None else []
        if not hist: continue
        
        # Get tokens
        tokens = converter.get_semantic_tokens(hist, 'video')
        print(f"\nUser {row.Index}: History Len={len(hist)}, Tokens={len(tokens)}")
        if len(tokens) > 0:
            print(f"  First 5 Tokens: {list(tokens)[:5]}")
            
        # Retrieve
        results = index.retrieve_similar(hist, -1, 'video', k=3, sample_limit=1000)
        print(f"  Top Match Score: {results[0]['score'] if results else 'None'}")
        
        if results:
            print(f"  Match UID: {results[0]['data']['uid']}")

if __name__ == "__main__":
    debug_matching()
