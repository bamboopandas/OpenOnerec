import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random

# ================= Configuration =================
SOURCE_DATA_PATH = "raw_data/onerec_data/onerec_bench_release.parquet"
VIDEO_PID2SID_PATH = "raw_data/onerec_data/video_ad_pid2sid.parquet"
PRODUCT_PID2SID_PATH = "raw_data/onerec_data/product_pid2sid.parquet"

BENCHMARK_BASE_DIR = "raw_data/onerec_data/benchmark_data"
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_rag"

# SID Formatting
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

# Task Configuration
# Defines how to retrieve and how to format the example
TASK_CONFIG = {
    'video': {
        'index_type': 'video', # Use Video Index
        'format_type': 'video_rec',
        'bench_hist_col': 'hist_pid'
    },
    'ad': {
        'index_type': 'ad', # Use Ad Index
        'format_type': 'ad_rec',
        'bench_hist_col': 'hist_ad'
    },
    'product': {
        'index_type': 'product', # Use Product Index
        'format_type': 'product_rec',
        'bench_hist_col': 'hist_goods'
    }
}

# Tasks to ENABLE RAG for
ENABLED_TASKS = ['video', 'ad', 'product']

# ================= Helper Classes =================

class SidConverter:
    """Handles PID to SID conversion for different domains."""
    def __init__(self):
        self.maps = {}
        
    def load_maps(self):
        print("Loading PID-to-SID mappings...")
        
        # 1. Video/Ad Map
        if os.path.exists(VIDEO_PID2SID_PATH):
            try:
                df = pd.read_parquet(VIDEO_PID2SID_PATH)
                self.maps['video'] = dict(zip(df['pid'], df['sid']))
                print(f"  Loaded {len(self.maps['video'])} video/ad SIDs.")
            except Exception as e:
                print(f"  Error loading video map: {e}")
                self.maps['video'] = {}
        else:
            print(f"  Video map not found at {VIDEO_PID2SID_PATH}")
            self.maps['video'] = {}

        # 2. Product Map
        if os.path.exists(PRODUCT_PID2SID_PATH):
            try:
                df = pd.read_parquet(PRODUCT_PID2SID_PATH)
                self.maps['product'] = dict(zip(df['pid'], df['sid']))
                print(f"  Loaded {len(self.maps['product'])} product SIDs.")
            except Exception as e:
                print(f"  Error loading product map: {e}")
                self.maps['product'] = {}
        else:
            print(f"  Product map not found at {PRODUCT_PID2SID_PATH}")
            self.maps['product'] = {}

    def to_sid_str(self, pids, domain='video', max_len=20):
        """Converts list of PIDs to formatted SID string."""
        if pids is None: return ""
        try:
            if len(pids) == 0: return ""
        except:
            return ""
        
        mapping = self.maps.get(domain, {})
        if not mapping: return "" # Return empty if no map, strictly follow format

        sids = []
        count = 0
        # Use simple iteration
        try:
            pid_list = list(pids)
        except:
            return ""

        # Take last N items (history is usually chronological)
        if len(pid_list) > max_len:
            pid_list = pid_list[-max_len:]
            
        for pid in pid_list:
            if pid in mapping:
                code = mapping[pid]
                try:
                    sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
                    sids.append(sid)
                except:
                    continue
        
        return "".join(sids)

class InvertedIndex:
    """Simple Inverted Index for Jaccard Similarity Search."""
    def __init__(self, name):
        self.name = name
        self.index = defaultdict(list)
        # Store full row data for rich formatting
        self.user_data = {} 
        self.total_users = 0

    def add_users(self, df, hist_col, target_col, extra_cols=None):
        print(f"[{self.name}] Indexing...")
        
        # Determine extra columns to store
        cols_to_store = [hist_col, target_col]
        if extra_cols:
            cols_to_store.extend(extra_cols)
            
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Indexing {self.name}"):
            uid = row.Index
            hist_raw = getattr(row, hist_col)
            
            try:
                hist_set = set(hist_raw) if hist_raw is not None else set()
            except:
                continue
            if not hist_set: continue

            # Store Data
            data_entry = {
                'hist': list(hist_raw),
                'target': getattr(row, target_col),
            }
            if extra_cols:
                for col in extra_cols:
                    data_entry[col] = getattr(row, col)
            
            self.user_data[uid] = data_entry
            
            # Indexing
            for item in hist_set:
                self.index[item].append(uid)
            
            self.total_users += 1
        print(f"[{self.name}] Done. {len(self.index)} items.")

    def retrieve_similar(self, query_hist_list, k=3, sample_limit=1000):
        if not query_hist_list: return []
        query_set = set(query_hist_list)
        if not query_set: return []

        candidate_counts = defaultdict(int)
        for item in query_set:
            if item in self.index:
                for uid in self.index[item]:
                    candidate_counts[uid] += 1
        
        if not candidate_counts: return []

        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:sample_limit]

        scored_users = []
        for uid, intersection in sorted_candidates:
            train_hist = set(self.user_data[uid]['hist'])
            union = len(query_set) + len(train_hist) - intersection
            score = intersection / union if union > 0 else 0
            scored_users.append((uid, score))

        scored_users.sort(key=lambda x: x[1], reverse=True)
        best_users = scored_users[:k]

        results = []
        for uid, score in best_users:
            results.append({
                'data': self.user_data[uid],
                'score': score
            })
        return results

# ================= Main Logic =================

def format_rag_block(examples, converter, task_type):
    """
    Formats examples using SFT-aligned prompts.
    Matches logic in data/onerec_data/sft/*.py
    """
    if not examples: return ""

    block = "\n\n以下是相似用户的参考示例：\n"
    
    for i, ex in enumerate(examples):
        data = ex['data']
        
        # --- Video Rec Format ---
        if task_type == 'video_rec':
            # Hist: Video PIDs -> SIDs
            hist_sids = converter.to_sid_str(data['hist'], 'video', max_len=20)
            # Target: Video PIDs -> SIDs
            target_sids = converter.to_sid_str(data['target'], 'video', max_len=5)
            
            if not hist_sids or not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
            block += f"用户观看过的视频：{hist_sids}\n"
            block += f"用户实际观看的视频：{target_sids}\n"

        # --- Ad Rec Format ---
        elif task_type == 'ad_rec':
            # Hist 1: Longview Video PIDs -> SIDs
            longview_sids = converter.to_sid_str(data.get('hist_longview_video_list'), 'video', max_len=20)
            # Hist 2: Ad PIDs -> SIDs
            ad_sids = converter.to_sid_str(data['hist'], 'video', max_len=20) # Ad uses video map
            # Target: Ad PIDs -> SIDs
            target_sids = converter.to_sid_str(data['target'], 'video', max_len=5)
            
            if not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
            if longview_sids:
                block += f"用户观看过的视频：{longview_sids}\n"
            if ad_sids:
                block += f"用户点击过的广告视频：{ad_sids}\n"
            block += f"用户实际点击的广告：{target_sids}\n"

        # --- Product Rec Format ---
        elif task_type == 'product_rec':
            # Hist 1: Longview Video PIDs -> SIDs
            longview_sids = converter.to_sid_str(data.get('hist_longview_video_list'), 'video', max_len=20)
            # Hist 2: Goods PIDs -> SIDs
            goods_sids = converter.to_sid_str(data['hist'], 'product', max_len=20)
            # Target: Goods PIDs -> SIDs
            target_sids = converter.to_sid_str(data['target'], 'product', max_len=5)
            
            if not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
            if longview_sids:
                block += f"用户观看过的视频：{longview_sids}\n"
            if goods_sids:
                block += f"用户点击过的商品：{goods_sids}\n"
            block += f"用户实际点击的商品：{target_sids}\n"

    block += "\n[示例结束]\n基于上述示例和当前用户的历史，请进行预测。"
    return block

def inject_rag(messages_json, rag_text):
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                last_msg['content'] = content + rag_text
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        part['text'] = part['text'] + rag_text
                        break
                else:
                    content.append({'type': 'text', 'text': rag_text})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR):
        print(f"Benchmark dir not found: {BENCHMARK_BASE_DIR}")
        return

    # 1. Load Maps
    converter = SidConverter()
    converter.load_maps()

    # 2. Load Source Data
    print(f"Loading Source Data {SOURCE_DATA_PATH}...")
    try:
        df_source = pd.read_parquet(SOURCE_DATA_PATH)
        df_train = df_source[df_source['split'] == 0].copy()
        print(f"Training Data: {len(df_train)} rows")
    except Exception as e:
        print(f"Failed to load source data: {e}")
        return

    # 3. Build Indexes with specific extra columns needed for formatting
    indexes = {}
    
    # Video Index
    indexes['video'] = InvertedIndex("Video")
    indexes['video'].add_users(df_train, 'hist_video_pid', 'target_video_pid')
    
    # Ad Index (Needs 'hist_longview_video_list' for formatting)
    indexes['ad'] = InvertedIndex("Ad")
    indexes['ad'].add_users(df_train, 'hist_ad_pid', 'target_ad_pid', extra_cols=['hist_longview_video_list'])
    
    # Product Index (Needs 'hist_longview_video_list')
    indexes['product'] = InvertedIndex("Product")
    indexes['product'].add_users(df_train, 'hist_goods_pid', 'target_goods_pid', extra_cols=['hist_longview_video_list'])

    # 4. Process Tasks
    task_dirs = [d for d in os.listdir(BENCHMARK_BASE_DIR) if os.path.isdir(os.path.join(BENCHMARK_BASE_DIR, d))]
    
    for task_name in task_dirs:
        print(f"\n>>> Processing Task: {task_name}")
        task_input_dir = os.path.join(BENCHMARK_BASE_DIR, task_name)
        task_output_dir = os.path.join(OUTPUT_BASE_DIR, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(task_input_dir) if f.endswith('.parquet')]
        
        should_rag = task_name in ENABLED_TASKS
        task_config = TASK_CONFIG.get(task_name)
        
        if not should_rag:
            print(f"  [Skipping RAG] Task '{task_name}' not in enabled list. Copying...")
        
        for file_name in files:
            input_file = os.path.join(task_input_dir, file_name)
            output_file = os.path.join(task_output_dir, file_name)
            
            try:
                df_bench = pd.read_parquet(input_file)
            except Exception as e:
                print(f"    Error reading file: {e}")
                continue

            if not should_rag or not task_config:
                df_bench.to_parquet(output_file)
                continue
            
            # Setup for RAG
            bench_hist_col = task_config['bench_hist_col']
            index_key = task_config['index_type']
            format_type = task_config['format_type']
            
            current_index = indexes.get(index_key)
            
            if bench_hist_col not in df_bench.columns:
                print(f"    Missing history column '{bench_hist_col}'. Copying...")
                df_bench.to_parquet(output_file)
                continue

            new_messages = []
            for row in tqdm(df_bench.itertuples(), total=len(df_bench), desc=f"    Injecting RAG ({file_name})"):
                q_hist = getattr(row, bench_hist_col)
                q_hist_list = list(q_hist) if q_hist is not None else []
                
                examples = current_index.retrieve_similar(q_hist_list, k=3)
                
                # Format using specific prompts
                rag_text = format_rag_block(examples, converter, format_type)
                
                orig_msg = getattr(row, 'messages')
                new_msg = inject_rag(orig_msg, rag_text)
                new_messages.append(new_msg)
            
            df_bench_rag = df_bench.copy()
            df_bench_rag['messages'] = new_messages
            df_bench_rag.to_parquet(output_file)
            print(f"    Saved to {output_file}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
