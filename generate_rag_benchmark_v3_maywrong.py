import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re

# ================= Configuration =================
SOURCE_DATA_PATH = "raw_data/onerec_data/onerec_bench_release.parquet"
VIDEO_PID2SID_PATH = "raw_data/onerec_data/video_ad_pid2sid.parquet"
PRODUCT_PID2SID_PATH = "raw_data/onerec_data/product_pid2sid.parquet"

BENCHMARK_BASE_DIR = "raw_data/onerec_data/benchmark_data"
# Changed output directory to benchmark_data_id
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_id"

# SID Formatting
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

# Configuration
# Sparse PID matching needs very low threshold to find anything
MIN_SIMILARITY_SCORE = 0.005 
EXAMPLE_HIST_LEN = 10 
MATCHING_HIST_LEN = 20

TASK_CONFIG = {
    'video': {
        'index_type': 'video', 
        'format_type': 'video_rec',
        'bench_hist_col': 'hist_pid',
        'labels': {
            'hist': "用户观看过的视频：",
            'target': "用户实际观看的视频："
        }
    },
    'ad': {
        'index_type': 'ad', 
        'format_type': 'ad_rec',
        'bench_hist_col': 'hist_ad',
        'labels': {
            'hist_longview': "用户观看过的视频：",
            'hist_ad': "用户点击过的广告视频：",
            'target': "用户实际点击的广告："
        }
    },
    'product': {
        'index_type': 'product', 
        'format_type': 'product_rec',
        'bench_hist_col': 'hist_goods',
        'labels': {
            'hist_longview': "用户观看过的视频：",
            'hist_goods': "用户点击过的商品：",
            'target': "用户实际点击的商品："
        }
    }
}

ENABLED_TASKS = ['video', 'ad', 'product']

# ================= Helper Classes =================

class SidConverter:
    """Handles PID to SID conversion for display only."""
    def __init__(self):
        self.maps = {}
        
    def load_maps(self):
        print("Loading PID-to-SID mappings...")
        for domain, path in [('video', VIDEO_PID2SID_PATH), ('product', PRODUCT_PID2SID_PATH)]:
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    self.maps[domain] = dict(zip(df['pid'], df['sid']))
                    print(f"  Loaded {len(self.maps[domain])} {domain} SIDs.")
                except Exception as e:
                    print(f"  Error loading {domain} map: {e}")
                    self.maps[domain] = {}
            else:
                self.maps[domain] = {}

    def to_sid_str(self, pids, domain='video', max_len=20, deduplicate=True):
        """Converts PIDs to formatted SID string with optional deduplication."""
        if pids is None: return ""
        try:
            if len(pids) == 0: return ""
        except:
            return ""
        
        mapping = self.maps.get(domain, {})
        if not mapping: return ""

        try:
            pid_list = list(pids)
        except:
            return ""

        # 1. Truncate (keep recent)
        if len(pid_list) > max_len:
            pid_list = pid_list[-max_len:]
        
        # 2. Deduplicate consecutive PIDs
        final_pids = []
        if deduplicate:
            for pid in pid_list:
                if not final_pids or final_pids[-1] != pid:
                    final_pids.append(pid)
        else:
            final_pids = pid_list

        # 3. Format
        sids = []
        for pid in final_pids:
            if pid in mapping:
                code = mapping[pid]
                try:
                    sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
                    sids.append(sid)
                except:
                    continue
        
        return "".join(sids)

class InvertedIndex:
    """Raw PID Inverted Index."""
    def __init__(self, name):
        self.name = name
        self.index = defaultdict(list) # pid -> list of uids
        self.user_data = {} # uid -> full row data
        self.user_tokens = {} # uid -> set of pids

    def add_users(self, df, hist_col, target_col, extra_cols=None):
        print(f"[{self.name}] Indexing {len(df)} users (PID mode)...")
        
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Indexing {self.name}"):
            uid = row.Index
            
            hist_raw = getattr(row, hist_col)
            try:
                hist_list = list(hist_raw) if hist_raw is not None else []
            except:
                hist_list = []
                
            if not hist_list: continue
            
            # Use raw PIDs as tokens (recent history)
            matching_hist = hist_list[-MATCHING_HIST_LEN:]
            tokens = set(matching_hist)
            
            if not tokens: continue

            # Store Data
            data_entry = {
                'hist': hist_list,
                'target': getattr(row, target_col),
                'uid': row.uid
            }
            if extra_cols:
                for col in extra_cols:
                    data_entry[col] = getattr(row, col)

            self.user_data[uid] = data_entry
            self.user_tokens[uid] = tokens
            
            # Indexing
            for t in tokens:
                self.index[t].append(uid)
                
        print(f"[{self.name}] Done. {len(self.index)} unique items.")

    def retrieve_similar(self, query_hist_list, query_uid, k=3, sample_limit=1000):
        # Use raw PIDs
        matching_query = list(query_hist_list)[-MATCHING_HIST_LEN:]
        query_tokens = set(matching_query)
        
        if not query_tokens: return []

        candidate_counts = defaultdict(int)
        filtered_count = 0
        for t in query_tokens:
            if t in self.index:
                for uid in self.index[t]:
                    # Debug: Check types if needed, but they are ints
                    cand_uid = self.user_data[uid]['uid']
                    if cand_uid == query_uid:
                        filtered_count += 1
                        continue
                    candidate_counts[uid] += 1
        
        # Only print if we actually filtered something for the first few rows
        if filtered_count > 0 and query_uid > -1:
             # print(f"DEBUG: Filtered self-match for UID {query_uid} ({filtered_count} times)")
             pass
        
        if not candidate_counts: return []

        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:sample_limit]

        scored_users = []
        for uid, intersection in sorted_candidates:
            train_tokens = self.user_tokens[uid]
            union = len(query_tokens) + len(train_tokens) - intersection
            score = intersection / union if union > 0 else 0
            
            # Filter clones (Self-match with different UID)
            if score >= MIN_SIMILARITY_SCORE and score < 0.95:
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

def format_rag_block(examples, converter, task_config):
    """Formats examples (same as before, uses SIDs for display)."""
    if not examples: return ""

    labels = task_config['labels']
    task_type = task_config['format_type']
    
    block = "以下是相似用户的参考示例：\n"
    
    for i, ex in enumerate(examples):
        data = ex['data']
        
        raw_target = data['target']
        single_target = []
        try:
            t_list = list(raw_target)
            if t_list:
                single_target = [t_list[-1]]
        except:
            pass

        if not single_target: continue

        # --- Video Rec ---
        if task_type == 'video_rec':
            hist_sids = converter.to_sid_str(data['hist'], 'video', max_len=EXAMPLE_HIST_LEN, deduplicate=True)
            target_sids = converter.to_sid_str(single_target, 'video', max_len=1)
            
            block += f"\n[示例 {i+1} (UID: {data.get('uid', 'N/A')})]\n"
            block += f"{labels['hist']}{hist_sids}\n"
            block += f"{labels['target']}{target_sids}\n"

        # --- Ad Rec ---
        elif task_type == 'ad_rec':
            longview_sids = converter.to_sid_str(data.get('hist_longview_video_list'), 'video', max_len=EXAMPLE_HIST_LEN, deduplicate=True)
            ad_sids = converter.to_sid_str(data['hist'], 'video', max_len=EXAMPLE_HIST_LEN, deduplicate=True)
            target_sids = converter.to_sid_str(single_target, 'video', max_len=1)
            
            if not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
            if longview_sids:
                block += f"{labels['hist_longview']}{longview_sids}\n"
            if ad_sids:
                block += f"{labels['hist_ad']}{ad_sids}\n"
            block += f"{labels['target']}{target_sids}\n"

        # --- Product Rec ---
        elif task_type == 'product_rec':
            longview_sids = converter.to_sid_str(data.get('hist_longview_video_list'), 'video', max_len=EXAMPLE_HIST_LEN, deduplicate=True)
            goods_sids = converter.to_sid_str(data['hist'], 'product', max_len=EXAMPLE_HIST_LEN, deduplicate=True)
            target_sids = converter.to_sid_str(single_target, 'product', max_len=1)
            
            if not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
            if longview_sids:
                block += f"{labels['hist_longview']}{longview_sids}\n"
            if goods_sids:
                block += f"{labels['hist_goods']}{goods_sids}\n"
            block += f"{labels['target']}{target_sids}\n"

    block += "\n[示例结束]\n\n"
    return block

def inject_rag_prepend(messages_json, rag_text):
    """Prepends RAG text."""
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                last_msg['content'] = rag_text + content
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        part['text'] = rag_text + part['text']
                        break
                else:
                    content.insert(0, {'type': 'text', 'text': rag_text})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR):
        print("Benchmark dir not found.")
        return

    # 1. Load Maps (Only for display formatting now)
    converter = SidConverter()
    converter.load_maps()

    # 2. Load Source Data
    print(f"Loading Source Data {SOURCE_DATA_PATH}...")
    try:
        df_source = pd.read_parquet(SOURCE_DATA_PATH)
        df_train = df_source[df_source['split'] == 0].copy()
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # 3. Build Raw PID Indexes
    indexes = {}
    
    # Video
    indexes['video'] = InvertedIndex("Video")
    indexes['video'].add_users(df_train, 'hist_video_pid', 'target_video_pid')
    
    # Ad
    indexes['ad'] = InvertedIndex("Ad")
    indexes['ad'].add_users(df_train, 'hist_ad_pid', 'target_ad_pid', extra_cols=['hist_longview_video_list'])
    
    # Product
    indexes['product'] = InvertedIndex("Product")
    indexes['product'].add_users(df_train, 'hist_goods_pid', 'target_goods_pid', extra_cols=['hist_longview_video_list'])

    # 4. Process
    task_dirs = [d for d in os.listdir(BENCHMARK_BASE_DIR) if os.path.isdir(os.path.join(BENCHMARK_BASE_DIR, d))]
    
    for task_name in task_dirs:
        print(f"\n>>> Task: {task_name}")
        task_in = os.path.join(BENCHMARK_BASE_DIR, task_name)
        task_out = os.path.join(OUTPUT_BASE_DIR, task_name)
        os.makedirs(task_out, exist_ok=True)
        
        should_rag = task_name in ENABLED_TASKS
        task_conf = TASK_CONFIG.get(task_name)
        
        for fname in os.listdir(task_in):
            if not fname.endswith('.parquet'): continue
            
            f_in = os.path.join(task_in, fname)
            f_out = os.path.join(task_out, fname)
            
            try:
                df_bench = pd.read_parquet(f_in)
            except:
                continue

            if not should_rag or not task_conf:
                df_bench.to_parquet(f_out)
                continue

            # RAG Logic
            bench_hist_col = task_conf['bench_hist_col']
            index_key = task_conf['index_type']
            
            new_msgs = []
            for row in tqdm(df_bench.itertuples(), total=len(df_bench), desc=f"  RAG {fname}"):
                # Query UID
                q_uid = -1
                try:
                    meta = json.loads(row.metadata)
                    if 'uid' in meta: q_uid = meta['uid']
                except Exception as e:
                    print(f"Metadata parse error: {e}")
                    pass
                
                # Debug print for first few rows
                if row.Index < 5:
                    print(f"DEBUG: Row {row.Index}, q_uid={q_uid}, Type={type(q_uid)}")

                # Query History (Raw PIDs)
                q_hist = getattr(row, bench_hist_col, None)
                q_hist_list = list(q_hist) if q_hist is not None else []
                
                # Retrieve (PID matching)
                examples = indexes[index_key].retrieve_similar(
                    q_hist_list, 
                    query_uid=q_uid, 
                    k=3
                )
                
                # Format (SIDs) & Inject
                rag_text = format_rag_block(examples, converter, task_conf)
                orig_msg = getattr(row, 'messages')
                
                if rag_text:
                    new_msg = inject_rag_prepend(orig_msg, rag_text)
                else:
                    new_msg = orig_msg
                    
                new_msgs.append(new_msg)
            
            df_rag = df_bench.copy()
            df_rag['messages'] = new_msgs
            df_rag.to_parquet(f_out)
            print(f"  Saved {f_out}")

    print("\nDone.")

if __name__ == "__main__":
    main()
