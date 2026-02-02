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
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_rag"

# SID Formatting
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

# Configuration
MIN_SIMILARITY_SCORE = 0.005
EXAMPLE_HIST_LEN = 10  # Truncate example history for display
MATCHING_HIST_LEN = 20 # Use recent history for similarity matching

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
    """Handles PID to SID conversion and parsing."""
    def __init__(self):
        self.maps = {}
        # Parsed cache: pid -> (c0, c1, c2)
        self.parsed_maps = {} 
        
    def load_maps(self):
        print("Loading PID-to-SID mappings...")
        for domain, path in [('video', VIDEO_PID2SID_PATH), ('product', PRODUCT_PID2SID_PATH)]:
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    # map: pid -> array([c0, c1, c2])
                    self.maps[domain] = dict(zip(df['pid'], df['sid']))
                    self.parsed_maps[domain] = self.maps[domain] # Direct access to array
                    print(f"  Loaded {len(self.maps[domain])} {domain} SIDs.")
                except Exception as e:
                    print(f"  Error loading {domain} map: {e}")
                    self.maps[domain] = {}
            else:
                self.maps[domain] = {}

    def get_semantic_tokens(self, pids, domain='video'):
        """
        Converts PIDs to 'Topic Tokens' (Level 1+2 of SID).
        Returns a set of strings like 'a_X_b_Y'.
        """
        if pids is None: return set()
        try:
            if len(pids) == 0: return set()
        except:
            return set()
        
        mapping = self.parsed_maps.get(domain, {})
        tokens = set()
        
        # Handle numpy arrays/lists safely
        try:
            pid_iterable = list(pids)
        except:
            return set()

        for pid in pid_iterable:
            if pid in mapping:
                code = mapping[pid]
                try:
                    # Use first 2 levels: s_a_{c0}><s_b_{c1}
                    token = f"{code[0]}_{code[1]}" 
                    tokens.add(token)
                except:
                    continue
        return tokens

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
    """Semantic Inverted Index."""
    def __init__(self, name, converter):
        self.name = name
        self.converter = converter
        self.index = defaultdict(list) # topic_token -> list of uids
        self.user_data = {} # uid -> full row data
        self.user_tokens = {} # uid -> set of topic tokens (for fast jaccard)

    def add_users(self, df, hist_col, target_col, domain, extra_cols=None):
        print(f"[{self.name}] Indexing {len(df)} users...")
        
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Indexing {self.name}"):
            uid = row.Index # Using index as UID key
            
            hist_raw = getattr(row, hist_col)
            # Safe convert to list
            try:
                hist_list = list(hist_raw) if hist_raw is not None else []
            except:
                hist_list = []
                
            if not hist_list: continue
            
            # 1. Get Semantic Tokens (Topic Interest)
            # Use matching length limit
            matching_hist = hist_list[-MATCHING_HIST_LEN:]
            tokens = self.converter.get_semantic_tokens(matching_hist, domain)
            
            if not tokens: continue

            # 2. Store Data
            data_entry = {
                'hist': hist_list, # Store full for later display truncation
                'target': getattr(row, target_col),
                'uid': row.uid # Store actual UID for filtering
            }
            if extra_cols:
                for col in extra_cols:
                    data_entry[col] = getattr(row, col)

            self.user_data[uid] = data_entry
            self.user_tokens[uid] = tokens
            
            # 3. Add to Index
            for t in tokens:
                self.index[t].append(uid)
                
        print(f"[{self.name}] Done. {len(self.index)} semantic topics.")

    def retrieve_similar(self, query_hist_list, query_uid, domain, k=3, sample_limit=1000):
        # 1. Get Query Tokens
        # Use recent history for matching
        matching_query = list(query_hist_list)[-MATCHING_HIST_LEN:]
        query_tokens = self.converter.get_semantic_tokens(matching_query, domain)
        
        if not query_tokens: return []

        # 2. Candidate Generation
        candidate_counts = defaultdict(int)
        for t in query_tokens:
            if t in self.index:
                for uid in self.index[t]:
                    # Exclude self (Data Leakage Prevention)
                    # Check against stored UID (row.uid)
                    if self.user_data[uid]['uid'] == query_uid:
                        continue
                    candidate_counts[uid] += 1
        
        if not candidate_counts: return []

        # 3. Top Candidates
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:sample_limit]

        # 4. Jaccard Scoring
        scored_users = []
        for uid, intersection in sorted_candidates:
            train_tokens = self.user_tokens[uid]
            union = len(query_tokens) + len(train_tokens) - intersection
            score = intersection / union if union > 0 else 0
            
            if score >= MIN_SIMILARITY_SCORE:
                scored_users.append((uid, score))

        # 5. Sort & Return
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
    """Formats examples with aligned prompts and restrictions."""
    if not examples: return ""

    labels = task_config['labels']
    task_type = task_config['format_type']
    
    # Header
    block = "以下是相似用户的参考示例：\n"
    
    for i, ex in enumerate(examples):
        data = ex['data']
        
        # Enforce Single Target (Last one)
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
            
            if not hist_sids or not target_sids: continue
            
            block += f"\n[示例 {i+1}]\n"
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
    """Prepends RAG text to the last user message."""
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                # Prepend
                last_msg['content'] = rag_text + content
            elif isinstance(content, list):
                # Find text part
                for part in content:
                    if part.get('type') == 'text':
                        part['text'] = rag_text + part['text']
                        break
                else:
                    # Insert at beginning if no text part found (unlikely)
                    content.insert(0, {'type': 'text', 'text': rag_text})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR):
        print("Benchmark dir not found.")
        return

    # 1. Load Maps
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

    # 3. Build Semantic Indexes
    indexes = {}
    
    # Video
    indexes['video'] = InvertedIndex("Video", converter)
    indexes['video'].add_users(df_train, 'hist_video_pid', 'target_video_pid', 'video')
    
    # Ad
    indexes['ad'] = InvertedIndex("Ad", converter)
    indexes['ad'].add_users(df_train, 'hist_ad_pid', 'target_ad_pid', 'video', extra_cols=['hist_longview_video_list'])
    
    # Product
    indexes['product'] = InvertedIndex("Product", converter)
    indexes['product'].add_users(df_train, 'hist_goods_pid', 'target_goods_pid', 'product', extra_cols=['hist_longview_video_list'])

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
            
            # Extract query UID from metadata if possible, else 0
            # benchmark data usually has metadata json string
            
            new_msgs = []
            for row in tqdm(df_bench.itertuples(), total=len(df_bench), desc=f"  RAG {fname}"):
                # Get Query UID for filtering
                q_uid = -1
                try:
                    meta = json.loads(row.metadata)
                    if 'uid' in meta: q_uid = meta['uid']
                except:
                    pass

                # Get History
                q_hist = getattr(row, bench_hist_col, None)
                q_hist_list = list(q_hist) if q_hist is not None else []
                
                # Retrieve (Semantic)
                # Determine domain for query conversion
                q_domain = 'product' if index_key == 'product' else 'video'
                
                examples = indexes[index_key].retrieve_similar(
                    q_hist_list, 
                    query_uid=q_uid, 
                    domain=q_domain,
                    k=3
                )
                
                # Format & Inject
                rag_text = format_rag_block(examples, converter, task_conf)
                orig_msg = getattr(row, 'messages')
                
                # Prepend!
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