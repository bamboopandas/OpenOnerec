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
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_id_v4_cleandata"

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
SID_PATTERN = r'<\|sid_begin\|>.*?<\|sid_end\|>'

MIN_SIMILARITY_SCORE = 0.005 
# Truncation limits based on SFT scripts
TRUNCATION_LIMITS = {
    'video': 100, # Conservative
    'ad': 100, 
    'product': 100,
    'default': 50
}
MATCHING_HIST_LEN = 20

TASK_CONFIG = {
    'video': {'index_type': 'video', 'format_type': 'video_rec', 'bench_hist_col': 'hist_pid'},
    'ad': {'index_type': 'ad', 'format_type': 'ad_rec', 'bench_hist_col': 'hist_ad'},
    'product': {'index_type': 'product', 'format_type': 'product_rec', 'bench_hist_col': 'hist_goods'}
}

ENABLED_TASKS = ['video', 'ad', 'product']

# ================= Helper Classes =================

class SidConverter:
    def __init__(self):
        self.maps = {}
        self.parsed_maps = {} 
        
    def load_maps(self):
        print("Loading PID-to-SID mappings...")
        for domain, path in [('video', VIDEO_PID2SID_PATH), ('product', PRODUCT_PID2SID_PATH)]:
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    self.maps[domain] = dict(zip(df['pid'], df['sid']))
                    self.parsed_maps[domain] = self.maps[domain]
                    print(f"  Loaded {len(self.maps[domain])} {domain} SIDs.")
                except Exception as e:
                    print(f"  Error loading {domain} map: {e}")
                    self.maps[domain] = {}
            else:
                self.maps[domain] = {}

    def get_semantic_tokens(self, pids, domain='video'):
        if pids is None: return set()
        try:
            if len(pids) == 0: return set()
        except:
            return set()
        
        mapping = self.parsed_maps.get(domain, {})
        tokens = set()
        try:
            pid_iterable = list(pids)
        except:
            return set()

        for pid in pid_iterable:
            if pid in mapping:
                code = mapping[pid]
                try:
                    token = f"{code[0]}_{code[1]}" 
                    tokens.add(token)
                except:
                    continue
        return tokens

    def to_sid_str(self, pids, domain='video', max_len=100, deduplicate=True):
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

        if len(pid_list) > max_len:
            pid_list = pid_list[-max_len:]
        
        final_pids = []
        if deduplicate:
            for pid in pid_list:
                if not final_pids or final_pids[-1] != pid:
                    final_pids.append(pid)
        else:
            final_pids = pid_list

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
    def __init__(self, name):
        self.name = name
        self.index = defaultdict(list)
        self.user_data = {} 
        self.user_tokens = {}

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
            
            matching_hist = hist_list[-MATCHING_HIST_LEN:]
            tokens = set(matching_hist)
            if not tokens: continue

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
            for t in tokens:
                self.index[t].append(uid)
        print(f"[{self.name}] Done.")

    def retrieve_similar(self, query_hist_list, query_uid, k=1, sample_limit=1000):
        matching_query = list(query_hist_list)[-MATCHING_HIST_LEN:]
        query_tokens = set(matching_query)
        if not query_tokens: return []

        candidate_counts = defaultdict(int)
        for t in query_tokens:
            if t in self.index:
                for uid in self.index[t]:
                    if self.user_data[uid]['uid'] == query_uid:
                        continue
                    candidate_counts[uid] += 1
        if not candidate_counts: return []

        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:sample_limit]
        scored_users = []
        for uid, intersection in sorted_candidates:
            union = len(query_tokens) + len(self.user_tokens[uid]) - intersection
            score = intersection / union if union > 0 else 0
            recall = intersection / len(query_tokens) if len(query_tokens) > 0 else 0
            
            if score >= MIN_SIMILARITY_SCORE and recall < 0.8:
                scored_users.append((uid, score))

        scored_users.sort(key=lambda x: x[1], reverse=True)
        best_users = scored_users[:k]
        
        results = []
        for uid, score in best_users:
            results.append({'data': self.user_data[uid], 'score': score})
        return results

# ================= Template Logic =================

def fill_template(query_text, example_data, converter, max_len):
    """
    Parses query_text to find SID slots and labels, then fills with example data.
    """
    # Split by SIDs
    # capturing group for SIDs: (.*?) (<\|sid_begin\|>.*?<\|sid_end\|>)
    # Iterate to find all segments
    
    # We want to identify segments like: "Label: [SIDs]"
    # Regex to find all SID blocks
    sid_block_regex = re.compile(r'(<\|sid_begin\|>.*?<\|sid_end\|>)')
    
    parts = sid_block_regex.split(query_text)
    # parts: [prefix0, sids0, prefix1, sids1, ..., suffix]
    
    example_text = ""
    
    # Process pairs (prefix, sids)
    for i in range(0, len(parts) - 1, 2):
        prefix = parts[i]
        # sids = parts[i+1] # We replace this
        
        # Determine data source based on prefix keywords
        source_data = []
        domain = 'video'
        
        if "长时间" in prefix: # Longview
            source_data = example_data.get('hist_longview_video_list', [])
        elif "商品" in prefix or "购物" in prefix: # Product
            source_data = example_data.get('hist', []) # 'hist' stores main domain
            domain = 'product'
        elif "广告" in prefix: # Ad
            source_data = example_data.get('hist', []) # 'hist' stores main domain (for ad task)
            domain = 'video' # Ad uses video/ad mixed space
        else: # Default (Video)
            source_data = example_data.get('hist', [])
            domain = 'video'
            
        ex_sids_str = converter.to_sid_str(source_data, domain, max_len=max_len, deduplicate=True)
        
        example_text += prefix + ex_sids_str
        
    # Add the final suffix (Instruction)
    suffix = parts[-1]
    example_text += suffix
    
    # Add Target (with implicit "Answer" formatting if possible? No, simply append target)
    # But where does target go?
    # The suffix is usually "Please recommend:".
    # So we append target after suffix.
    
    target_data = example_data.get('target', [])
    target_list = list(target_data) if target_data is not None else []
    if target_list:
        single_target = [target_list[-1]] # Last item
        # Determine domain for target
        # If task is product, target is product. Else video.
        # We can infer from 'hist' domain?
        # Or check if 'hist_goods' in data?
        # Hacky but 'hist_goods_pid' was mapped to 'hist'.
        # If we are in 'product' task, target is product.
        # But this function doesn't know task type explicitly.
        # However, SidConverter uses 'product' or 'video' map.
        # Let's guess: if we used 'product' domain earlier, target is product.
        # Default video.
        target_domain = 'video'
        if 'product' in example_data.get('hist', []): # No, this is list
             pass
        # Better: use the index/task config. But we are inside template filler.
        # Let's pass task_type to this function.
        pass

    return example_text, suffix

def format_rag_entry(example, query_text, converter, task_type, max_len):
    """Reconstructs example text mirroring query structure."""
    data = example['data']
    
    sid_block_regex = re.compile(r'(<\|sid_begin\|>.*?<\|sid_end\|>)')
    parts = sid_block_regex.split(query_text)
    
    example_str = ""
    
    # Iterate prefixes
    for i in range(0, len(parts) - 1, 2):
        prefix = parts[i]
        
        # Decide Data Source
        source = []
        domain = 'video'
        
        if "长时间" in prefix:
            source = data.get('hist_longview_video_list', [])
        elif "商品" in prefix or "购物" in prefix:
            source = data.get('hist', [])
            domain = 'product'
        elif "广告" in prefix:
            source = data.get('hist', [])
            domain = 'video'
        else:
            source = data.get('hist', [])
            domain = 'video' # Default
            
        sids = converter.to_sid_str(source, domain, max_len, deduplicate=True)
        example_str += prefix + sids
        
    suffix = parts[-1]
    example_str += suffix
    
    # Append Target
    # Target domain based on task_type
    tgt_domain = 'product' if task_type == 'product_rec' else 'video'
    target_data = data.get('target', [])
    if target_data is not None and len(target_data) > 0:
        tgt_sids = converter.to_sid_str([target_data[-1]], tgt_domain, max_len=1)
        example_str += tgt_sids + "\n" # Add newline after target?
    
    return example_str

def inject_rag_prepend(messages_json, rag_text):
    """Prepends RAG text to the last user message."""
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                last_msg['content'] = rag_text + "\n" + content
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        part['text'] = rag_text + "\n" + part['text']
                        break
                else:
                    content.insert(0, {'type': 'text', 'text': rag_text})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR): return
    
    converter = SidConverter()
    converter.load_maps()
    
    try:
        df_source = pd.read_parquet(SOURCE_DATA_PATH)
        df_train = df_source[df_source['split'] == 0].copy()
    except: return

    indexes = {}
    indexes['video'] = InvertedIndex("Video")
    indexes['video'].add_users(df_train, 'hist_video_pid', 'target_video_pid')
    indexes['ad'] = InvertedIndex("Ad")
    indexes['ad'].add_users(df_train, 'hist_ad_pid', 'target_ad_pid', extra_cols=['hist_longview_video_list'])
    indexes['product'] = InvertedIndex("Product")
    indexes['product'].add_users(df_train, 'hist_goods_pid', 'target_goods_pid', extra_cols=['hist_longview_video_list'])

    for task_name in ['video', 'ad', 'product']:
        print(f"\n>>> Task: {task_name}")
        task_in = os.path.join(BENCHMARK_BASE_DIR, task_name)
        task_out = os.path.join(OUTPUT_BASE_DIR, task_name)
        os.makedirs(task_out, exist_ok=True)
        
        task_conf = TASK_CONFIG.get(task_name)
        trunc_len = TRUNCATION_LIMITS.get(task_name, 50)
        
        for fname in os.listdir(task_in):
            if not fname.endswith('.parquet'): continue
            
            f_in = os.path.join(task_in, fname)
            f_out = os.path.join(task_out, fname)
            
            try: df_bench = pd.read_parquet(f_in)
            except: continue

            new_msgs = []
            for row in tqdm(df_bench.itertuples(), total=len(df_bench), desc=f"  RAG {fname}"):
                q_uid = -1
                try:
                    meta = json.loads(row.metadata)
                    if 'uid' in meta: q_uid = meta['uid']
                except: pass

                q_hist = getattr(row, task_conf['bench_hist_col'], [])
                if q_hist is None: q_hist = []
                
                examples = indexes[task_conf['index_type']].retrieve_similar(list(q_hist), q_uid, k=1)
                
                # Get Original Message Text
                orig_json = row.messages
                orig_text = ""
                try:
                    msgs = json.loads(orig_json)
                    for m in msgs:
                        if m['role'] == 'user':
                            content = m['content']
                            if isinstance(content, str): orig_text = content
                            elif isinstance(content, list):
                                for p in content:
                                    if p['type'] == 'text': orig_text += p['text']
                            break
                except: pass
                
                rag_text = ""
                if examples and orig_text:
                    rag_text = format_rag_entry(examples[0], orig_text, converter, task_conf['format_type'], trunc_len)
                
                if rag_text:
                    new_msgs.append(inject_rag_prepend(orig_json, rag_text))
                else:
                    new_msgs.append(orig_json)
            
            df_rag = df_bench.copy()
            df_rag['messages'] = new_msgs
            df_rag.to_parquet(f_out)
            print(f"  Saved {f_out}")

if __name__ == "__main__":
    main()
