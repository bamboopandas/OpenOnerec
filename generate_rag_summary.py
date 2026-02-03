import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re

# ================= Configuration =================
SOURCE_TRAIN_PATH = "raw_data/onerec_data/onerec_bench_release.parquet"
BENCHMARK_BASE_DIR = "raw_data/onerec_data/benchmark_data_1000"
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_1000_test_raganswersummary"
CAPTION_PATH = "raw_data/onerec_data/pid2caption.parquet"

# RAG Configuration
MIN_SIMILARITY_SCORE = 0.005 
MATCHING_HIST_LEN = 20
TOP_K_USERS = 3

TASK_CONFIG = {
    'video': {'index_type': 'video', 'bench_hist_col': 'hist_pid'},
    'ad': {'index_type': 'ad', 'bench_hist_col': 'hist_ad'},
    'product': {'index_type': 'product', 'bench_hist_col': 'hist_goods'},
}

# 1. Semantic Stop Words (Don't want these as keywords)
STOP_WORDS = {
    '视频', '一个', '关于', '内容', '展示', '主要', '描述', '了', '的', '在', '和', '是', 
    '有', '也', '都', '就', '这', '那', '去', '说', '要', '会', '能', '好', '看', 
    '什么', '怎么', '个', '里', '里头', '上', '下', '前', '后', '中', '之', '与', 
    '及', '等', '或', '可以', '没有', '对于', '这个', '那个', '为', '把', '被', 
    '让', '到', '往', '但', '但是', '因为', '所以', '如果', '我们', '你们', '他们', 
    '自己', '大家', '由于', '提供', '有限', '具体', '未知', '无法', '针对', '进行', 
    '压缩', '总结', '信息', '介绍', '分享', '包含', '其中', '以及', '非常', '十分',
    '通过', '观众', '一种', '一样', '一旦', '一直', '一些', '一切'
}

# 2. Structural Stop Chars/Words (N-grams cannot START or END with these)
# e.g. "是一个" starts with "是" -> Filtered. "的视频" starts with "的" -> Filtered.
STRUCTURAL_STOPS = {
    '的', '了', '是', '在', '和', '与', '或', '对', '将', '以', '这', '那', '有', 
    '个', '为', '被', '从', '到', '让', '给', '去', '来', '把', '又', '也', '很',
    '但', '都', '要', '会', '能', '好', '看', '做', '用', '及', '其', '于'
}

# ================= Helper Classes & Functions =================

def load_captions(path):
    print(f"Loading captions from {path}...")
    if not os.path.exists(path):
        print("Caption file not found!")
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df['pid'], df['dense_caption']))

def get_ngrams(text, n_list=[2, 3, 4, 5, 6]):
    """Extracts n-grams from cleaned text, supporting longer entities."""
    # Keep chinese, numbers, and english letters
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    grams = []
    l = len(text)
    for n in n_list:
        if l >= n:
            grams.extend([text[i:i+n] for i in range(l-n+1)])
    return grams

def is_valid_ngram(gram):
    """
    Checks if an n-gram is valid based on stop word rules.
    """
    if gram in STOP_WORDS:
        return False
    
    # Check boundaries (Start/End)
    # Get first char/word logic roughly by checking first character for Chinese
    if gram[0] in STRUCTURAL_STOPS or gram[-1] in STRUCTURAL_STOPS:
        return False
    
    # Check if it contains ONLY stop words (rough check)
    # e.g. "这是一个" -> "这", "是", "一个" are all stops.
    # But usually boundary check catches "是一个" (starts with 是).
    
    return True

def filter_redundant_terms(counter, threshold_ratio=1.2):
    """
    Filters out terms that are substrings of longer terms with similar frequency.
    Ex: 'Doubao AI' (10) vs 'Doubao AI Assistant' (10) -> Keep longer.
    """
    # Sort by Length DESCENDING (Longest first)
    items = list(counter.items())
    # Sort primarily by length (desc), secondary by freq (desc)
    items.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    
    final_terms = {}
    
    # Check each term against valid longer terms found so far? 
    # No, we need to check against ALL terms.
    
    sorted_by_freq = sorted(items, key=lambda x: x[1], reverse=True)
    # Limit to top candidate pool to avoid O(N^2) on huge lists
    candidates = sorted_by_freq[:60] 
    
    candidate_map = dict(candidates)
    
    processed_terms = set()
    
    # Iterate through candidates. Since we want to KEEP the best, 
    # we can just iterate and see if it is a substring of another "better" candidate.
    
    for term, count in candidates:
        is_redundant = False
        for other, other_count in candidates:
            if term == other: continue
            
            # If term is inside other (e.g. "Peace" in "Peace Elite")
            if term in other:
                # If "Peace" freq is not much higher than "Peace Elite", 
                # it means "Peace" rarely appears alone.
                if count <= other_count * threshold_ratio:
                    is_redundant = True
                    break
        
        if not is_redundant:
            final_terms[term] = count
            
    return Counter(final_terms)

def summarize_commonality(pids, caption_map):
    if not pids: return ""
    captions = []
    for pid in pids:
        try: pid_int = int(pid)
        except: continue
        cap = caption_map.get(pid_int)
        if cap:
            captions.append(cap)
    
    if not captions: return ""

    counter = Counter()
    for cap in captions:
        grams = get_ngrams(cap, [2, 3, 4, 5, 6]) # Longer n-grams
        for g in grams:
            if is_valid_ngram(g):
                counter[g] += 1
                
    if not counter: return ""

    refined_counter = filter_redundant_terms(counter)
    common_terms = [item[0] for item in refined_counter.most_common(3)]
    
    if not common_terms: return ""
        
    summary = f"用户当前的偏好涉及{', '.join(common_terms)}等主题。"
    return summary

class InvertedIndex:
    def __init__(self, name):
        self.name = name
        self.index = defaultdict(list)
        self.user_data = {} 
        self.user_tokens = {}

    def add_users(self, df, hist_col, target_col):
        print(f"[{self.name}] Indexing {len(df)} users...")
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Indexing {self.name}"):
            uid = row.Index
            hist_raw = getattr(row, hist_col)
            try:
                hist_list = [int(x) for x in hist_raw] if hist_raw is not None else []
            except:
                hist_list = []
            if not hist_list: continue
            
            matching_hist = hist_list[-MATCHING_HIST_LEN:]
            tokens = set(matching_hist)
            if not tokens: continue

            tgt_raw = getattr(row, target_col)
            try:
                if isinstance(tgt_raw, (list, np.ndarray)):
                    tgt = [int(x) for x in tgt_raw]
                elif pd.notna(tgt_raw):
                    tgt = [int(tgt_raw)]
                else:
                    tgt = []
            except:
                tgt = []

            self.user_data[uid] = {'target': tgt, 'uid': row.uid}
            self.user_tokens[uid] = tokens
            for t in tokens:
                self.index[t].append(uid)
        print(f"[{self.name}] Done.")

    def retrieve_similar(self, query_hist_list, query_uid, k=3, sample_limit=1000):
        matching_query = query_hist_list[-MATCHING_HIST_LEN:]
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
            results.append(self.user_data[uid])
        return results

def process_messages(messages_json, summary_text):
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            if isinstance(content, str):
                last_msg['content'] = content + "\n" + summary_text
            elif isinstance(content, list):
                text_found = False
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        part['text'] = part['text'] + "\n" + summary_text
                        text_found = True
                        break
                if not text_found:
                    content.append({'type': 'text', 'text': "\n" + summary_text})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR):
        print(f"Benchmark dir not found: {BENCHMARK_BASE_DIR}")
        return
    
    caption_map = load_captions(CAPTION_PATH)
    
    print(f"Loading Source Data {SOURCE_TRAIN_PATH}...")
    try:
        df_source = pd.read_parquet(SOURCE_TRAIN_PATH)
        df_train = df_source[df_source['split'] == 0].copy()
    except Exception as e:
        print(f"Failed to load train data: {e}")
        return

    indexes = {}
    indexes['video'] = InvertedIndex("Video")
    indexes['video'].add_users(df_train, 'hist_video_pid', 'target_video_pid')
    indexes['ad'] = InvertedIndex("Ad")
    indexes['ad'].add_users(df_train, 'hist_ad_pid', 'target_ad_pid')
    indexes['product'] = InvertedIndex("Product")
    indexes['product'].add_users(df_train, 'hist_goods_pid', 'target_goods_pid')

    task_dirs = [d for d in os.listdir(BENCHMARK_BASE_DIR) if os.path.isdir(os.path.join(BENCHMARK_BASE_DIR, d))]
    
    for task_name in task_dirs:
        task_conf = TASK_CONFIG.get(task_name)
        if not task_conf:
            continue

        print(f"\n>>> Processing Task: {task_name}")
        task_in = os.path.join(BENCHMARK_BASE_DIR, task_name)
        task_out = os.path.join(OUTPUT_BASE_DIR, task_name)
        os.makedirs(task_out, exist_ok=True)
        
        bench_hist_col = task_conf['bench_hist_col']
        index = indexes[task_conf['index_type']]
        
        for fname in os.listdir(task_in):
            if not fname.endswith('.parquet'): continue
            
            f_in = os.path.join(task_in, fname)
            f_out = os.path.join(task_out, fname)
            
            try: df_bench = pd.read_parquet(f_in)
            except: continue

            # For RAG, we process the FULL file if not limited by user request. 
            # (If user wants limit, uncomment next line)
            # df_bench = df_bench.head(10)

            new_msgs = []
            records = df_bench.to_dict('records')
            
            for row in tqdm(records, desc=f"  RAG-Summary {fname}"):
                q_uid = -1
                try:
                    meta = json.loads(row.get('metadata', '{}'))
                    if 'uid' in meta: q_uid = meta['uid']
                except: pass

                q_hist = row.get(bench_hist_col)
                try:
                    q_hist_list = [int(x) for x in q_hist] if q_hist is not None else []
                except:
                    q_hist_list = []
                
                similar_users = index.retrieve_similar(q_hist_list, q_uid, k=TOP_K_USERS)
                
                retrieved_pids = []
                for user_data in similar_users:
                    tgt = user_data.get('target')
                    if tgt is None: continue
                    if isinstance(tgt, (list, np.ndarray)):
                        retrieved_pids.extend(tgt)
                    else:
                        retrieved_pids.append(tgt)
                
                summary = summarize_commonality(retrieved_pids, caption_map)
                
                orig_msg = row.get('messages')
                if summary:
                    new_msgs.append(process_messages(orig_msg, summary))
                else:
                    new_msgs.append(orig_msg)
            
            df_res = pd.DataFrame(records)
            df_res['messages'] = new_msgs
            df_res.to_parquet(f_out)
            print(f"  Saved {f_out}")

if __name__ == "__main__":
    main()
