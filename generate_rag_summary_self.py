import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re
import math
import random

# ================= Configuration =================
BENCHMARK_BASE_DIR = "raw_data/onerec_data/benchmark_data_1000"
OUTPUT_BASE_DIR = "raw_data/onerec_data/benchmark_data_1000_test_self_latestitemsummary_v1"
CAPTION_PATH = "raw_data/onerec_data/pid2caption.parquet"

# Config
MATCHING_HIST_LEN = 20 # Use last 20 items for summary

TASK_CONFIG = {
    'video': {'bench_hist_col': 'hist_pid'},
    'ad': {'bench_hist_col': 'hist_ad'},
    'product': {'bench_hist_col': 'hist_goods'}, # Prioritize interested (goods) over viewed (longview)
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
    '通过', '观众', '一种', '一样', '一旦', '一直', '一些', '一切',
    # Added common artifacts and generic terms
    '视频中', '画面', '兴趣', '生活', '相关', '部分', '点击', '链接', '详情', 
    '查看', '更多', '商品', '产品', '推荐', '物品', '东西', '事物', '可能', 
    '涉及', '主题', '感觉', '觉得', '认为', '使用', '利用', '喜欢', '喜爱',
    # Bad fragments
    '览器', '费书', '空浏', '笔小新'
}

# 2. Structural Stop Chars/Words (N-grams cannot START or END with these)
STRUCTURAL_STOPS = {
    '的', '了', '是', '在', '和', '与', '或', '对', '将', '以', '这', '那', '有', 
    '个', '为', '被', '从', '到', '让', '给', '去', '来', '把', '又', '也', '很',
    '但', '都', '要', '会', '能', '好', '看', '做', '用', '及', '其', '于',
    # Added positionals and classifiers
    '中', '上', '下', '里', '内', '外', '旁', '边', '前', '后', 
    '位', '款', '种', '项', '类', '群', '些', '点', '次', '回', '名'
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
    """Extracts n-grams from cleaned text."""
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
    if gram[0] in STRUCTURAL_STOPS or gram[-1] in STRUCTURAL_STOPS:
        return False
    
    # Heuristic: If > 50% of the string corresponds to stop-chars/words, drop it.
    stop_char_count = sum(1 for c in gram if c in STRUCTURAL_STOPS or c in STOP_WORDS)
    if stop_char_count / len(gram) > 0.5:
        return False

    return True

class IDFCalculator:
    def __init__(self, caption_map, sample_size=50000):
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.compute_idf(caption_map, sample_size)

    def compute_idf(self, caption_map, sample_size):
        print(f"Computing global IDF from sample of {sample_size} captions...")
        all_pids = list(caption_map.keys())
        if len(all_pids) > sample_size:
            sampled_pids = random.sample(all_pids, sample_size)
        else:
            sampled_pids = all_pids
        
        self.total_docs = len(sampled_pids)
        
        for pid in tqdm(sampled_pids, desc="IDF Build"):
            text = caption_map[pid]
            grams = set(get_ngrams(text, [2, 3, 4, 5, 6])) # Use set to count DF (once per doc)
            for g in grams:
                if is_valid_ngram(g):
                    self.doc_freq[g] += 1
        
        print(f"IDF computed for {len(self.doc_freq)} terms.")

    def get_score(self, term):
        # IDF = log(N / (df + 1))
        df = self.doc_freq.get(term, 0)
        # Smooth IDF: +1 to avoid division by zero (though df+1 handles it)
        # Using base 10 log
        return math.log10((self.total_docs + 1) / (df + 1))

def filter_redundant_terms(scored_terms, top_k=3):
    """
    Selects top terms that are not too similar to each other.
    scored_terms: list of (term, score), sorted by score desc.
    """
    # 1. Sort by Length DESC
    candidates = sorted(scored_terms, key=lambda x: len(x[0]), reverse=True)
    
    kept_terms = [] # List of (term, score)
    
    for term, score in candidates:
        is_redundant = False
        for existing_term, existing_score in kept_terms:
            # Check substring
            if term in existing_term: 
                # term is a substring of existing (longer) term.
                # Only keep if score is significantly higher (rare case for TF-IDF if IDF is boosting short term?)
                # But generally we prefer the longer term.
                is_redundant = True
                break
            
            # Check overlap (Jaccard-ish for characters)
            set_a = set(term)
            set_b = set(existing_term)
            overlap = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            if union > 0 and overlap / union > 0.6: 
                is_redundant = True
                break
        
        if not is_redundant:
            kept_terms.append((term, score))
            
    # 3. Sort by Score DESC to pick top K
    kept_terms.sort(key=lambda x: x[1], reverse=True)
    
    return [t[0] for t in kept_terms[:top_k]]

def summarize_commonality(pids, caption_map, idf_calc):
    if not pids: return ""
    captions = []
    for pid in pids:
        try: pid_int = int(pid)
        except: continue
        cap = caption_map.get(pid_int)
        if cap:
            captions.append(cap)
    
    if not captions: return ""

    # TF Counting
    tf_counter = Counter()
    for cap in captions:
        grams = get_ngrams(cap, [2, 3, 4, 5, 6]) 
        for g in grams:
            if is_valid_ngram(g):
                tf_counter[g] += 1
                
    if not tf_counter: return ""

    # TF-IDF Scoring
    scored_terms = []
    for term, tf in tf_counter.items():
        idf = idf_calc.get_score(term)
        score = tf * idf
        scored_terms.append((term, score))
    
    # Sort by Score DESC (to filter candidates pool if needed, though we passed all to filter func before)
    # Actually, we should probably limit candidates BEFORE filtering to avoid O(N^2)
    scored_terms.sort(key=lambda x: x[1], reverse=True)
    
    # Take candidates for diversity filtering
    candidates = scored_terms[:60] 
    
    common_terms = filter_redundant_terms(candidates, top_k=3)
    
    if not common_terms: return ""
        
    summary = f"用户当前的偏好涉及{', '.join(common_terms)}等主题。"
    return summary

def process_messages(messages_json, summary_text):
    try:
        messages = json.loads(messages_json)
        if not messages: return messages_json
        
        last_msg = messages[-1]
        if last_msg['role'] == 'user':
            content = last_msg['content']
            marker = "<|sid_end|>"
            
            if isinstance(content, str):
                idx = content.rfind(marker)
                if idx != -1:
                    insert_pos = idx + len(marker)
                    # Check if there is a newline after marker
                    if content[insert_pos:].startswith("\n"):
                        # Insert summary after the newline following <|sid_end|>
                        last_msg['content'] = content[:insert_pos+1] + summary_text + "\n" + content[insert_pos+1:]
                    else:
                        last_msg['content'] = content[:insert_pos] + "\n" + summary_text + "\n" + content[insert_pos:]
                else:
                    last_msg['content'] = summary_text + "\n" + content
            elif isinstance(content, list):
                processed = False
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text = part['text']
                        idx = text.rfind(marker)
                        if idx != -1:
                            insert_pos = idx + len(marker)
                            if text[insert_pos:].startswith("\n"):
                                part['text'] = text[:insert_pos+1] + summary_text + "\n" + text[insert_pos+1:]
                            else:
                                part['text'] = text[:insert_pos] + "\n" + summary_text + "\n" + text[insert_pos:]
                            processed = True
                            break
                if not processed:
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            part['text'] = summary_text + "\n" + part['text']
                            processed = True
                            break
                if not processed:
                    content.insert(0, {'type': 'text', 'text': summary_text + "\n"})
        
        return json.dumps(messages, ensure_ascii=False)
    except:
        return messages_json

def main():
    if not os.path.exists(BENCHMARK_BASE_DIR):
        print(f"Benchmark dir not found: {BENCHMARK_BASE_DIR}")
        return
    
    caption_map = load_captions(CAPTION_PATH)
    
    # Initialize IDF Calculator
    idf_calc = IDFCalculator(caption_map, sample_size=50000)
    
    # No need to load df_train or build index for self-summary

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
        
        for fname in os.listdir(task_in):
            if not fname.endswith('.parquet'): continue
            
            f_in = os.path.join(task_in, fname)
            f_out = os.path.join(task_out, fname)
            
            try: df_bench = pd.read_parquet(f_in)
            except: continue

            new_msgs = []
            records = df_bench.to_dict('records')
            
            for row in tqdm(records, desc=f"  Self-Summary {fname}"):
                q_hist = row.get(bench_hist_col)
                try:
                    q_hist_list = [int(x) for x in q_hist] if q_hist is not None else []
                except:
                    q_hist_list = []
                
                # Use current user's most recent interactions
                recent_items = q_hist_list[-MATCHING_HIST_LEN:]
                
                summary = summarize_commonality(recent_items, caption_map, idf_calc)
                
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
