import os
import json
import pandas as pd
from collections import Counter
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ================= Configuration =================
SOURCE_DIR = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000"
TARGET_DIR = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000_test_answersummary"
CAPTION_PATH = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/pid2caption.parquet"

# Expanded Stop Words
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

STRUCTURAL_STOPS = {
    '的', '了', '是', '在', '和', '与', '或', '对', '将', '以', '这', '那', '有', 
    '个', '为', '被', '从', '到', '让', '给', '去', '来', '把', '又', '也', '很',
    '但', '都', '要', '会', '能', '好', '看', '做', '用', '及', '其', '于'
}

# Global variable for multiprocessing
global_caption_map = {}

def load_captions(path):
    print(f"Loading captions from {path}...")
    if not os.path.exists(path):
        print("Caption file not found!")
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df['pid'], df['dense_caption']))

def get_ngrams(text, n_list=[2, 3, 4, 5, 6]):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    grams = []
    l = len(text)
    for n in n_list:
        if l >= n:
            grams.extend([text[i:i+n] for i in range(l-n+1)])
    return grams

def is_valid_ngram(gram):
    if gram in STOP_WORDS:
        return False
    if gram[0] in STRUCTURAL_STOPS or gram[-1] in STRUCTURAL_STOPS:
        return False
    return True

def filter_redundant_terms_optimized(counter, top_k_candidates=60, threshold_ratio=1.2):
    most_common = counter.most_common(top_k_candidates)
    if not most_common: return []
    
    # Candidates: sorted by freq desc
    candidates = [x[0] for x in most_common]
    candidate_counts = {x[0]: x[1] for x in most_common}
    
    final_terms = []
    
    for term in candidates:
        if not is_valid_ngram(term): continue
        
        is_redundant = False
        count = candidate_counts[term]
        
        for other in candidates:
            if term == other: continue
            
            if term in other:
                if count <= candidate_counts[other] * threshold_ratio:
                    is_redundant = True
                    break
        
        if not is_redundant:
            final_terms.append(term)
            if len(final_terms) >= 3:
                break
            
    return final_terms[:3]

def summarize_commonality(pids):
    caption_map = global_caption_map
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
        grams = get_ngrams(cap, [2, 3, 4, 5, 6])
        for g in grams:
            if is_valid_ngram(g):
                counter[g] += 1
                
    if not counter: return ""

    common_terms = filter_redundant_terms_optimized(counter)
    
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

def process_single_file(file_info):
    source_path, target_path = file_info
    try:
        df = pd.read_parquet(source_path)
    except Exception as e:
        print(f"Failed to read {source_path}: {e}")
        return

    new_messages = []
    for row in df.itertuples():
        orig_msgs = row.messages
        metadata_str = getattr(row, 'metadata', '{}')
        summary = ""
        try:
            meta = json.loads(metadata_str)
            # Check answer_pid (video/ad) OR answer_iid (product)
            pids = meta.get('answer_pid') or meta.get('answer_iid')
            if pids and isinstance(pids, list):
                summary = summarize_commonality(pids)
        except:
            pass
        
        if summary:
            new_msg = process_messages(orig_msgs, summary)
            new_messages.append(new_msg)
        else:
            new_messages.append(orig_msgs)
            
    df['messages'] = new_messages
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    df.to_parquet(target_path)
    return f"Processed {os.path.basename(source_path)}"

def main():
    global global_caption_map
    if not os.path.exists(SOURCE_DIR): return

    global_caption_map = load_captions(CAPTION_PATH)
    print(f"Loaded {len(global_caption_map)} captions.")

    tasks = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        rel_path = os.path.relpath(root, SOURCE_DIR)
        target_root = os.path.join(TARGET_DIR, rel_path)
        for file in files:
            if file.endswith('.parquet'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                tasks.append((source_file, target_file))
            elif file.endswith('.json'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                os.system(f"cp {source_file} {target_file}")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting processing with {max_workers} workers for {len(tasks)} files...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
