import pandas as pd
import json
import re

def count_sids(text):
    """Counts the number of SID tokens in a string."""
    # SID pattern: <s_a_...><s_b_...><s_c_...>
    # Assuming one SID corresponds to one <|sid_begin|>...<|sid_end|> block is WRONG.
    # The format is <|sid_begin|><s_a_1><s_b_2><s_c_3><s_a_4>...<|sid_end|>
    # Wait, the format in SidConverter is: <|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|> per item?
    # Let's check SidConverter.to_sid_str in generate_rag_benchmark.py
    # "sid = SID_FORMAT.format(...) ... sids.append(sid) ... return "".join(sids)"
    # SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
    # So each item is wrapped in <|sid_begin|>...<|sid_end|>.
    
    # Therefore, counting occurrences of <|sid_begin|> gives the number of items.
    return text.count("<|sid_begin|>")

def check_file(file_path, limit=512):
    print(f"Checking {file_path} for truncation limit: {limit}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    max_found = 0
    over_limit_count = 0
    total_rag_messages = 0

    for idx, row in df.iterrows():
        try:
            messages = json.loads(row['messages'])
        except:
            continue
            
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content']
                text_content = ""
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'text':
                            text_content += part.get('text', '')

                # We are looking for the RAG part which is prepended.
                # It usually starts with something like "用户浏览过的视频内容：" or similar prefixes found in the code.
                # However, the user simply said truncation didn't work.
                # Let's scan the whole text for sequences of SIDs.
                
                # Split by newline or just count total SIDs in the message? 
                # The RAG part is injected. The original query is also there.
                # If we count ALL SIDs, it might be RAG + Original Query.
                # The original query is usually "User history: ...".
                
                # Let's try to isolate the RAG part.
                # In inject_rag_prepend, rag_text is added + "\n" + content.
                # So the first part of the message is the RAG examples.
                
                # Let's count SIDs in the entire message first.
                count = count_sids(text_content)
                if count > max_found:
                    max_found = count
                
                # To be more precise, let's look for the longest sequence of SIDs.
                # Actually, the user likely saw a very long list of SIDs.
                
                if count > limit * 1.5: # Arbitrary buffer for RAG + Query
                     # print(f"Warning: High SID count {count} at row {idx}")
                     pass

                # Let's look specifically at the RAG example part.
                # The prompt structure in generate code:
                # prefix + sids
                # "User watched: <sid>...<sid>"
                
                # If we split by newline, we might separate RAG examples from current query.
                lines = text_content.split('\n')
                for line in lines:
                    line_sids = count_sids(line)
                    if line_sids > limit:
                        over_limit_count += 1
                        # print(f"  Row {idx}: Found line with {line_sids} SIDs (Limit {limit})")
                        # print(f"  Line snippet: {line[:100]}...")
                        
    print(f"Total rows: {len(df)}")
    print(f"Max SIDs found in a single message: {max_found}")
    print(f"Lines exceeding limit {limit}: {over_limit_count}")

if __name__ == "__main__":
    check_file('raw_data/onerec_data/benchmark_data_id_v4_cleandata_small/video/video_test.parquet', 512)
