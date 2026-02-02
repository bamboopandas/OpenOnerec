
import pandas as pd
import json
import random
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_message_data(row):
    print("\n" + "="*80)
    print(f" 数据来源: {row.get('source', 'Unknown')}")
    print(f" UUID: {row.get('uuid', 'N/A')}")
    print("="*80)

    messages_json = row.get('messages')
    if messages_json:
        try:
            messages = json.loads(messages_json)
            for msg in messages:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                
                print(f"\n[{role}]")
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            print(item.get('text'))
                else:
                    print(content)
        except Exception as e:
            print(f"解析消息 JSON 失败: {e}")
            print(f"原始数据: {messages_json}")
    
    metadata = row.get('metadata')
    if metadata:
        print("\n" + "-"*40)
        print(f" 元数据 (Metadata): {metadata}")
    
    print("\n" + "="*80)

def main():
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/onerec_bench_release.parquet'
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data/label_cond/label_cond_test.parquet'

    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data/ad/ad_test.parquet'
    file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_id_v4/ad/ad_test.parquet'
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_id_v4/ad/ad_test.parquet'
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_id_v4_cleandata_small/video/video_test.parquet'
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_id_v4_cleandata_small/ad/ad_test.parquet'
    
    
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/split_data_pretrain/part-02270-of-33947.parquet'
    # file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/sft_rec_reason.parquet'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    print(f"正在读取文件: {file_path} ...")
    try:
        df = pd.read_parquet(file_path)
        total_rows = len(df)
        print(f"成功加载 {total_rows} 条数据。")
    except Exception as e:
        print(f"读取 Parquet 失败: {e}")
        return

    while True:
        # 随机采样
        idx = random.randint(0, total_rows - 1)
        # idx=22427
        row = df.iloc[idx]
        
        clear_screen()
        print(f"当前采样索引: {idx} / {total_rows}")
        print_message_data(row)
        
        user_input = input("\n[回车] 继续查看下一条, [q] 退出: ").strip().lower()
        if user_input == 'q':
            print("退出脚本。")
            break

if __name__ == "__main__":
    main()
