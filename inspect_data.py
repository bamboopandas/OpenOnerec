import os
import glob
import pandas as pd
import json
import pyarrow.parquet as pq

# 配置显示宽度
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.width', 200)

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def inspect_parquet(file_path, rows=2):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    try:
        # 只读取前几行
        df = pd.read_parquet(file_path).head(rows)
        print(f"✅ 文件: {os.path.basename(file_path)}")
        print(f"   路径: {file_path}")
        print(f"   列名: {list(df.columns)}")
        print(f"   样例数据 (Top {rows}):")
        print(df.to_string(index=False))
        
        # 尝试解析复杂列
        if 'messages' in df.columns and len(df) > 0:
            print("\n   🔍 深度解析 'messages' 列的第一条:")
            msg_content = df.iloc[0]['messages']
            if isinstance(msg_content, str):
                try:
                    print(json.dumps(json.loads(msg_content), indent=2, ensure_ascii=False)[:500] + " ...")
                except:
                    print(msg_content[:500])
            else:
                # 可能是 array
                print(msg_content)
        
        if 'segments' in df.columns and len(df) > 0:
            print("\n   🔍 深度解析 'segments' 列的第一条:")
            seg_content = df.iloc[0]['segments']
            print(seg_content)

    except Exception as e:
        print(f"❌ 读取失败: {e}")

# ================= 1. 检查原始数据 (Raw Data) =================
print_separator("Phase 0: 原始数据 (Raw Data)")

# 1.1 通用文本
general_files = glob.glob("raw_data/general_text/pretrain/*.parquet")
if general_files:
    print(f"📚 发现通用预训练数据: {len(general_files)} 个文件")
    inspect_parquet(general_files[0])
else:
    print("⚠️ 未发现通用预训练数据 (raw_data/general_text/pretrain)")

# 1.2 推荐数据
rec_file = "raw_data/onerec_data/onerec_bench_release.parquet"
if os.path.exists(rec_file):
    print(f"\n📚 发现推荐业务数据")
    inspect_parquet(rec_file)
else:
    print(f"\n⚠️ 未发现推荐业务数据 ({rec_file})")

# 1.3 映射表
mapping_file = "raw_data/onerec_data/video_ad_pid2sid.parquet"
if os.path.exists(mapping_file):
    print(f"\n📚 发现 ID 映射表")
    inspect_parquet(mapping_file)

# ================= 2. 检查处理后的中间数据 (Processed Output) =================
print_separator("Phase 1.1: 处理后的推荐数据 (output/*.parquet)")
output_files = glob.glob("output/*.parquet")
if output_files:
    print(f"📚 发现处理后的文件: {len(output_files)} 个")
    # 找一个 SFT 的和一个 Pretrain 的看
    sft_files = [f for f in output_files if 'sft' in f]
    pretrain_files = [f for f in output_files if 'pretrain' in f]
    
    if sft_files:
        print("\n--- SFT 格式样例 ---")
        inspect_parquet(sft_files[0])
    if pretrain_files:
        print("\n--- Pretrain 格式样例 ---")
        inspect_parquet(pretrain_files[0])
else:
    print("⚠️ output/ 目录下没有 Parquet 文件 (尚未运行 data/onerec_data/run.sh ?)")

# ================= 3. 检查分片数据 (Sharded Split Data) =================
print_separator("Phase 3: 最终分片数据 (output/split_data_*")
split_dirs = glob.glob("output/split_data_*")
if split_dirs:
    for d in split_dirs:
        print(f"\n📂 目录: {d}")
        files = glob.glob(os.path.join(d, "*.parquet"))
        print(f"   包含文件数: {len(files)}")
        if files:
            inspect_parquet(files[0], rows=1)
else:
    print("⚠️ 未发现分片数据目录 (尚未运行 prepare_*.sh ?)")

print("\n" + "="*80)
print("检查结束")
