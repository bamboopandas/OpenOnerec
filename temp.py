# # # stage 1
# # # import json
# # # obj = json.load(open("benchmarks/results/v1.0_1000_thinktrue/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"))
# # # # obj = json.load(open("benchmarks/results/v1.0_1000_thinkfalse/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"))
# # # samples = obj["samples"]

# # # gt_has0 = sum(1 for ex in samples.values() if 0 in ex.get("metadata", {}).get("answer_pid", []))
# # # print("GT contains pid=0:", gt_has0, "/", len(samples))

# # # stage 2
# # import json, re

# # path = "benchmarks/results/v1.0_1000_thinktrue/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"
# # obj = json.load(open(path))
# # samples = obj["samples"]

# # def extract_sid_triplet(gen: str):
# #     # try to extract the first <s_a_x><s_b_y><s_c_z>
# #     m = re.search(r"<s_a_(\d+)>\s*<s_b_(\d+)>\s*<s_c_(\d+)>", gen)
# #     return m.group(0) if m else None

# # shown = 0
# # for sid, ex in samples.items():
# #     pids = ex.get("pid_generations", [])[:32]
# #     gens = ex.get("generations", [])[:32]
# #     for i, pid in enumerate(pids):
# #         if pid == 0:
# #             g = gens[i] if i < len(gens) else ""
# #             trip = extract_sid_triplet(g)  # item token 是否完整
# #             print("="*80)
# #             print("sample", sid, "cand", i)
# #             print("triplet:", trip)
# #             # 只看最后 300 字符（通常 item 在末尾）
# #             print("tail:", g[-300:].replace("\n","\\n"))
# #             shown += 1
# #             if shown >= 20:
# #                 raise SystemExit


# # stage 3
# import json
# from statistics import mean

# path_think = "benchmarks/results/v1.0_1000_thinktrue/results_results_1.7B/OneRec-1.7B/ad/test_generated.json"
# obj = json.load(open(path_think))
# samples = obj["samples"]

# def recall32(ex):
#     gt = set(ex["metadata"]["answer_pid"])
#     pids = ex["pid_generations"][:32]
#     pids = [p for p in pids if p != 0]
#     return len(set(pids) & gt) / len(gt) if gt else 0.0

# r0, r1 = [], []
# for ex in samples.values():
#     inv = sum(1 for p in ex["pid_generations"][:32] if p == 0)
#     if inv == 0:
#         r0.append(recall32(ex))
#     else:
#         r1.append(recall32(ex))

# print("recall@32 | no pid=0 :", mean(r0), "n=", len(r0))
# print("recall@32 | has pid=0:", mean(r1), "n=", len(r1))


#print

import pandas as pd
import json
import pprint

# Load the processed file
file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000_test_raganswersummary_v2/video/video_test.parquet'
# file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000_test_raganswersummary/product/product_test.parquet'
# file_path = '/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/raw_data/onerec_data/benchmark_data_1000_test_raganswersummary/ad/ad_test.parquet'
df = pd.read_parquet(file_path)

# Take the first row
for i in range (20):
    row = df.iloc[i].to_dict()

    # Convert JSON strings to objects for clear printing
    if 'metadata' in row and isinstance(row['metadata'], str):
        row['metadata'] = json.loads(row['metadata'])
    if 'messages' in row and isinstance(row['messages'], str):
        row['messages'] = json.loads(row['messages'])

    # Use pprint to display the entire structure clearly
    pprint.pprint(row, sort_dicts=False, width=120)

    # --- 新增判定部分 ---
    user_input = input("\n回车继续查看下一条，输入 'q' 退出: ").strip().lower()
    if user_input == 'q':
        print("已退出循环。")
        break
    print("-" * 40) # 打印分割线，方便区分每一条数据
