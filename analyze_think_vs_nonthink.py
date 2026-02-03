import json
import argparse
from statistics import mean, median

def spearman_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def rank(data):
        idx = sorted(range(n), key=lambda i: data[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and data[idx[j + 1]] == data[idx[i]]:
                j += 1
            avg = (i + j + 2) / 2.0  # ranks start from 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sum((rx[i] - mx) ** 2 for i in range(n))
    deny = sum((ry[i] - my) ** 2 for i in range(n))
    if denx == 0 or deny == 0:
        return None
    return num / ((denx ** 0.5) * (deny ** 0.5))

def get_samples(obj):
    # your file has top-level "samples"
    if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], dict):
        return obj["samples"]
    # fallback: maybe already samples dict
    if isinstance(obj, dict):
        return obj
    return {}

def safe_list(x):
    return x if isinstance(x, list) else []

def compute_metrics(ex):
    # output_tokens: think often [cot, item], nonthink often [item]
    out_tokens = ex.get("output_tokens", [])
    out_tokens = safe_list(out_tokens)
    cot_len = out_tokens[0] if len(out_tokens) >= 2 else 0
    item_len = out_tokens[1] if len(out_tokens) >= 2 else (out_tokens[0] if len(out_tokens) == 1 else 0)

    # pid_generations: list of 32 ints; sometimes contains 0 as invalid
    pids = safe_list(ex.get("pid_generations", []))[:32]
    valid_pids = [p for p in pids if isinstance(p, int) and p != 0]
    invalid_cnt = 32 - len(valid_pids)

    unique_pid = len(set(valid_pids))
    dup_ratio = (1 - unique_pid / len(valid_pids)) if len(valid_pids) > 0 else 1.0

    # ground truth pids from metadata.answer_pid
    md = ex.get("metadata", {})
    md = md if isinstance(md, dict) else {}
    gt = md.get("answer_pid", [])
    gt = set([p for p in safe_list(gt) if isinstance(p, int) and p != 0])

    # best GT rank in the 32 candidates
    best_rank = 999
    if gt:
        for i, pid in enumerate(pids):
            if pid in gt:
                best_rank = i + 1
                break

    # recompute recall@k using unique GT set
    def recall_at(k):
        if not gt:
            return 0.0
        topk = [p for p in pids[:k] if isinstance(p, int) and p != 0]
        hit = len(set(topk) & gt)
        return hit / len(gt)

    return {
        "cot_len": cot_len,
        "item_len": item_len,
        "invalid_cnt": invalid_cnt,
        "unique_pid": unique_pid,
        "dup_ratio": dup_ratio,
        "best_gt_rank": best_rank,
        "recall1": recall_at(1),
        "recall5": recall_at(5),
        "recall10": recall_at(10),
        "recall32": recall_at(32),
    }

def summarize(rows, name):
    print(f"\n==== {name} ====")
    print(f"n = {len(rows)}")
    if len(rows) == 0:
        print("No rows parsed. Check JSON structure.")
        return

    def q(vals, p):
        vals = sorted(vals)
        idx = int(round((len(vals) - 1) * p))
        return vals[idx]

    keys = ["cot_len","invalid_cnt","unique_pid","dup_ratio","best_gt_rank","recall1","recall5","recall10","recall32"]
    for key in keys:
        vals = [r[key] for r in rows]
        print(f"{key:12s} mean={mean(vals):.4f}  median={median(vals):.4f}  p10={q(vals,0.1):.4f}  p90={q(vals,0.9):.4f}")

    xs = [r["cot_len"] for r in rows]
    for ykey in ["recall32", "best_gt_rank", "unique_pid", "dup_ratio"]:
        ys = [r[ykey] for r in rows]
        corr = spearman_corr(xs, ys)
        print(f"spearman(cot_len, {ykey}) = {corr if corr is not None else 'NA'}")

def load_rows(path):
    with open(path, "r") as f:
        obj = json.load(f)
    samples = get_samples(obj)

    rows = []
    for _, ex in samples.items():
        if isinstance(ex, dict) and ("pid_generations" in ex or "metadata" in ex):
            rows.append(compute_metrics(ex))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", required=True)
    ap.add_argument("--nonthink", required=True)
    args = ap.parse_args()

    think_rows = load_rows(args.think)
    non_rows = load_rows(args.nonthink)

    summarize(think_rows, "THINK")
    summarize(non_rows, "NONTHINK")

    if len(think_rows) and len(non_rows):
        print("\n==== DIFF (THINK - NONTHINK, means) ====")
        for key in ["cot_len","invalid_cnt","unique_pid","dup_ratio","best_gt_rank","recall1","recall5","recall10","recall32"]:
            dt = mean([r[key] for r in think_rows]) - mean([r[key] for r in non_rows])
            print(f"{key:12s} {dt:+.6f}")

if __name__ == "__main__":
    main()
