from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from recipe.onerec.rubric_reward import extract_sid_blocks as extract_candidate_sid_blocks

SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")


def extract_prediction_tail(prediction: str) -> str:
    if "</think>" in prediction and "<think>" in prediction:
        return prediction.split("</think>")[-1]
    return prediction


def extract_sid_blocks(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return extract_candidate_sid_blocks(text)


def _load_records(input_path: str) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted(path.glob("*.jsonl")):
            records.extend(_load_records(str(child)))
        return records
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return payload["samples"]
        return [payload]
    raise ValueError(f"Unsupported input payload type: {type(payload)}")


def _first_sid(text: str) -> str:
    sid_blocks = extract_sid_blocks(extract_prediction_tail(text))
    return sid_blocks[0] if sid_blocks else ""


def _extract_ground_truth_ids(ground_truth: str) -> list[str]:
    return extract_sid_blocks(ground_truth)


def _compute_recall_at_k(predicted_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    if not predicted_ids or not ground_truth_ids:
        return 0.0
    pred_set = {item for item in predicted_ids[:k] if item}
    gt_set = {item for item in ground_truth_ids if item}
    if not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)


def _compute_pass_at_k(predicted_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    if not predicted_ids or not ground_truth_ids:
        return 0.0
    pred_set = {item for item in predicted_ids[:k] if item}
    gt_set = {item for item in ground_truth_ids if item}
    if not gt_set:
        return 0.0
    return float(bool(pred_set & gt_set))


def _compute_ndcg_at_k(predicted_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    if not predicted_ids or not ground_truth_ids:
        return 0.0
    gt_set = {item for item in ground_truth_ids if item}
    if not gt_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(predicted_ids[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record.get("generations"), list):
            grouped_key = json.dumps(
                {
                    "step": record.get("step"),
                    "input": record.get("input", ""),
                    "ground_truth": record.get("ground_truth", ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            grouped[grouped_key] = {
                "input": record.get("input", ""),
                "ground_truth": record.get("ground_truth", ""),
                "outputs": list(record.get("generations", [])),
                "rubric_scores": list(record.get("rubric_scores", [])),
                "objective_anchors": list(record.get("objective_anchors", [])),
                "unresolved_sid_ratios": list(record.get("unresolved_sid_ratios", [])),
            }
            continue

        grouped_key = json.dumps(
            {
                "step": record.get("step"),
                "index": record.get("index"),
                "input": record.get("input", ""),
                "ground_truth": record.get("ground_truth", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        bucket = grouped.setdefault(
            grouped_key,
            {
                "input": record.get("input", ""),
                "ground_truth": record.get("ground_truth", ""),
                "outputs": [],
                "rubric_scores": [],
                "objective_anchors": [],
                "unresolved_sid_ratios": [],
            },
        )
        bucket["outputs"].append(record.get("output", ""))
        if "rubric_score" in record:
            bucket["rubric_scores"].append(float(record["rubric_score"]))
        if "objective_anchor" in record:
            bucket["objective_anchors"].append(float(record["objective_anchor"]))
        if "unresolved_sid_ratio" in record:
            bucket["unresolved_sid_ratios"].append(float(record["unresolved_sid_ratio"]))
    return list(grouped.values())


def evaluate_groups(
    groups: list[dict[str, Any]],
    *,
    k: int,
    pass_ks: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pass_ks = tuple(sorted({int(value) for value in (pass_ks or []) if int(value) > 0}))
    per_sample: list[dict[str, Any]] = []
    hit_rubric_scores: list[float] = []
    miss_rubric_scores: list[float] = []

    for group in groups:
        outputs = list(group.get("outputs", []))
        ground_truth = str(group.get("ground_truth", ""))
        ground_truth_ids = _extract_ground_truth_ids(ground_truth)
        ground_truth_set = set(ground_truth_ids)
        predicted_ids = [_first_sid(output) for output in outputs]
        top1_hit = float(bool(predicted_ids[:1] and predicted_ids[0] in ground_truth_set))
        beam_hit = float(bool(set(predicted_ids[:k]) & ground_truth_set))
        recall = _compute_recall_at_k(predicted_ids, ground_truth_ids, k)
        ndcg = _compute_ndcg_at_k(predicted_ids, ground_truth_ids, k)

        rubric_scores = list(group.get("rubric_scores", []))
        if rubric_scores:
            for idx, pred_id in enumerate(predicted_ids):
                if idx >= len(rubric_scores):
                    break
                if pred_id and pred_id in ground_truth_set:
                    hit_rubric_scores.append(rubric_scores[idx])
                else:
                    miss_rubric_scores.append(rubric_scores[idx])

        sample_metrics = {
            "input": group.get("input", ""),
            "ground_truth": ground_truth,
            "top1_hit": top1_hit,
            f"beam_hit@{k}": beam_hit,
            f"recall@{k}": recall,
            f"ndcg@{k}": ndcg,
            "candidate_count": len(outputs),
        }
        for pass_k in pass_ks:
            sample_metrics[f"pass@{pass_k}"] = _compute_pass_at_k(predicted_ids, ground_truth_ids, pass_k)
        if rubric_scores:
            sample_metrics["top1_rubric_score"] = rubric_scores[0] if rubric_scores else 0.0
        per_sample.append(sample_metrics)

    summary: dict[str, float] = {
        "num_samples": len(per_sample),
        "mean_candidate_count": mean([item["candidate_count"] for item in per_sample]) if per_sample else 0.0,
        "top1_hit": mean([item["top1_hit"] for item in per_sample]) if per_sample else 0.0,
        f"beam_hit@{k}": mean([item[f"beam_hit@{k}"] for item in per_sample]) if per_sample else 0.0,
        f"recall@{k}": mean([item[f"recall@{k}"] for item in per_sample]) if per_sample else 0.0,
        f"ndcg@{k}": mean([item[f"ndcg@{k}"] for item in per_sample]) if per_sample else 0.0,
    }
    for pass_k in pass_ks:
        summary[f"pass@{pass_k}"] = mean([item[f"pass@{pass_k}"] for item in per_sample]) if per_sample else 0.0

    top1_rubric_values = [item["top1_rubric_score"] for item in per_sample if "top1_rubric_score" in item]
    if top1_rubric_values:
        summary["top1_rubric_score"] = mean(top1_rubric_values)
    if hit_rubric_scores:
        summary["mean_hit_rubric_score"] = mean(hit_rubric_scores)
    if miss_rubric_scores:
        summary["mean_miss_rubric_score"] = mean(miss_rubric_scores)
    if hit_rubric_scores and miss_rubric_scores:
        summary["hit_minus_miss_rubric_gap"] = mean(hit_rubric_scores) - mean(miss_rubric_scores)

    unresolved_values: list[float] = []
    objective_values: list[float] = []
    for group in groups:
        unresolved_values.extend(float(value) for value in group.get("unresolved_sid_ratios", []))
        objective_values.extend(float(value) for value in group.get("objective_anchors", []))
    if unresolved_values:
        summary["unresolved_sid_ratio"] = mean(unresolved_values)
    if objective_values:
        summary["objective_anchor"] = mean(objective_values)

    return summary, per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Beauty proxy generations")
    parser.add_argument("--input_path", required=True, help="JSONL file or directory dumped by trainer.validation_data_dir")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff")
    parser.add_argument("--output_path", help="Optional JSON path for evaluation summary")
    parser.add_argument("--details_path", help="Optional JSON path for per-sample details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.input_path)
    groups = _group_records(records)
    summary, per_sample = evaluate_groups(groups, k=args.k)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if args.details_path:
        details_path = Path(args.details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            json.dump(per_sample, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
