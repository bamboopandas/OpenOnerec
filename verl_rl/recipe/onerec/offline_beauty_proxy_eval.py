from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.onerec.evaluate_beauty_proxy import evaluate_groups
from recipe.onerec.rubric_reward import (
    build_judge_prompt,
    compute_single_score,
    get_offline_hf_judge_client,
    load_sidecar_index,
    load_rubric,
    parse_json_like,
    resolve_predicted_items,
)

logger = logging.getLogger(__name__)

FORCE_PREFIX_CONTENT = "<think>\n"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")

DEFAULT_STRATEGIES: list[dict[str, Any]] = [
    {"name": "greedy", "do_sample": False, "temperature": None, "top_p": None},
    {"name": "sample_t08_p095", "do_sample": True, "temperature": 0.8, "top_p": 0.95},
    {"name": "sample_t10_p090", "do_sample": True, "temperature": 1.0, "top_p": 0.90},
]


@dataclass
class PromptExample:
    uuid: str
    prompt_messages: list[dict[str, str]]
    ground_truth: str
    extra_info: dict[str, Any]
    raw_messages: list[dict[str, Any]]


class OfflineHFGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        from recipe.onerec.rubric_reward import OfflineHFJudgeClient

        self.model_name_or_path = model_name_or_path
        self.client = OfflineHFJudgeClient(
            model_name_or_path=model_name_or_path,
            max_new_tokens=1,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = self.client.tokenizer
        self.model = self.client.model

    def _render_prompt(self, prompt_messages: list[dict[str, str]], use_force_prefix: bool) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(message["content"] for message in prompt_messages if message.get("content"))

        if use_force_prefix:
            prompt += FORCE_PREFIX_CONTENT
        return prompt

    def generate(
        self,
        prompt_messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        strategy: dict[str, Any],
        use_force_prefix: bool,
    ) -> str:
        import torch

        rendered_prompt = self._render_prompt(prompt_messages, use_force_prefix=use_force_prefix)
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {key: value.to(self.model.device) for key, value in model_inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": bool(strategy.get("do_sample", False)),
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(strategy.get("temperature", 1.0))
            generate_kwargs["top_p"] = float(strategy.get("top_p", 1.0))

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)

        generated_ids = output_ids[0, model_inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if use_force_prefix:
            return FORCE_PREFIX_CONTENT + decoded
        return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Beauty proxy evaluation with local teacher judges")
    parser.add_argument("--input_file", required=True, help="Beauty proxy parquet file, typically test.parquet")
    parser.add_argument("--output_dir", required=True, help="Directory for candidate pools and summaries")
    parser.add_argument("--sidecar_index_path", required=True, help="Sidecar parquet/json for SID captions")
    parser.add_argument("--rubric_dir", required=True, help="Rubric directory")
    parser.add_argument("--candidate_source", default="retrieval", choices=["retrieval", "model", "hybrid"], help="How to build the candidate pool")
    parser.add_argument("--candidate_model_paths", nargs="*", default=[], help="Local candidate generator models used when candidate_source includes model")
    parser.add_argument("--judge_model_path", required=True, help="Local offline judge model used for rubric scoring")
    parser.add_argument("--max_samples", type=int, default=256, help="Max number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for Recall/NDCG")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for candidate generation")
    parser.add_argument("--judge_max_new_tokens", type=int, default=768, help="Max tokens for judge generation")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map for candidate/judge models")
    parser.add_argument("--torch_dtype", default="bfloat16", help="Torch dtype for candidate/judge models")
    parser.add_argument("--attn_implementation", default="none", help="Optional transformers attn implementation")
    parser.add_argument("--use_force_prefix", action="store_true", help="Append the OneRec force prefix before generation")
    parser.add_argument("--candidate_pool_path", default="", help="Optional existing candidate pool jsonl path")
    parser.add_argument("--strategy_preset", default="default", choices=["default", "greedy"], help="Candidate decoding strategy preset")
    parser.add_argument("--num_retrieval_candidates", type=int, default=20, help="Number of heuristic retrieval candidates per sample")
    return parser.parse_args()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            segment.get("text", "")
            for segment in content
            if isinstance(segment, dict) and segment.get("type") == "text"
        )
    return ""


def _extract_example(row: dict[str, Any]) -> PromptExample:
    messages = parse_json_like(row.get("messages"), default=[])
    clean_messages = [
        {
            "role": message.get("role"),
            "content": _message_text(message.get("content", [])),
        }
        for message in messages
        if isinstance(message, dict)
    ]
    prompt_messages = clean_messages[:-1]
    for message in prompt_messages:
        if message.get("role") == "user":
            message["content"] = message["content"] + "/think"
    ground_truth = clean_messages[-1]["content"]
    extra_info = parse_json_like(row.get("extra_info"), default={})
    return PromptExample(
        uuid=str(row.get("uuid", "")),
        prompt_messages=prompt_messages,
        ground_truth=ground_truth,
        extra_info=extra_info,
        raw_messages=messages,
    )


def _strategy_set(preset: str) -> list[dict[str, Any]]:
    if preset == "greedy":
        return [DEFAULT_STRATEGIES[0]]
    return DEFAULT_STRATEGIES


def _model_label(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name or model_path


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "") if token}


def _extract_history_sids(prompt_text: str) -> set[str]:
    return {match.strip() for match in SID_BLOCK_PATTERN.findall(prompt_text or "") if match.strip()}


def _extract_category_leafs(captions: list[str]) -> set[str]:
    leaves: set[str] = set()
    for caption in captions:
        text = str(caption or "")
        if "|" not in text:
            continue
        _, _, tail = text.partition("|")
        normalized = tail.replace("categories:", "").strip()
        if not normalized:
            continue
        for category in normalized.split(">"):
            category = category.strip().lower()
            if category:
                leaves.add(category)
    return leaves


def _load_sidecar_records(sidecar_index_path: str) -> list[dict[str, Any]]:
    path = Path(sidecar_index_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        df = pd.DataFrame(raw_data)

    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        sid = str(record.get("sid", "")).strip()
        if not sid:
            continue
        caption = str(record.get("caption", "")).strip()
        title = str(record.get("title", "")).strip()
        categories = str(record.get("categories", "")).strip()
        text = " | ".join(part for part in (title, categories, caption) if part)
        records.append(
            {
                "sid": sid,
                "caption": caption,
                "title": title,
                "categories": categories,
                "tokens": _tokenize(text),
                "category_leafs": _extract_category_leafs([f"title: {title} | categories: {categories}"]),
            }
        )
    return records


def load_examples(input_file: str, *, max_samples: int, seed: int) -> list[PromptExample]:
    df = pd.read_parquet(input_file)
    if 0 < max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed)
    return [_extract_example(record) for record in df.to_dict(orient="records")]


def _load_candidate_pool(candidate_pool_path: str) -> dict[str, dict[str, Any]]:
    if not candidate_pool_path:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with Path(candidate_pool_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["uuid"])] = record
    return records


def build_retrieval_candidate_pool(
    examples: list[PromptExample],
    *,
    sidecar_index_path: str,
    output_path: str,
    top_k: int,
    existing_pool_path: str,
) -> dict[str, dict[str, Any]]:
    existing_pool = _load_candidate_pool(existing_pool_path)
    if existing_pool:
        logger.info("Loaded existing candidate pool from %s", existing_pool_path)
        return existing_pool

    sidecar_records = _load_sidecar_records(sidecar_index_path)
    output_records: dict[str, dict[str, Any]] = {}
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for example_idx, example in enumerate(examples, start=1):
            prompt_text = str(example.extra_info.get("prompt_text", ""))
            history_sids = _extract_history_sids(prompt_text)
            history_captions = list(example.extra_info.get("history_product_captions", [])) or list(example.extra_info.get("history_item_captions", []))
            history_tokens = _tokenize(" ".join(history_captions))
            history_leafs = _extract_category_leafs(history_captions)

            scored_records: list[tuple[float, dict[str, Any]]] = []
            for record in sidecar_records:
                sid = record["sid"]
                if sid in history_sids:
                    continue
                token_overlap = len(history_tokens & record["tokens"])
                category_overlap = len(history_leafs & record["category_leafs"])
                score = 3.0 * category_overlap + 1.0 * token_overlap
                if score <= 0:
                    continue
                scored_records.append((score, record))

            scored_records.sort(key=lambda item: (item[0], item[1]["sid"]), reverse=True)
            candidates = [
                {
                    "model_path": "retrieval_heuristic",
                    "model_name": "retrieval_heuristic",
                    "strategy": "history_overlap",
                    "output": record["sid"],
                    "retrieval_score": score,
                }
                for score, record in scored_records[:top_k]
            ]

            record = {
                "uuid": example.uuid,
                "ground_truth": example.ground_truth,
                "extra_info": example.extra_info,
                "prompt_messages": example.prompt_messages,
                "candidates": candidates,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_records[example.uuid] = record

            if example_idx % 16 == 0 or example_idx == len(examples):
                logger.info("Built retrieval candidates for %d/%d samples", example_idx, len(examples))

    return output_records


def build_candidate_pool(
    examples: list[PromptExample],
    *,
    candidate_source: str,
    candidate_model_paths: list[str],
    sidecar_index_path: str,
    output_path: str,
    strategy_preset: str,
    max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str | None,
    use_force_prefix: bool,
    existing_pool_path: str,
    num_retrieval_candidates: int,
) -> dict[str, dict[str, Any]]:
    if candidate_source in {"retrieval", "hybrid"}:
        retrieval_records = build_retrieval_candidate_pool(
            examples,
            sidecar_index_path=sidecar_index_path,
            output_path=output_path if candidate_source == "retrieval" else f"{output_path}.retrieval",
            top_k=num_retrieval_candidates,
            existing_pool_path=existing_pool_path if candidate_source == "retrieval" else "",
        )
        if candidate_source == "retrieval":
            return retrieval_records

    existing_pool = _load_candidate_pool(existing_pool_path)
    if existing_pool:
        logger.info("Loaded existing candidate pool from %s", existing_pool_path)
        return existing_pool

    if not candidate_model_paths:
        raise ValueError("candidate_model_paths is required when candidate_source includes model")

    output_records: dict[str, dict[str, Any]] = {}
    strategies = _strategy_set(strategy_preset)
    generators = {
        model_path: OfflineHFGenerator(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        for model_path in candidate_model_paths
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for example_idx, example in enumerate(examples, start=1):
            candidates: list[dict[str, Any]] = []
            seen_outputs: set[str] = set()
            for model_path in candidate_model_paths:
                generator = generators[model_path]
                for strategy in strategies:
                    output = generator.generate(
                        example.prompt_messages,
                        max_new_tokens=max_new_tokens,
                        strategy=strategy,
                        use_force_prefix=use_force_prefix,
                    )
                    normalized = output.strip()
                    if not normalized or normalized in seen_outputs:
                        continue
                    seen_outputs.add(normalized)
                    candidates.append(
                        {
                            "model_path": model_path,
                            "model_name": _model_label(model_path),
                            "strategy": strategy["name"],
                            "output": output,
                        }
                    )

            record = {
                "uuid": example.uuid,
                "ground_truth": example.ground_truth,
                "extra_info": example.extra_info,
                "prompt_messages": example.prompt_messages,
                "candidates": candidates,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_records[example.uuid] = record

            if example_idx % 16 == 0 or example_idx == len(examples):
                logger.info("Generated candidates for %d/%d samples", example_idx, len(examples))

    if candidate_source == "hybrid":
        merged_records: dict[str, dict[str, Any]] = {}
        for example in examples:
            retrieval_record = retrieval_records.get(example.uuid, {})
            model_record = output_records.get(example.uuid, {})
            merged_candidates: list[dict[str, Any]] = []
            seen_outputs: set[str] = set()
            for source_record in (model_record, retrieval_record):
                for candidate in source_record.get("candidates", []):
                    output = str(candidate.get("output", "")).strip()
                    if not output or output in seen_outputs:
                        continue
                    seen_outputs.add(output)
                    merged_candidates.append(candidate)
            merged_records[example.uuid] = {
                "uuid": example.uuid,
                "ground_truth": example.ground_truth,
                "extra_info": example.extra_info,
                "prompt_messages": example.prompt_messages,
                "candidates": merged_candidates,
            }
        output_records = merged_records

    return output_records


def _write_candidate_pool(output_path: str, candidate_pool: dict[str, dict[str, Any]], ordered_uuids: list[str]) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for uuid in ordered_uuids:
            record = candidate_pool.get(uuid)
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_ground_truth_sids(ground_truth: str) -> set[str]:
    return {match.strip() for match in SID_BLOCK_PATTERN.findall(ground_truth or "") if match.strip()}


def _candidate_contains_ground_truth(candidates: list[dict[str, Any]], ground_truth: str) -> bool:
    ground_truth_sids = _extract_ground_truth_sids(ground_truth)
    if not ground_truth_sids:
        return False
    for candidate in candidates:
        output = str(candidate.get("output", ""))
        predicted_sids = {match.strip() for match in SID_BLOCK_PATTERN.findall(output) if match.strip()}
        if predicted_sids & ground_truth_sids:
            return True
    return False


def _build_coverage_summary(records: list[dict[str, Any]], pass_ks: tuple[int, ...]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for pass_k in pass_ks:
        hit_count = 0
        for record in records:
            candidates = list(record.get("candidates", []))
            if _candidate_contains_ground_truth(candidates[:pass_k], str(record.get("ground_truth", ""))):
                hit_count += 1
        summary[f"pass@{pass_k}"] = (hit_count / len(records)) if records else 0.0
    return summary


def _rank_candidates_by_rubric(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enumerated = list(enumerate(scored_candidates))
    ranked = sorted(
        enumerated,
        key=lambda item: (
            float(item[1].get("rubric_score", 0.0)),
            float(item[1].get("retrieval_score", 0.0)),
            -int(item[0]),
        ),
        reverse=True,
    )
    return [candidate for _, candidate in ranked]


def _to_eval_group(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": scored_candidates[0].get("prompt_text", ""),
        "ground_truth": scored_candidates[0]["ground_truth"],
        "outputs": [candidate["output"] for candidate in scored_candidates],
        "rubric_scores": [float(candidate["rubric_score"]) for candidate in scored_candidates],
        "unresolved_sid_ratios": [float(candidate["unresolved_sid_ratio"]) for candidate in scored_candidates],
    }


def evaluate_with_judge(
    examples: list[PromptExample],
    candidate_pool: dict[str, dict[str, Any]],
    *,
    judge_model_path: str,
    rubric_dir: str,
    sidecar_index_path: str,
    output_dir: str,
    cache_path: str,
    k: int,
    judge_max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str | None,
    judge_impl: Any | None = None,
) -> dict[str, Any]:
    if judge_impl is None:
        judge_impl = get_offline_hf_judge_client(
            judge_model=judge_model_path,
            max_new_tokens=judge_max_new_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    ordered_records: list[dict[str, Any]] = []
    for example in examples:
        record = candidate_pool.get(example.uuid)
        if record is None:
            record = {
                "uuid": example.uuid,
                "ground_truth": example.ground_truth,
                "extra_info": example.extra_info,
                "prompt_messages": example.prompt_messages,
                "candidates": [],
            }
        ordered_records.append(record)

    coverage_pass_ks = (10, 20, 50, 100)
    rerank_pass_ks = (1, 5, 10)
    coverage_summary = _build_coverage_summary(ordered_records, coverage_pass_ks)
    sidecar_lookup = load_sidecar_index(sidecar_index_path)
    rubric_snapshot: dict[str, list[dict[str, Any]]] = {}
    judge_prompts: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    raw_groups: list[dict[str, Any]] = []
    rubric_groups: list[dict[str, Any]] = []

    for record in ordered_records:
        candidates = record.get("candidates", [])
        prompt_text = record.get("extra_info", {}).get("prompt_text", "")
        contains_ground_truth = _candidate_contains_ground_truth(candidates, record["ground_truth"])

        detail_record: dict[str, Any] = {
            "uuid": record["uuid"],
            "prompt_text": prompt_text,
            "ground_truth": record["ground_truth"],
            "candidate_count": len(candidates),
            "contains_ground_truth_in_pool": contains_ground_truth,
        }
        if not candidates:
            detail_record["raw_ranking"] = []
            details.append(detail_record)
            continue
        if not contains_ground_truth:
            detail_record["raw_ranking"] = [
                {
                    **candidate,
                    "raw_rank": idx + 1,
                }
                for idx, candidate in enumerate(candidates)
            ]
            details.append(detail_record)
            continue

        scored_candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            schema_id = str(record.get("extra_info", {}).get("schema_id") or record.get("extra_info", {}).get("task_name") or "unknown")
            rubric = load_rubric(schema_id, rubric_dir)
            rubric_snapshot[schema_id] = rubric
            predicted_items, _ = resolve_predicted_items(str(candidate.get("output", "")), sidecar_lookup)
            judge_prompts.append(
                {
                    "uuid": record["uuid"],
                    "raw_rank": idx + 1,
                    "schema_id": schema_id,
                    "candidate_output": candidate.get("output", ""),
                    "retrieval_score": float(candidate.get("retrieval_score", 0.0)),
                    "rubric": rubric,
                    "judge_prompt": build_judge_prompt(
                        schema_id=schema_id,
                        rubric=rubric,
                        extra_info=record["extra_info"],
                        predicted_items=predicted_items,
                        raw_prediction=str(candidate.get("output", "")),
                    ),
                }
            )
            score_payload = compute_single_score(
                data_source="beauty_proxy_offline",
                solution_str=candidate["output"],
                ground_truth=record["ground_truth"],
                extra_info=record["extra_info"],
                rubric_dir=rubric_dir,
                sidecar_index_path=sidecar_index_path,
                cache_path=cache_path,
                timeout_s=60,
                consensus_n=1,
                reward_mode="objective_rubric",
                judge_backend="offline_hf",
                judge_impl=judge_impl,
            )
            scored_candidates.append(
                {
                    **candidate,
                    "rubric_score": float(score_payload["rubric_score"]),
                    "judge_reason": str(score_payload["judge_reason"]),
                    "unresolved_sid_ratio": float(score_payload["unresolved_sid_ratio"]),
                    "cache_hit": float(score_payload["cache_hit"]),
                    "rubric_applied": float(score_payload["rubric_applied"]),
                    "ground_truth": record["ground_truth"],
                    "prompt_text": prompt_text,
                    "raw_rank": idx + 1,
                }
            )

        raw_ranked = list(scored_candidates)
        rubric_ranked = _rank_candidates_by_rubric(scored_candidates)
        raw_groups.append(_to_eval_group(raw_ranked))
        rubric_groups.append(_to_eval_group(rubric_ranked))
        detail_record["raw_ranking"] = raw_ranked
        detail_record["rubric_rerank"] = rubric_ranked
        details.append(detail_record)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "rubric_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(rubric_snapshot, handle, ensure_ascii=False, indent=2)
    with (output_path / "judge_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in judge_prompts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    raw_summary, _ = evaluate_groups(raw_groups, k=k, pass_ks=rerank_pass_ks)
    rubric_summary, _ = evaluate_groups(rubric_groups, k=k, pass_ks=rerank_pass_ks)
    rubric_diagnostics = {
        key: raw_summary.get(key, 0.0)
        for key in ("mean_hit_rubric_score", "mean_miss_rubric_score", "hit_minus_miss_rubric_gap")
        if key in raw_summary
    }

    return {
        "judge_model_path": judge_model_path,
        "judge_model_name": _model_label(judge_model_path),
        "num_total_examples": len(ordered_records),
        "num_rerank_examples": len(raw_groups),
        "candidate_coverage": coverage_summary,
        "raw_ranking": raw_summary,
        "rubric_rerank": rubric_summary,
        "rubric_diagnostics": rubric_diagnostics,
        "delta": {
            "pass@1": rubric_summary.get("pass@1", 0.0) - raw_summary.get("pass@1", 0.0),
            "pass@5": rubric_summary.get("pass@5", 0.0) - raw_summary.get("pass@5", 0.0),
            "pass@10": rubric_summary.get("pass@10", 0.0) - raw_summary.get("pass@10", 0.0),
            f"recall@{k}": rubric_summary.get(f"recall@{k}", 0.0) - raw_summary.get(f"recall@{k}", 0.0),
            f"ndcg@{k}": rubric_summary.get(f"ndcg@{k}", 0.0) - raw_summary.get(f"ndcg@{k}", 0.0),
        },
        "details": details,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    attn_implementation = None if args.attn_implementation.lower() in {"", "none", "null"} else args.attn_implementation

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.input_file, max_samples=args.max_samples, seed=args.seed)
    logger.info("Loaded %d examples from %s", len(examples), args.input_file)

    candidate_pool_path = output_dir / "candidate_pool.jsonl"
    candidate_pool = build_candidate_pool(
        examples,
        candidate_source=args.candidate_source,
        candidate_model_paths=args.candidate_model_paths,
        sidecar_index_path=args.sidecar_index_path,
        output_path=str(candidate_pool_path),
        strategy_preset=args.strategy_preset,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=attn_implementation,
        use_force_prefix=args.use_force_prefix,
        existing_pool_path=args.candidate_pool_path,
        num_retrieval_candidates=args.num_retrieval_candidates,
    )
    _write_candidate_pool(
        output_path=str(output_dir / "candidate_pool.jsonl"),
        candidate_pool=candidate_pool,
        ordered_uuids=[example.uuid for example in examples],
    )

    logger.info("Evaluating candidate pool with judge model: %s", args.judge_model_path)
    judge_result = evaluate_with_judge(
        examples,
        candidate_pool,
        judge_model_path=args.judge_model_path,
        rubric_dir=args.rubric_dir,
        sidecar_index_path=args.sidecar_index_path,
        output_dir=str(output_dir),
        cache_path=str(output_dir / "judge_cache.sqlite"),
        k=args.k,
        judge_max_new_tokens=args.judge_max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=attn_implementation,
    )

    with (output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump(judge_result["details"], handle, ensure_ascii=False, indent=2)

    summary = {
        "input_file": args.input_file,
        "candidate_source": args.candidate_source,
        "candidate_model_paths": args.candidate_model_paths,
        "judge_model_path": args.judge_model_path,
        "num_total_examples": judge_result["num_total_examples"],
        "num_rerank_examples": judge_result["num_rerank_examples"],
        "k": args.k,
        "strategy_preset": args.strategy_preset,
        "use_force_prefix": args.use_force_prefix,
        "num_retrieval_candidates": args.num_retrieval_candidates,
        "candidate_coverage": judge_result["candidate_coverage"],
        "raw_ranking": judge_result["raw_ranking"],
        "rubric_rerank": judge_result["rubric_rerank"],
        "rubric_diagnostics": judge_result["rubric_diagnostics"],
        "delta": judge_result["delta"],
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
