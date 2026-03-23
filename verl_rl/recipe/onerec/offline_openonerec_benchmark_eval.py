from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.onerec.evaluate_beauty_proxy import evaluate_groups
from recipe.onerec.rubric_reward import (
    build_judge_prompt,
    compute_rubric_only_score,
    extract_sid_blocks as extract_candidate_sid_blocks,
    get_offline_hf_judge_client,
    load_rubric,
    load_sidecar_index,
    parse_json_like,
    resolve_predicted_items,
)

logger = logging.getLogger(__name__)

SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")
PID_COLUMN_CANDIDATES = (
    "pid",
    "item_id",
    "goods_pid",
    "product_pid",
    "video_pid",
    "ad_pid",
)
CAPTION_COLUMN_CANDIDATES = (
    "dense_caption",
    "caption",
    "title",
    "item_title",
    "text",
    "content",
    "name",
)
DEFAULT_PASS_KS = (1, 5, 10, 32)
CONTEXT_VERSION = "openonerec_benchmark_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline rubric rerank on OpenOneRec benchmark generations")
    parser.add_argument("--generation_file", required=True, help="Path to benchmark test_generated.json")
    parser.add_argument("--task_name", required=True, choices=["video", "ad", "product", "interactive", "label_cond"])
    parser.add_argument("--task_data_file", required=True, help="Path to the corresponding task parquet")
    parser.add_argument("--output_dir", required=True, help="Directory for summary/details/prompts")
    parser.add_argument("--rubric_dir", required=True, help="Rubric directory")
    parser.add_argument("--judge_model_path", required=True, help="Local offline judge model")
    parser.add_argument("--sidecar_index_path", default="", help="Optional existing sidecar parquet/json")
    parser.add_argument("--mapping_files", nargs="*", default=[], help="Mapping parquet files used to build sidecar if needed")
    parser.add_argument("--caption_files", nargs="*", default=[], help="Caption parquet files used to build sidecar if needed")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on evaluated samples")
    parser.add_argument("--k", type=int, default=32, help="Top-k cutoff for beam_hit/recall/ndcg")
    parser.add_argument("--pass_ks", nargs="*", type=int, default=list(DEFAULT_PASS_KS), help="Pass@k values to report")
    parser.add_argument("--history_limit", type=int, default=20, help="Maximum captions per history field")
    parser.add_argument("--caption_max_chars", type=int, default=96, help="Maximum caption length")
    parser.add_argument("--judge_max_new_tokens", type=int, default=384, help="Max judge generation tokens")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map for judge model")
    parser.add_argument("--torch_dtype", default="bfloat16", help="Torch dtype for judge model")
    parser.add_argument("--attn_implementation", default="none", help="Optional transformers attention implementation")
    parser.add_argument("--coverage_ks", nargs="*", type=int, default=[10, 20, 50, 100], help="Pass@k values for raw candidate coverage")
    parser.add_argument("--shard_id", type=int, default=0, help="Zero-based shard id")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    return parser.parse_args()


def _parse_json_like(raw_value: Any, default: Any) -> Any:
    if raw_value is None:
        return default
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return default
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(stripped)
            except Exception:
                continue
    return default


def _guess_pid_column(df: pd.DataFrame) -> str:
    for column in PID_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"Unable to find pid-like column in columns: {list(df.columns)}")


def _guess_caption_column(df: pd.DataFrame) -> str:
    for column in CAPTION_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    for column in df.columns:
        if column == _guess_pid_column(df):
            continue
        if df[column].dtype == object:
            return column
    raise ValueError(f"Unable to find caption-like column in columns: {list(df.columns)}")


def _codes_to_sid(codes: Any) -> str:
    parsed = _parse_json_like(codes, default=codes)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return ""
    try:
        c0, c1, c2 = int(parsed[0]), int(parsed[1]), int(parsed[2])
    except (TypeError, ValueError):
        return ""
    return f"<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>"


def _sid_value_to_sid(raw_sid: Any) -> str:
    if raw_sid is None:
        return ""
    if hasattr(raw_sid, "tolist"):
        raw_sid = raw_sid.tolist()
    if isinstance(raw_sid, str):
        normalized = raw_sid.strip()
        if not normalized:
            return ""
        if normalized.startswith("<|sid_begin|>"):
            return normalized
        return _codes_to_sid(normalized)
    if isinstance(raw_sid, (list, tuple)):
        return _codes_to_sid(raw_sid)
    return ""


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_sidecar_index(
    mapping_files: list[str],
    caption_files: list[str],
    caption_max_chars: int,
    *,
    allowed_sids: set[str] | None = None,
) -> pd.DataFrame:
    raw_records: list[dict[str, Any]] = []
    seen_sids: set[str] = set()
    matched_pids: set[Any] = set()

    for mapping_file in mapping_files:
        if not mapping_file:
            continue
        mapping_df = pd.read_parquet(mapping_file)
        pid_column = _guess_pid_column(mapping_df)
        if "sid" in mapping_df.columns:
            sid_series = mapping_df["sid"].map(_sid_value_to_sid)
            if allowed_sids is not None:
                keep_mask = sid_series.isin(allowed_sids)
                mapping_df = mapping_df[keep_mask]
                sid_series = sid_series[keep_mask]
        elif "codes" in mapping_df.columns:
            sid_series = mapping_df["codes"].map(_sid_value_to_sid)
        else:
            raise ValueError(f"Mapping parquet {mapping_file} must contain `sid` or `codes` column")

        for pid, sid in zip(mapping_df[pid_column], sid_series, strict=False):
            sid = str(sid).strip() if sid is not None else ""
            if not sid or sid in seen_sids:
                continue
            if allowed_sids is not None and sid not in allowed_sids:
                continue
            seen_sids.add(sid)
            matched_pids.add(pid)
            raw_records.append(
                {
                    "sid": sid,
                    "pid": pid,
                    "mapping_file": mapping_file,
                }
            )

    caption_map = _build_pid_caption_lookup(caption_files, caption_max_chars, allowed_pids=matched_pids)
    records = [
        {
            **record,
            "caption": caption_map.get(record["pid"], ""),
        }
        for record in raw_records
    ]
    return pd.DataFrame(records, columns=["sid", "pid", "caption", "mapping_file"])


def _build_pid_caption_lookup(
    caption_files: list[str],
    caption_max_chars: int,
    *,
    allowed_pids: set[Any] | None = None,
) -> dict[Any, str]:
    lookup: dict[Any, str] = {}
    for caption_file in caption_files:
        if not caption_file:
            continue
        caption_df = pd.read_parquet(caption_file)
        pid_column = _guess_pid_column(caption_df)
        caption_column = _guess_caption_column(caption_df)
        if allowed_pids is not None:
            caption_df = caption_df[caption_df[pid_column].isin(allowed_pids)]
        for _, row in caption_df[[pid_column, caption_column]].dropna().iterrows():
            pid = row[pid_column]
            if allowed_pids is not None and pid not in allowed_pids:
                continue
            if pid in lookup:
                continue
            caption = _truncate_text(row[caption_column], caption_max_chars)
            if caption:
                lookup[pid] = caption
    return lookup


def _ensure_sidecar_index(
    sidecar_index_path: str,
    mapping_files: list[str],
    caption_files: list[str],
    caption_max_chars: int,
    *,
    allowed_sids: set[str] | None = None,
) -> str:
    if sidecar_index_path:
        sidecar_path = Path(sidecar_index_path)
        if sidecar_path.exists():
            return str(sidecar_path)

    if not mapping_files or not caption_files:
        raise ValueError("mapping_files and caption_files are required when sidecar_index_path is missing")

    sidecar_path = Path(sidecar_index_path) if sidecar_index_path else Path(caption_files[0]).with_name("rubric_sidecar_index.parquet")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_df = _build_sidecar_index(
        mapping_files,
        caption_files,
        caption_max_chars,
        allowed_sids=allowed_sids,
    )
    sidecar_df.to_parquet(sidecar_path, index=False)
    logger.info("Built sidecar index at %s with %d rows", sidecar_path, len(sidecar_df))
    return str(sidecar_path)


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


def _extract_messages(row: pd.Series) -> list[dict[str, Any]]:
    return _parse_json_like(row.get("messages"), default=[])


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        cleaned.append(
            {
                "role": str(message.get("role", "")),
                "content": _message_text(message.get("content", [])),
            }
        )
    return cleaned


def _extract_sid_blocks(text: str) -> list[str]:
    if not text:
        return []
    return extract_candidate_sid_blocks(text)


def _strip_sid_blocks(text: str) -> str:
    stripped = SID_BLOCK_PATTERN.sub(" ", text or "")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _parse_pid_sequence(raw_value: Any) -> list[Any]:
    if raw_value is None:
        return []
    if hasattr(raw_value, "tolist"):
        raw_value = raw_value.tolist()
    parsed = _parse_json_like(raw_value, default=raw_value)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return []


def _resolve_pid_captions(pids: list[Any], pid_caption_lookup: dict[Any, str], limit: int) -> list[str]:
    captions: list[str] = []
    seen: set[str] = set()
    for pid in pids:
        caption = _truncate_text(pid_caption_lookup.get(pid, ""), 96)
        if not caption or caption in seen:
            continue
        seen.add(caption)
        captions.append(caption)
        if len(captions) >= limit:
            break
    return captions


def _resolve_sidecar_captions(sids: list[str], sidecar_lookup: dict[str, dict[str, Any]], limit: int) -> list[str]:
    captions: list[str] = []
    seen: set[str] = set()
    for sid in sids:
        caption = _truncate_text(sidecar_lookup.get(sid, {}).get("caption", ""), 96)
        if not caption or caption in seen:
            continue
        seen.add(caption)
        captions.append(caption)
        if len(captions) >= limit:
            break
    return captions


def _schema_id(task_name: str, metadata: dict[str, Any]) -> str:
    if task_name != "label_cond":
        return task_name
    interaction_type = str(metadata.get("target_interaction", "")).strip()
    if interaction_type:
        return f"label_cond::{interaction_type}"
    return "label_cond"


def _build_extra_info(
    task_name: str,
    row: pd.Series,
    metadata: dict[str, Any],
    ground_truth: str,
    pid_caption_lookup: dict[Any, str],
    sidecar_lookup: dict[str, dict[str, Any]],
    history_limit: int,
    caption_max_chars: int,
) -> dict[str, Any]:
    del caption_max_chars
    messages = _clean_messages(_extract_messages(row))
    system_text = " ".join(message["content"] for message in messages if message.get("role") == "system").strip()
    user_text = " ".join(message["content"] for message in messages if message.get("role") == "user").strip()

    history_item_captions: list[str] = []
    history_ad_captions: list[str] = []
    history_product_captions: list[str] = []

    if task_name == "video":
        history_item_captions = _resolve_pid_captions(_parse_pid_sequence(row.get("hist_pid")), pid_caption_lookup, history_limit)
    elif task_name == "ad":
        history_item_captions = _resolve_pid_captions(_parse_pid_sequence(row.get("hist_longview")), pid_caption_lookup, history_limit)
        history_ad_captions = _resolve_pid_captions(_parse_pid_sequence(row.get("hist_ad")), pid_caption_lookup, history_limit)
    elif task_name == "product":
        history_item_captions = _resolve_pid_captions(_parse_pid_sequence(row.get("hist_longview")), pid_caption_lookup, history_limit)
        history_product_captions = _resolve_pid_captions(_parse_pid_sequence(row.get("hist_goods")), pid_caption_lookup, history_limit)
    elif task_name == "label_cond":
        label_histories = []
        for column in ("hist_longview", "hist_like", "hist_follow", "hist_forward", "hist_not_interested"):
            label_histories.extend(_parse_pid_sequence(row.get(column)))
        history_item_captions = _resolve_pid_captions(label_histories, pid_caption_lookup, history_limit)
    else:
        history_item_captions = _resolve_sidecar_captions(_extract_sid_blocks(user_text), sidecar_lookup, history_limit)

    ground_truth_sids = _extract_sid_blocks(ground_truth)
    ground_truth_pids = _parse_pid_sequence(metadata.get("answer_pid")) or _parse_pid_sequence(metadata.get("answer_iid"))
    ground_truth_captions = _resolve_pid_captions(ground_truth_pids, pid_caption_lookup, history_limit)
    if not ground_truth_captions:
        ground_truth_captions = _resolve_sidecar_captions(ground_truth_sids, sidecar_lookup, history_limit)

    return {
        "task_name": task_name,
        "task_variant": "benchmark_test",
        "schema_id": _schema_id(task_name, metadata),
        "user_profile_text": system_text,
        "query_text": _strip_sid_blocks(user_text),
        "interaction_type": str(metadata.get("target_interaction", "")),
        "history_item_captions": history_item_captions[:history_limit],
        "history_ad_captions": history_ad_captions[:history_limit],
        "history_product_captions": history_product_captions[:history_limit],
        "ground_truth_sids": ground_truth_sids[:history_limit],
        "ground_truth_captions": ground_truth_captions[:history_limit],
        "context_version": CONTEXT_VERSION,
        "prompt_text": user_text,
    }


def _load_generation_samples(
    generation_file: str,
    max_samples: int,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[str, dict[str, Any]]:
    with Path(generation_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", {})
    if not isinstance(samples, dict):
        raise ValueError("generation file must contain `samples` dict")
    ordered_items = list(samples.items())
    if max_samples > 0:
        ordered_items = ordered_items[:max_samples]
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if num_shards > 1:
        ordered_items = [item for index, item in enumerate(ordered_items) if index % num_shards == shard_id]
    return str(payload.get("model_name", "")), dict(ordered_items)


def _collect_candidate_sids(generation_samples: dict[str, Any]) -> set[str]:
    candidate_sids: set[str] = set()
    for sample in generation_samples.values():
        candidate_sids.update(_extract_sid_blocks(str(sample.get("ground_truth", ""))))
        for output in sample.get("generations", []):
            candidate_sids.update(_extract_sid_blocks(str(output)))
    return candidate_sids


def _collect_context_pids(task_df: pd.DataFrame, generation_samples: dict[str, Any]) -> set[Any]:
    context_pids: set[Any] = set()
    for sample in generation_samples.values():
        metadata = parse_json_like(sample.get("metadata"), default={})
        row = _get_task_row(task_df, metadata.get("row_index"))
        for column in (
            "hist_pid",
            "hist_longview",
            "hist_ad",
            "hist_goods",
            "hist_like",
            "hist_follow",
            "hist_forward",
            "hist_not_interested",
        ):
            context_pids.update(_parse_pid_sequence(row.get(column)))
        context_pids.update(_parse_pid_sequence(metadata.get("answer_pid")))
        context_pids.update(_parse_pid_sequence(metadata.get("answer_iid")))
    return context_pids


def _get_task_row(task_df: pd.DataFrame, row_index: Any) -> pd.Series:
    if row_index is None:
        raise KeyError("row_index is missing from generation metadata")
    try:
        normalized_row_index = int(row_index)
    except (TypeError, ValueError) as exc:
        raise KeyError(f"invalid row_index: {row_index}") from exc

    if normalized_row_index in task_df.index:
        row = task_df.loc[normalized_row_index]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    if 0 <= normalized_row_index < len(task_df):
        return task_df.iloc[normalized_row_index]
    raise KeyError(f"row_index {normalized_row_index} not found in task parquet")


def _sort_candidates_by_rubric(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.get("rubric_score", 0.0)),
            int(candidate.get("raw_rank", 0)),
        ),
    )


def _first_candidate_sid(output: str) -> str:
    sid_blocks = _extract_sid_blocks(str(output))
    return sid_blocks[0] if sid_blocks else ""


def _is_rerankable(outputs: list[str], ground_truth: str) -> bool:
    ground_truth_set = set(_extract_sid_blocks(ground_truth))
    if not ground_truth_set:
        return False
    predicted_ids = [_first_candidate_sid(output) for output in outputs]
    return any(predicted_id in ground_truth_set for predicted_id in predicted_ids if predicted_id)


def _build_group(prompt: str, ground_truth: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": prompt,
        "ground_truth": ground_truth,
        "outputs": [candidate["output"] for candidate in candidates],
        "rubric_scores": [candidate["rubric_score"] for candidate in candidates],
        "unresolved_sid_ratios": [candidate["unresolved_sid_ratio"] for candidate in candidates],
    }


def _select_summary_keys(summary: dict[str, float], prefixes: tuple[str, ...]) -> dict[str, float]:
    selected: dict[str, float] = {}
    for key, value in summary.items():
        if key.startswith(prefixes):
            selected[key] = value
    return selected


def _compute_delta(raw_summary: dict[str, float], rerank_summary: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in set(raw_summary) & set(rerank_summary):
        if key.startswith(("pass@", "recall@", "ndcg@", "beam_hit@")) or key == "top1_hit":
            delta[key] = rerank_summary[key] - raw_summary.get(key, 0.0)
    return delta


def summarize_candidate_pool_records(
    candidate_pool_records: list[dict[str, Any]],
    *,
    k: int,
    pass_ks: list[int],
    coverage_ks: list[int],
) -> dict[str, Any]:
    raw_groups: list[dict[str, Any]] = []
    rerank_groups: list[dict[str, Any]] = []
    rerankable_raw_groups: list[dict[str, Any]] = []
    rerankable_rerank_groups: list[dict[str, Any]] = []

    for record in candidate_pool_records:
        prompt = str(record.get("prompt", ""))
        ground_truth = str(record.get("ground_truth", ""))
        raw_candidates = list(record.get("candidates", []))
        reranked_candidates = _sort_candidates_by_rubric(raw_candidates)
        raw_group = _build_group(prompt, ground_truth, raw_candidates)
        rerank_group = _build_group(prompt, ground_truth, reranked_candidates)
        raw_groups.append(raw_group)
        rerank_groups.append(rerank_group)
        if _is_rerankable(raw_group["outputs"], ground_truth):
            rerankable_raw_groups.append(raw_group)
            rerankable_rerank_groups.append(rerank_group)

    coverage_summary, _ = evaluate_groups(
        raw_groups,
        k=max([k, *coverage_ks]) if coverage_ks else k,
        pass_ks=coverage_ks,
    )
    all_raw_summary, _ = evaluate_groups(raw_groups, k=k, pass_ks=pass_ks)
    all_rerank_summary, _ = evaluate_groups(rerank_groups, k=k, pass_ks=pass_ks)
    rerankable_raw_summary, _ = evaluate_groups(rerankable_raw_groups, k=k, pass_ks=pass_ks)
    rerankable_rerank_summary, _ = evaluate_groups(rerankable_rerank_groups, k=k, pass_ks=pass_ks)

    return {
        "num_total_examples": len(candidate_pool_records),
        "num_rerank_examples": len(rerankable_raw_groups),
        "candidate_coverage": _select_summary_keys(coverage_summary, ("pass@",)),
        "all_examples": {
            "raw_ranking": all_raw_summary,
            "rubric_rerank": all_rerank_summary,
            "delta": _compute_delta(all_raw_summary, all_rerank_summary),
        },
        "rerankable_subset": {
            "raw_ranking": rerankable_raw_summary,
            "rubric_rerank": rerankable_rerank_summary,
            "delta": _compute_delta(rerankable_raw_summary, rerankable_rerank_summary),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _score_samples(
    *,
    task_name: str,
    task_df: pd.DataFrame,
    generation_samples: dict[str, Any],
    rubric_dir: str,
    sidecar_index_path: str,
    judge_model_path: str,
    cache_path: str,
    history_limit: int,
    caption_max_chars: int,
    judge_max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str | None,
    output_dir: Path,
    caption_files: list[str],
    generation_file: str,
    task_data_file: str,
    context_pids: set[Any],
    k: int,
    pass_ks: list[int],
    coverage_ks: list[int],
) -> dict[str, Any]:
    sidecar_lookup = load_sidecar_index(sidecar_index_path)
    predicted_pids = {record.get("pid") for record in sidecar_lookup.values() if record.get("pid") is not None}
    pid_caption_lookup = _build_pid_caption_lookup(
        caption_files,
        caption_max_chars,
        allowed_pids=set(context_pids) | predicted_pids,
    )
    judge_impl = get_offline_hf_judge_client(
        judge_model=judge_model_path,
        max_new_tokens=judge_max_new_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    details: list[dict[str, Any]] = []
    candidate_pool_records: list[dict[str, Any]] = []
    judge_prompts: list[dict[str, Any]] = []
    rubric_snapshot: dict[str, list[dict[str, Any]]] = {}

    for sample_id, sample in generation_samples.items():
        metadata = parse_json_like(sample.get("metadata"), default={})
        row = _get_task_row(task_df, metadata.get("row_index"))
        ground_truth = str(sample.get("ground_truth", ""))
        extra_info = _build_extra_info(
            task_name=task_name,
            row=row,
            metadata=metadata,
            ground_truth=ground_truth,
            pid_caption_lookup=pid_caption_lookup,
            sidecar_lookup=sidecar_lookup,
            history_limit=history_limit,
            caption_max_chars=caption_max_chars,
        )
        schema_id = str(extra_info.get("schema_id", task_name))
        rubric = load_rubric(schema_id, rubric_dir)
        rubric_snapshot[schema_id] = rubric

        raw_candidates: list[dict[str, Any]] = []
        for rank, output in enumerate(sample.get("generations", []), start=1):
            predicted_items, _ = resolve_predicted_items(str(output), sidecar_lookup)
            judge_prompt = build_judge_prompt(
                schema_id=schema_id,
                rubric=rubric,
                extra_info=extra_info,
                predicted_items=predicted_items,
                raw_prediction=str(output),
            )
            judge_prompts.append(
                {
                    "sample_id": sample_id,
                    "uuid": metadata.get("uuid", ""),
                    "task_name": task_name,
                    "raw_rank": rank,
                    "schema_id": schema_id,
                    "candidate_output": output,
                    "judge_prompt": judge_prompt,
                    "rubric": rubric,
                }
            )
            rubric_payload = compute_rubric_only_score(
                solution_str=str(output),
                extra_info=extra_info,
                rubric_dir=rubric_dir,
                sidecar_index_path=sidecar_index_path,
                cache_path=cache_path,
                timeout_s=60,
                consensus_n=1,
                judge_backend="offline_hf",
                judge_model=judge_model_path,
                judge_max_new_tokens=judge_max_new_tokens,
                judge_device_map=device_map,
                judge_torch_dtype=torch_dtype,
                judge_attn_implementation=attn_implementation,
                judge_impl=judge_impl,
            )
            raw_candidates.append(
                {
                    "output": output,
                    "raw_rank": rank,
                    "rubric_score": float(rubric_payload["rubric_score"]),
                    "judge_reason": rubric_payload["judge_reason"],
                    "unresolved_sid_ratio": float(rubric_payload["unresolved_sid_ratio"]),
                    "cache_hit": float(rubric_payload["cache_hit"]),
                    "rubric_applied": float(rubric_payload["rubric_applied"]),
                }
            )

        reranked_candidates = _sort_candidates_by_rubric(raw_candidates)
        rerankable = _is_rerankable([candidate["output"] for candidate in raw_candidates], ground_truth)
        candidate_pool_records.append(
            {
                "sample_id": sample_id,
                "uuid": metadata.get("uuid", ""),
                "task_name": task_name,
                "prompt": sample.get("prompt", ""),
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "candidates": raw_candidates,
                "rerankable": rerankable,
            }
        )
        details.append(
            {
                "sample_id": sample_id,
                "uuid": metadata.get("uuid", ""),
                "task_name": task_name,
                "extra_info": extra_info,
                "ground_truth": ground_truth,
                "prompt": sample.get("prompt", ""),
                "rerankable": rerankable,
                "raw_ranking": raw_candidates,
                "rubric_rerank": [
                    {
                        **candidate,
                        "rerank_rank": idx + 1,
                    }
                    for idx, candidate in enumerate(reranked_candidates)
                ],
            }
        )

    _write_json(output_dir / "details.json", details)
    _write_json(output_dir / "rubric_snapshot.json", rubric_snapshot)
    with (output_dir / "candidate_pool.jsonl").open("w", encoding="utf-8") as handle:
        for record in candidate_pool_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "judge_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in judge_prompts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_candidate_pool_records(
        candidate_pool_records,
        k=k,
        pass_ks=pass_ks,
        coverage_ks=coverage_ks,
    )

    return {
        **summary,
        "task_name": task_name,
        "generation_file": str(Path(generation_file).resolve()),
        "task_data_file": str(Path(task_data_file).resolve()),
        "judge_model_path": judge_model_path,
        "sidecar_index_path": sidecar_index_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name, generation_samples = _load_generation_samples(
        args.generation_file,
        args.max_samples,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    task_df = pd.read_parquet(args.task_data_file)
    context_pids = _collect_context_pids(task_df, generation_samples)

    sidecar_index_path = _ensure_sidecar_index(
        sidecar_index_path=args.sidecar_index_path,
        mapping_files=args.mapping_files,
        caption_files=args.caption_files,
        caption_max_chars=args.caption_max_chars,
        allowed_sids=_collect_candidate_sids(generation_samples),
    )
    cache_path = str(output_dir / "judge_cache.sqlite")
    summary = _score_samples(
        task_name=args.task_name,
        task_df=task_df,
        generation_samples=generation_samples,
        rubric_dir=args.rubric_dir,
        sidecar_index_path=sidecar_index_path,
        judge_model_path=args.judge_model_path,
        cache_path=cache_path,
        history_limit=args.history_limit,
        caption_max_chars=args.caption_max_chars,
        judge_max_new_tokens=args.judge_max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        output_dir=output_dir,
        caption_files=args.caption_files,
        generation_file=args.generation_file,
        task_data_file=args.task_data_file,
        context_pids=context_pids,
        k=args.k,
        pass_ks=args.pass_ks,
        coverage_ks=args.coverage_ks,
    )
    summary["model_name"] = model_name
    summary["k"] = args.k
    summary["pass_ks"] = list(args.pass_ks)
    summary["coverage_ks"] = list(args.coverage_ks)
    summary["shard_id"] = args.shard_id
    summary["num_shards"] = args.num_shards

    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
