#!/usr/bin/env python3
"""Prepare enriched RL data for rubric-based reward.

This script extends the existing RL train/test split workflow by:
1. Merging multiple SFT parquet files used by RL.
2. Building a SID -> semantic sidecar index from mapping/caption parquet files.
3. Deriving `extra_info` JSON for each sample so reward functions can access
   schema ids, semantic history, queries, and ground-truth captions.
4. Splitting the merged dataset into shuffled train/test parquet files.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from train_test_split import load_all_parquet_files, shuffle_dataframe, split_train_test


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

CONTEXT_VERSION = "v1"


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


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_sidecar_index(
    mapping_files: list[str],
    caption_files: list[str],
    caption_max_chars: int,
) -> pd.DataFrame:
    caption_map: dict[Any, str] = {}

    for caption_file in caption_files:
        if not caption_file:
            continue
        caption_df = pd.read_parquet(caption_file)
        pid_column = _guess_pid_column(caption_df)
        caption_column = _guess_caption_column(caption_df)
        for _, row in caption_df[[pid_column, caption_column]].dropna().iterrows():
            pid = row[pid_column]
            if pid in caption_map:
                continue
            caption = _truncate_text(row[caption_column], caption_max_chars)
            if caption:
                caption_map[pid] = caption

    records: list[dict[str, Any]] = []
    seen_sids: set[str] = set()

    for mapping_file in mapping_files:
        if not mapping_file:
            continue
        mapping_df = pd.read_parquet(mapping_file)
        pid_column = _guess_pid_column(mapping_df)

        if "sid" in mapping_df.columns:
            sid_series = mapping_df["sid"]
        elif "codes" in mapping_df.columns:
            sid_series = mapping_df["codes"].map(_codes_to_sid)
        else:
            raise ValueError(f"Mapping parquet {mapping_file} must contain `sid` or `codes` column")

        for pid, sid in zip(mapping_df[pid_column], sid_series, strict=False):
            sid = str(sid).strip() if sid is not None else ""
            if not sid or sid in seen_sids:
                continue
            seen_sids.add(sid)
            records.append(
                {
                    "sid": sid,
                    "pid": pid,
                    "caption": caption_map.get(pid, ""),
                    "mapping_file": mapping_file,
                }
            )

    return pd.DataFrame(records, columns=["sid", "pid", "caption", "mapping_file"])


def load_sidecar_lookup(sidecar_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if sidecar_df.empty:
        return lookup

    for _, row in sidecar_df.iterrows():
        sid = str(row["sid"]).strip()
        if not sid:
            continue
        lookup[sid] = {
            "pid": row.get("pid"),
            "caption": _truncate_text(row.get("caption", ""), max_chars=96),
            "mapping_file": row.get("mapping_file", ""),
        }
    return lookup


def _extract_messages(row: pd.Series) -> list[dict[str, Any]]:
    return _parse_json_like(row.get("messages"), default=[])


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        raw_content = message.get("content", [])
        if isinstance(raw_content, str):
            text = raw_content
        else:
            text = "".join(
                segment.get("text", "")
                for segment in raw_content
                if isinstance(segment, dict) and segment.get("type") == "text"
            )
        cleaned.append({"role": message.get("role", ""), "content": text})
    return cleaned


def _extract_sid_blocks(text: str) -> list[str]:
    if not text:
        return []
    return SID_BLOCK_PATTERN.findall(text)


def _resolve_captions(
    sid_blocks: list[str],
    sidecar_lookup: dict[str, dict[str, Any]],
    limit: int,
    max_chars: int,
) -> list[str]:
    captions: list[str] = []
    seen: set[str] = set()

    for sid in sid_blocks:
        caption = sidecar_lookup.get(sid, {}).get("caption", "")
        caption = _truncate_text(caption, max_chars)
        if not caption or caption in seen:
            continue
        seen.add(caption)
        captions.append(caption)
        if len(captions) >= limit:
            break

    return captions


def _strip_sid_blocks(text: str) -> str:
    text = SID_BLOCK_PATTERN.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _task_name_from_source(source: str) -> str:
    source = (source or "").lower()
    if "videorec" in source:
        return "video"
    if "adrec" in source:
        return "ad"
    if "productrec" in source:
        return "product"
    if "interactiverec" in source:
        return "interactive"
    if "labelcondrec" in source:
        return "label_cond"
    return "unknown"


def _schema_id(task_name: str, metadata: dict[str, Any]) -> str:
    if task_name != "label_cond":
        return task_name
    interaction_type = str(metadata.get("target_interaction", "")).strip()
    if interaction_type:
        return f"label_cond::{interaction_type}"
    return "label_cond"


def _derive_query_text(user_text: str, metadata: dict[str, Any]) -> str:
    keyword = metadata.get("keyword")
    if isinstance(keyword, str) and keyword.strip():
        return keyword.strip()

    regexes = (
        r"用户查询[:：]\s*(.+)",
        r"搜索关键词[:：]\s*(.+)",
        r"当前需求[:：]\s*(.+)",
        r"用户输入[】\]]?\s*(.+)",
    )
    for regex in regexes:
        match = re.search(regex, user_text)
        if match:
            return match.group(1).strip().strip("。")
    return ""


def _classify_history_captions(
    task_name: str,
    user_text: str,
    sidecar_lookup: dict[str, dict[str, Any]],
    history_limit: int,
    caption_max_chars: int,
) -> tuple[list[str], list[str], list[str]]:
    history_item_captions: list[str] = []
    history_ad_captions: list[str] = []
    history_product_captions: list[str] = []

    for line in (segment.strip() for segment in user_text.splitlines()):
        if not line:
            continue
        sid_blocks = _extract_sid_blocks(line)
        if not sid_blocks:
            continue

        resolved = _resolve_captions(sid_blocks, sidecar_lookup, history_limit, caption_max_chars)
        if not resolved:
            continue

        if task_name == "ad" and "广告" in line:
            history_ad_captions.extend(resolved)
        elif task_name == "product" and any(keyword in line for keyword in ("商品", "购物")):
            history_product_captions.extend(resolved)
        else:
            history_item_captions.extend(resolved)

    deduped = []
    for values in (history_item_captions, history_ad_captions, history_product_captions):
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
            if len(output) >= history_limit:
                break
        deduped.append(output)

    return tuple(deduped)  # type: ignore[return-value]


def build_extra_info(
    row: pd.Series,
    sidecar_lookup: dict[str, dict[str, Any]],
    history_limit: int,
    caption_max_chars: int,
) -> dict[str, Any]:
    messages = _clean_messages(_extract_messages(row))
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    prompt_text = "\n".join(message["content"] for message in prompt_messages if message["content"]).strip()
    assistant_text = next((message["content"] for message in reversed(messages) if message["role"] == "assistant"), "")

    source = row.get("source", "")
    metadata = _parse_json_like(row.get("metadata"), default={})
    task_name = _task_name_from_source(str(source))
    schema_id = _schema_id(task_name, metadata)
    query_text = _derive_query_text(prompt_text, metadata)
    interaction_type = str(metadata.get("target_interaction", "")).strip()
    history_item_captions, history_ad_captions, history_product_captions = _classify_history_captions(
        task_name=task_name,
        user_text=prompt_text,
        sidecar_lookup=sidecar_lookup,
        history_limit=history_limit,
        caption_max_chars=caption_max_chars,
    )

    ground_truth_sids = _extract_sid_blocks(assistant_text)
    ground_truth_captions = _resolve_captions(
        ground_truth_sids,
        sidecar_lookup=sidecar_lookup,
        limit=history_limit,
        max_chars=caption_max_chars,
    )

    return {
        "index": int(row.name),
        "task_name": task_name,
        "task_variant": interaction_type or task_name,
        "schema_id": schema_id,
        "user_profile_text": _truncate_text(_strip_sid_blocks(prompt_text), 512),
        "prompt_text": _truncate_text(prompt_text, 2048),
        "query_text": _truncate_text(query_text, 128),
        "interaction_type": interaction_type,
        "history_item_captions": history_item_captions[:history_limit],
        "history_ad_captions": history_ad_captions[:history_limit],
        "history_product_captions": history_product_captions[:history_limit],
        "ground_truth_sids": ground_truth_sids,
        "ground_truth_captions": ground_truth_captions[:history_limit],
        "context_version": CONTEXT_VERSION,
        "metadata": metadata,
    }


def enrich_dataframe(
    df: pd.DataFrame,
    sidecar_lookup: dict[str, dict[str, Any]],
    history_limit: int,
    caption_max_chars: int,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["extra_info"] = enriched.apply(
        lambda row: json.dumps(
            build_extra_info(
                row=row,
                sidecar_lookup=sidecar_lookup,
                history_limit=history_limit,
                caption_max_chars=caption_max_chars,
            ),
            ensure_ascii=False,
        ),
        axis=1,
    )
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare enriched RL data for rubric-based reward")
    parser.add_argument("--input_files", nargs="+", required=True, help="Input parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--test_size", type=int, required=True, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--engine", default="pyarrow", choices=["pyarrow", "fastparquet"], help="Parquet engine")
    parser.add_argument("--train_filename", default="train.parquet", help="Train parquet filename")
    parser.add_argument("--test_filename", default="test.parquet", help="Test parquet filename")
    parser.add_argument(
        "--mapping_files",
        nargs="*",
        default=[],
        help="Optional pid2sid parquet files used to build a sidecar index",
    )
    parser.add_argument(
        "--caption_files",
        nargs="*",
        default=[],
        help="Optional pid2caption parquet files used to build a sidecar index",
    )
    parser.add_argument(
        "--sidecar_index_out",
        default="sidecar_index.parquet",
        help="Relative or absolute output path for the sidecar index parquet",
    )
    parser.add_argument("--history_limit", type=int, default=20, help="Max history captions per field")
    parser.add_argument("--caption_max_chars", type=int, default=96, help="Max caption length in extra_info")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_df = build_sidecar_index(
        mapping_files=args.mapping_files,
        caption_files=args.caption_files,
        caption_max_chars=args.caption_max_chars,
    )
    sidecar_lookup = load_sidecar_lookup(sidecar_df)

    sidecar_out = Path(args.sidecar_index_out)
    if not sidecar_out.is_absolute():
        sidecar_out = output_dir / sidecar_out
    sidecar_out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_df.to_parquet(sidecar_out, index=False)
    logger.info("Saved sidecar index to %s with %d rows", sidecar_out, len(sidecar_df))

    combined_df = load_all_parquet_files(args.input_files, engine=args.engine)
    if combined_df.empty:
        raise SystemExit("No data loaded from input parquet files")

    enriched_df = enrich_dataframe(
        df=combined_df,
        sidecar_lookup=sidecar_lookup,
        history_limit=args.history_limit,
        caption_max_chars=args.caption_max_chars,
    )

    train_df, test_df = split_train_test(enriched_df, test_size=args.test_size, seed=args.seed)
    train_df = shuffle_dataframe(train_df, seed=args.seed + 1000)
    test_df = shuffle_dataframe(test_df, seed=args.seed + 2000)

    train_path = output_dir / args.train_filename
    test_path = output_dir / args.test_filename
    train_df.to_parquet(train_path, index=False, compression="snappy")
    test_df.to_parquet(test_path, index=False, compression="snappy")

    logger.info("Saved train parquet to %s with %d rows", train_path, len(train_df))
    logger.info("Saved test parquet to %s with %d rows", test_path, len(test_df))


if __name__ == "__main__":
    main()
