#!/usr/bin/env python3
"""Prepare Beauty proxy RL data from OneRec-Think public inputs.

The script supports two input modes:
1. Raw OneRec-Think files: ``sequential_data_processed.txt`` + ``Beauty.pretrain.json``.
2. Pre-generated OneRec-Think parquet files under a directory containing
   ``training_prediction_sid_data_{train,val,test}.parquet``.

It writes OpenOneRec-compatible chat parquet files for RL plus a Beauty-only
sidecar index used by rubric reward.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "OneRecThink-Beauty"
SOURCE_NAME = "OneRecThink_Beauty_ProductRec"
CONTEXT_VERSION = "v1"
ITEM_DESCRIPTION_PATTERN = re.compile(
    r"(?P<sid><\|sid_begin\|>.*?<\|sid_end\|>),\s*its title is\s*\"(?P<title>.*?)\",\s*its categories are\s*\"(?P<categories>.*?)\"",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_nonempty_scalar(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _normalize_text(item)
            if text:
                return text
        return ""
    return _normalize_text(value)


def _normalize_categories(value: Any) -> str:
    if isinstance(value, list):
        cleaned = [_normalize_text(item) for item in value if _normalize_text(item)]
        return " > ".join(cleaned)
    if isinstance(value, dict):
        cleaned = [f"{_normalize_text(key)}:{_normalize_text(val)}" for key, val in value.items() if _normalize_text(key) or _normalize_text(val)]
        return " > ".join(part for part in cleaned if part)
    return _normalize_text(value)


def _truncate_text(text: str, max_chars: int) -> str:
    text = _normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_caption(title: str, categories: str, max_chars: int) -> str:
    title = _normalize_text(title)
    categories = _normalize_categories(categories)
    if title and categories:
        return _truncate_text(f"{title} | {categories}", max_chars)
    if title:
        return _truncate_text(title, max_chars)
    return _truncate_text(categories, max_chars)


def _resolve_item_key(raw_key: Any, raw_record: dict[str, Any]) -> str:
    for candidate in (raw_key, raw_record.get("item_id"), raw_record.get("iid"), raw_record.get("asin"), raw_record.get("pid")):
        key = _normalize_text(candidate)
        if key:
            return key
    raise ValueError(f"Unable to resolve item key from record: {raw_record}")


def load_beauty_items(items_file: str) -> dict[str, dict[str, str]]:
    items_path = Path(items_file)
    with items_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if isinstance(raw_data, dict):
        iterable = raw_data.items()
    elif isinstance(raw_data, list):
        iterable = [(None, item) for item in raw_data]
    else:
        raise ValueError(f"Unsupported Beauty items structure: {type(raw_data)}")

    items: dict[str, dict[str, str]] = {}
    for raw_key, raw_record in iterable:
        if not isinstance(raw_record, dict):
            continue
        item_key = _resolve_item_key(raw_key, raw_record)
        sid = _normalize_text(raw_record.get("sid"))
        if not sid:
            continue
        title = _normalize_text(raw_record.get("title") or raw_record.get("name"))
        categories = _normalize_categories(raw_record.get("categories"))
        items[item_key] = {
            "item_id": item_key,
            "sid": sid,
            "title": title,
            "categories": categories,
        }
    logger.info("Loaded %d Beauty items from %s", len(items), items_path)
    return items


def build_sidecar_from_items(
    beauty_items: dict[str, dict[str, str]],
    *,
    caption_max_chars: int,
    mapping_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in beauty_items.values():
        sid = item["sid"]
        title = item["title"]
        categories = item["categories"]
        records.append(
            {
                "sid": sid,
                "pid": item["item_id"],
                "caption": _build_caption(title, categories, caption_max_chars),
                "title": title,
                "categories": categories,
                "mapping_file": mapping_name,
            }
        )
    return pd.DataFrame(records, columns=["sid", "pid", "caption", "title", "categories", "mapping_file"])


def _extract_sequence_from_line(
    line: str,
    beauty_items: dict[str, dict[str, str]],
) -> tuple[str, list[dict[str, str]]] | None:
    parts = line.strip().split()
    if len(parts) <= 1:
        return None
    user_id = parts[0]
    item_sequence: list[dict[str, str]] = []
    for item_id in parts[1:]:
        item = beauty_items.get(item_id)
        if item is None:
            continue
        item_sequence.append(item)
    if len(item_sequence) < 2:
        return None
    return user_id, item_sequence


def _parse_prediction_description(description: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in ITEM_DESCRIPTION_PATTERN.finditer(_normalize_text(description)):
        items.append(
            {
                "sid": _normalize_text(match.group("sid")),
                "title": _normalize_text(match.group("title")),
                "categories": _normalize_text(match.group("categories")),
            }
        )
    return items


def _build_messages(user_prompt: str, answer_sid: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a personalized Beauty product recommendation assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer_sid}],
        },
    ]
    return json.dumps(messages, ensure_ascii=False)


def _build_user_prompt(history_items: list[dict[str, str]]) -> str:
    lines = [
        "The user recently clicked the following Beauty products.",
        "Each line contains a SID, the item title, and category information.",
    ]
    for idx, item in enumerate(history_items, start=1):
        lines.append(
            f"{idx}. {item['sid']} | title: {item['title']} | categories: {item['categories']}"
        )
    lines.append(
        "Think about the user's next-click preference, then recommend exactly one product SID."
    )
    lines.append("After </think>, output exactly one SID and nothing else.")
    return "\n".join(lines)


def _build_example(
    *,
    user_id: str,
    split_name: str,
    history_items: list[dict[str, str]],
    target_item: dict[str, str],
    caption_max_chars: int,
) -> dict[str, Any] | None:
    if not history_items:
        return None
    prompt_text = _build_user_prompt(history_items)
    history_captions = [
        _build_caption(item["title"], item["categories"], caption_max_chars)
        for item in history_items
        if _build_caption(item["title"], item["categories"], caption_max_chars)
    ]
    target_caption = _build_caption(target_item["title"], target_item["categories"], caption_max_chars)
    metadata = {
        "uid": user_id,
        "split": split_name,
        "dataset_name": DATASET_NAME,
        "answer_pid": [target_item["item_id"]],
        "answer_iid": [target_item["item_id"]],
        "answer_title": target_item["title"],
        "answer_categories": target_item["categories"],
    }
    extra_info = {
        "task_name": "product",
        "task_variant": "beauty",
        "schema_id": "product",
        "dataset_name": DATASET_NAME,
        "prompt_text": prompt_text,
        "user_profile_text": "",
        "query_text": "",
        "interaction_type": "next_item_click",
        "history_item_captions": history_captions,
        "history_ad_captions": [],
        "history_product_captions": history_captions,
        "ground_truth_sids": [target_item["sid"]],
        "ground_truth_captions": [target_caption] if target_caption else [],
        "context_version": CONTEXT_VERSION,
    }
    return {
        "uuid": str(uuid5(NAMESPACE_URL, f"{DATASET_NAME}:{split_name}:{user_id}")),
        "source": SOURCE_NAME,
        "messages": _build_messages(prompt_text, target_item["sid"]),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


def build_split_frames_from_raw_inputs(
    sequential_file: str,
    items_file: str,
    *,
    max_history_len: int,
    caption_max_chars: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    beauty_items = load_beauty_items(items_file)
    sidecar_df = build_sidecar_from_items(
        beauty_items,
        caption_max_chars=caption_max_chars,
        mapping_name=Path(items_file).name,
    )

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    with Path(sequential_file).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = _extract_sequence_from_line(raw_line, beauty_items)
            if parsed is None:
                continue
            user_id, item_sequence = parsed

            split_sequences = {
                "train": item_sequence[:-2],
                "val": item_sequence[:-1],
                "test": item_sequence,
            }
            for split_name, candidate_sequence in split_sequences.items():
                if len(candidate_sequence) < 2:
                    continue
                target_item = candidate_sequence[-1]
                history_items = candidate_sequence[:-1][-max_history_len:]
                row = _build_example(
                    user_id=user_id,
                    split_name=split_name,
                    history_items=history_items,
                    target_item=target_item,
                    caption_max_chars=caption_max_chars,
                )
                if row is not None:
                    split_rows[split_name].append(row)

    split_frames = {name: pd.DataFrame(rows) for name, rows in split_rows.items()}
    return split_frames, sidecar_df


def build_split_frames_from_prediction_parquets(
    prediction_data_dir: str,
    *,
    max_history_len: int,
    caption_max_chars: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prediction_dir = Path(prediction_data_dir)
    split_frames: dict[str, pd.DataFrame] = {}
    sidecar_records: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "val", "test"):
        parquet_path = prediction_dir / f"training_prediction_sid_data_{split_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing OneRec-Think prediction parquet: {parquet_path}")
        raw_df = pd.read_parquet(parquet_path)
        rows: list[dict[str, Any]] = []
        for _, raw_row in raw_df.iterrows():
            history_items = _parse_prediction_description(raw_row.get("description", ""))[-max_history_len:]
            groundtruth_sid = _normalize_text(raw_row.get("groundtruth"))
            if not history_items or not groundtruth_sid:
                continue
            user_id = _normalize_text(raw_row.get("user_id") or raw_row.get("uid") or raw_row.get("user"))
            if not user_id:
                user_id = f"{split_name}-{len(rows)}"
            target_item = {
                "item_id": _first_nonempty_scalar(
                    raw_row.get("item_id") or raw_row.get("answer_pid") or raw_row.get("answer_iid") or groundtruth_sid
                ),
                "sid": groundtruth_sid,
                "title": _normalize_text(raw_row.get("title")),
                "categories": _normalize_categories(raw_row.get("categories")),
            }
            if not target_item["item_id"]:
                target_item["item_id"] = groundtruth_sid
            row = _build_example(
                user_id=user_id,
                split_name=split_name,
                history_items=history_items,
                target_item=target_item,
                caption_max_chars=caption_max_chars,
            )
            if row is None:
                continue
            rows.append(row)

            for item in history_items + [target_item]:
                sid = _normalize_text(item.get("sid"))
                if not sid or sid in sidecar_records:
                    continue
                sidecar_records[sid] = {
                    "sid": sid,
                    "pid": _normalize_text(item.get("item_id") or sid),
                    "caption": _build_caption(item.get("title", ""), item.get("categories", ""), caption_max_chars),
                    "title": _normalize_text(item.get("title")),
                    "categories": _normalize_text(item.get("categories")),
                    "mapping_file": parquet_path.name,
                }
        split_frames[split_name] = pd.DataFrame(rows)

    sidecar_df = pd.DataFrame(
        list(sidecar_records.values()),
        columns=["sid", "pid", "caption", "title", "categories", "mapping_file"],
    )
    return split_frames, sidecar_df


def _trim_split_frame(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or df.empty or len(df) <= max_rows:
        return df
    return df.iloc[:max_rows].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Beauty proxy RL data for OpenOneRec")
    parser.add_argument("--output_dir", required=True, help="Directory for train/val/test parquet outputs")
    parser.add_argument("--sidecar_index_out", default="sidecar_index.parquet", help="Sidecar output filename")
    parser.add_argument("--max_history_len", type=int, default=50, help="Maximum history length")
    parser.add_argument("--caption_max_chars", type=int, default=96, help="Maximum caption length")
    parser.add_argument("--max_rows_per_split", type=int, default=0, help="Optional cap for each split")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prediction_data_dir",
        help="Directory containing training_prediction_sid_data_{train,val,test}.parquet",
    )
    input_group.add_argument(
        "--sequential_file",
        help="Path to OneRec-Think sequential_data_processed.txt",
    )
    parser.add_argument("--items_file", help="Path to Beauty.pretrain.json (required with --sequential_file)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prediction_data_dir:
        split_frames, sidecar_df = build_split_frames_from_prediction_parquets(
            args.prediction_data_dir,
            max_history_len=args.max_history_len,
            caption_max_chars=args.caption_max_chars,
        )
    else:
        if not args.items_file:
            raise ValueError("--items_file is required when --sequential_file is used")
        split_frames, sidecar_df = build_split_frames_from_raw_inputs(
            args.sequential_file,
            args.items_file,
            max_history_len=args.max_history_len,
            caption_max_chars=args.caption_max_chars,
        )

    for split_name, split_df in split_frames.items():
        trimmed_df = _trim_split_frame(split_df, args.max_rows_per_split)
        split_path = output_dir / f"{split_name}.parquet"
        trimmed_df.to_parquet(split_path, index=False)
        logger.info("Saved %s split to %s with %d rows", split_name, split_path, len(trimmed_df))

    sidecar_out = Path(args.sidecar_index_out)
    if not sidecar_out.is_absolute():
        sidecar_out = output_dir / sidecar_out
    sidecar_df.to_parquet(sidecar_out, index=False)
    logger.info("Saved sidecar index to %s with %d rows", sidecar_out, len(sidecar_df))


if __name__ == "__main__":
    main()
