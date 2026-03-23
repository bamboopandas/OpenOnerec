from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.onerec.rubric_reward import DEFAULT_RUBRICS, OpenAICompatJudgeClient, parse_json_like, sanitize_schema_name

logger = logging.getLogger(__name__)


INITIAL_RUBRIC_PROMPT = """你是一名推荐系统奖励设计专家。请根据给定 schema 和上下文，产出一组二值、可验证、互不重叠的 rubric。

要求：
1. 输出必须是 JSON 数组。
2. 每一项都必须包含 `criterion`, `weight`, `explanation`。
3. `weight` 只能是 1, 2, 3。
4. criterion 必须只依赖给定上下文与候选 item 的语义信息，不能依赖 SID 字符串本身。
5. rubric 应优先区分高质量候选之间的细微差异，而不是只抓显而易见的错误。

schema_id:
{schema_id}

examples:
{examples_json}
"""


RTD_PROMPT = """你是一名推荐系统奖励设计专家，需要根据两条高质量但有差异的候选响应来精炼现有 rubric。

要求：
1. 输出必须是 JSON 数组。
2. 保留仍然有效的 criterion，可新增、拆分或改写，但不要输出与上下文无关的 rubric。
3. 新 rubric 必须更容易区分 response_a 和 response_b 这类高质量近邻候选。
4. 每一项都必须包含 `criterion`, `weight`, `explanation`。

schema_id:
{schema_id}

current_rubric:
{rubric_json}

context:
{context_json}

response_a:
{response_a}

response_b:
{response_b}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build schema-level rubrics for OneRec RL")
    parser.add_argument("--input_file", required=True, help="Enriched RL parquet with extra_info")
    parser.add_argument("--output_dir", required=True, help="Directory for rubric json files")
    parser.add_argument("--sample_size_per_schema", type=int, default=2000, help="Max samples per schema")
    parser.add_argument("--rtd_rounds", type=int, default=4, help="Number of RTD refinement rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--proposer_base_url", default="none", help="OpenAI-compatible proposer endpoint")
    parser.add_argument("--proposer_model", default="none", help="OpenAI-compatible proposer model")
    parser.add_argument(
        "--candidate_pool_file",
        default="",
        help="Optional parquet/jsonl file keyed by `uuid` with a `candidate_responses` list column",
    )
    return parser.parse_args()


def _load_client(base_url: str, model: str) -> OpenAICompatJudgeClient | None:
    if base_url.lower() in {"", "none", "null"} or model.lower() in {"", "none", "null"}:
        return None
    try:
        return OpenAICompatJudgeClient(base_url=base_url, model=model, timeout_s=60)
    except Exception as exc:
        logger.warning("failed to initialize proposer client: %s", exc)
        return None


def _parse_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        try:
            return json.loads(raw_messages)
        except Exception:
            return ast.literal_eval(raw_messages)
    return []


def _extract_prompt_text(row: pd.Series) -> str:
    messages = _parse_messages(row.get("messages"))
    prompt_parts: list[str] = []
    for message in messages:
        if message.get("role") == "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        prompt_parts.append(
            "".join(
                segment.get("text", "")
                for segment in content
                if isinstance(segment, dict) and segment.get("type") == "text"
            )
        )
    return "\n".join(part for part in prompt_parts if part).strip()


def _load_candidate_pool(path: str) -> dict[str, list[str]]:
    if not path:
        return {}
    candidate_path = Path(path)
    if not candidate_path.exists():
        logger.warning("candidate pool file not found: %s", candidate_path)
        return {}

    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
        result: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            uuid = str(row.get("uuid", "")).strip()
            candidates = parse_json_like(row.get("candidate_responses"), default=row.get("candidate_responses", []))
            if uuid and isinstance(candidates, list):
                result[uuid] = [str(item) for item in candidates if item]
        return result

    result: dict[str, list[str]] = {}
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_json_like(line, default={})
            if not isinstance(record, dict):
                continue
            uuid = str(record.get("uuid", "")).strip()
            candidates = record.get("candidate_responses", [])
            if uuid and isinstance(candidates, list):
                result[uuid] = [str(item) for item in candidates if item]
    return result


def _default_examples(schema_df: pd.DataFrame, sample_size: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    sampled_df = schema_df.head(sample_size)
    for _, row in sampled_df.iterrows():
        extra_info = parse_json_like(row.get("extra_info"), default={})
        examples.append(
            {
                "prompt_text": extra_info.get("prompt_text") or _extract_prompt_text(row),
                "query_text": extra_info.get("query_text", ""),
                "interaction_type": extra_info.get("interaction_type", ""),
                "history_item_captions": extra_info.get("history_item_captions", []),
                "history_ad_captions": extra_info.get("history_ad_captions", []),
                "history_product_captions": extra_info.get("history_product_captions", []),
                "ground_truth_captions": extra_info.get("ground_truth_captions", []),
            }
        )
    return examples


def _generate_initial_rubric(
    client: OpenAICompatJudgeClient | None,
    schema_id: str,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if client is None:
        return DEFAULT_RUBRICS.get(schema_id, DEFAULT_RUBRICS.get(schema_id.split("::")[0], DEFAULT_RUBRICS["unknown"]))

    prompt = INITIAL_RUBRIC_PROMPT.format(
        schema_id=schema_id,
        examples_json=json.dumps(examples, ensure_ascii=False, indent=2),
    )
    raw_response = client.generate(prompt)
    parsed = parse_json_like(raw_response, default=[])
    if isinstance(parsed, list) and parsed:
        return parsed
    logger.warning("proposer returned invalid initial rubric for %s, falling back to defaults", schema_id)
    return DEFAULT_RUBRICS.get(schema_id, DEFAULT_RUBRICS["unknown"])


def _refine_rubric_with_rtd(
    client: OpenAICompatJudgeClient | None,
    schema_id: str,
    rubric: list[dict[str, Any]],
    context: dict[str, Any],
    candidate_responses: list[str],
    rtd_rounds: int,
) -> list[dict[str, Any]]:
    if client is None or len(candidate_responses) < 2:
        return rubric

    current_rubric = rubric
    for round_idx in range(rtd_rounds):
        response_a = candidate_responses[(2 * round_idx) % len(candidate_responses)]
        response_b = candidate_responses[(2 * round_idx + 1) % len(candidate_responses)]
        prompt = RTD_PROMPT.format(
            schema_id=schema_id,
            rubric_json=json.dumps(current_rubric, ensure_ascii=False, indent=2),
            context_json=json.dumps(context, ensure_ascii=False, indent=2),
            response_a=response_a,
            response_b=response_b,
        )
        raw_response = client.generate(prompt)
        refined = parse_json_like(raw_response, default=[])
        if isinstance(refined, list) and refined:
            current_rubric = refined
        else:
            logger.warning("round %d RTD refinement failed for %s, keeping previous rubric", round_idx + 1, schema_id)
    return current_rubric


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_file)
    if "extra_info" not in df.columns:
        raise ValueError("input parquet must contain `extra_info` column")

    proposer_client = _load_client(args.proposer_base_url, args.proposer_model)
    candidate_pool = _load_candidate_pool(args.candidate_pool_file)

    summary: dict[str, Any] = {}
    schema_rows: dict[str, list[int]] = {}
    for idx, extra_info_raw in enumerate(df["extra_info"]):
        extra_info = parse_json_like(extra_info_raw, default={})
        schema_id = str(extra_info.get("schema_id", "unknown"))
        schema_rows.setdefault(schema_id, []).append(idx)

    for schema_id, row_indices in schema_rows.items():
        schema_df = df.iloc[row_indices].sample(
            n=min(len(row_indices), args.sample_size_per_schema),
            random_state=args.seed,
        )
        examples = _default_examples(schema_df, sample_size=min(len(schema_df), 8))
        rubric = _generate_initial_rubric(proposer_client, schema_id, examples)

        if proposer_client is not None and candidate_pool:
            first_row = schema_df.iloc[0]
            context = parse_json_like(first_row.get("extra_info"), default={})
            candidates = candidate_pool.get(str(first_row.get("uuid", "")), [])
            rubric = _refine_rubric_with_rtd(
                client=proposer_client,
                schema_id=schema_id,
                rubric=rubric,
                context=context,
                candidate_responses=candidates,
                rtd_rounds=args.rtd_rounds,
            )

        rubric_path = output_dir / f"{sanitize_schema_name(schema_id)}.json"
        with rubric_path.open("w", encoding="utf-8") as handle:
            json.dump(rubric, handle, ensure_ascii=False, indent=2)

        summary[schema_id] = {
            "num_samples": int(len(schema_df)),
            "rubric_path": str(rubric_path),
            "used_default": proposer_client is None,
            "used_candidate_pool": bool(candidate_pool),
        }
        logger.info("saved rubric for %s to %s", schema_id, rubric_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("saved rubric build summary to %s", summary_path)


if __name__ == "__main__":
    main()
