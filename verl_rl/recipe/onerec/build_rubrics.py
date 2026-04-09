from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.onerec.rubric_reward import DEFAULT_RUBRICS, get_judge_client, parse_json_like, sanitize_schema_name

logger = logging.getLogger(__name__)


INITIAL_RUBRIC_PROMPT = """你是一名推荐系统奖励设计专家。请根据给定 schema、上下文和真实候选池样例，产出一组适用于重排序(listwise rerank)的 rubric。

要求：
1. 输出必须是 JSON 数组。
2. 每一项都必须包含 `criterion`, `weight`, `explanation`。
3. `weight` 只能是 1, 2, 3。
4. criterion 必须只依赖给定上下文与候选 item 的语义信息，不能依赖 SID 字符串本身。
5. rubric 应优先区分高质量候选之间的细微差异，而不是只抓显而易见的错误。
6. rubric 必须适合整组候选的相对排序，不要写只适用于单个候选绝对分类的标准。
7. 避免空泛标准，如“更好”“更合理”“更符合购买意图”而没有可观察依据。
8. 只能使用当前输入里可观察到的证据：query_text、history_*_captions、candidate_examples 里的 caption。
9. 禁止使用当前输入里看不到的信号：销量、评分、访问量、CTR、价格、库存、评论数、点赞数、分享数、收藏数、图片质量、视频质量。
10. `criterion` 必须写成完整、可验证的句子，不要只写抽象名词。
11. 最多输出 6 条 rubric。

schema_id:
{schema_id}

examples:
{examples_json}
"""


RTD_PROMPT = """你是一名推荐系统奖励设计专家，需要根据同一上下文下两条高质量但有差异的候选响应来精炼现有 rubric。

要求：
1. 输出必须是 JSON 数组。
2. 保留仍然有效的 criterion，可新增、拆分或改写，但不要输出与上下文无关的 rubric。
3. 新 rubric 必须更容易区分 response_a 和 response_b 这类高质量近邻候选。
4. 每一项都必须包含 `criterion`, `weight`, `explanation`。
5. rubric 必须适用于整组候选排序，不要保留无法在候选池中直接观察或比较的标准。
6. 只能使用当前输入里可观察到的证据，禁止引入销量、评分、访问量、图片质量等外部信号。
7. `criterion` 必须写成完整、可验证的句子，不要只写抽象名词。
8. 最多输出 6 条 rubric。

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
    parser.add_argument("--proposer_backend", default="auto", help="auto | openai_compat | offline_hf")
    parser.add_argument("--proposer_max_new_tokens", type=int, default=1024, help="Max proposer generation tokens")
    parser.add_argument("--proposer_device_map", default="auto", help="Transformers device_map for offline proposer")
    parser.add_argument("--proposer_torch_dtype", default="bfloat16", help="Torch dtype for offline proposer")
    parser.add_argument("--proposer_attn_implementation", default="none", help="Optional attention implementation")
    parser.add_argument(
        "--candidate_pool_file",
        default="",
        help="Optional parquet/jsonl file keyed by `uuid` with a `candidate_responses` list column",
    )
    return parser.parse_args()


def _load_client(
    base_url: str,
    model: str,
    *,
    backend: str,
    max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
) -> Any | None:
    if model.lower() in {"", "none", "null"}:
        return None
    try:
        return get_judge_client(
            judge_base_url=base_url,
            judge_model=model,
            timeout_s=60,
            judge_backend=backend,
            judge_max_new_tokens=max_new_tokens,
            judge_device_map=device_map,
            judge_torch_dtype=torch_dtype,
            judge_attn_implementation=attn_implementation,
        )
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


def _load_input_records(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
        return df.to_dict("records")
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = parse_json_like(line, default={})
                if isinstance(record, dict):
                    records.append(record)
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return [item for item in payload["samples"] if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"unsupported input file format: {input_path}")


def _extract_uuid(record: dict[str, Any]) -> str:
    uuid = str(record.get("uuid", "")).strip()
    if uuid:
        return uuid
    metadata = parse_json_like(record.get("metadata"), default={})
    return str(metadata.get("uuid", "")).strip()


def _extract_extra_info(record: dict[str, Any]) -> dict[str, Any]:
    return parse_json_like(record.get("extra_info"), default={})


def _extract_candidate_responses(record: dict[str, Any]) -> list[str]:
    candidate_responses = parse_json_like(record.get("candidate_responses"), default=record.get("candidate_responses", []))
    if isinstance(candidate_responses, list) and candidate_responses:
        return [str(item) for item in candidate_responses if item]

    candidates = parse_json_like(record.get("candidates"), default=record.get("candidates", []))
    if isinstance(candidates, list):
        extracted: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                output = str(candidate.get("output", "")).strip()
                if output:
                    extracted.append(output)
            elif candidate:
                extracted.append(str(candidate))
        if extracted:
            return extracted
    return []


def _truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _truncate_text_list(values: list[Any], max_items: int, max_chars: int) -> list[str]:
    truncated: list[str] = []
    for value in values[:max_items]:
        text = _truncate_text(value, max_chars)
        if text:
            truncated.append(text)
    return truncated


def _extract_candidate_examples(record: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    candidates = parse_json_like(record.get("candidates"), default=record.get("candidates", []))
    examples: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return examples
    for candidate in candidates[:limit]:
        if not isinstance(candidate, dict):
            continue
        predicted_items = parse_json_like(candidate.get("predicted_items"), default=candidate.get("predicted_items", []))
        captions = []
        if isinstance(predicted_items, list):
            for item in predicted_items:
                if not isinstance(item, dict):
                    continue
                caption = str(item.get("caption", "")).strip()
                if caption:
                    captions.append(_truncate_text(caption, 96))
        examples.append(
            {
                "raw_rank": int(candidate.get("raw_rank", 0)),
                "rubric_score": float(candidate.get("rubric_score", 0.0)),
                "captions": captions[:2],
            }
        )
    return examples


def _parse_json_array_response(raw_response: str) -> list[dict[str, Any]]:
    response = str(raw_response or "").strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    parsed = parse_json_like(response, default=None)
    if isinstance(parsed, list) and parsed:
        return parsed

    array_match = re.search(r"(\[\s*\{.*\}\s*\])", response, flags=re.DOTALL)
    if array_match:
        parsed = parse_json_like(array_match.group(1), default=None)
        if isinstance(parsed, list) and parsed:
            return parsed
    return []


UNAVAILABLE_SIGNAL_KEYWORDS = (
    "销量",
    "评分",
    "访问量",
    "ctr",
    "点击率",
    "价格",
    "库存",
    "评论数",
    "点赞数",
    "分享数",
    "收藏数",
    "图片质量",
    "视频质量",
    "detail page",
    "page view",
)


def _sanitize_rubric_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        text_blob = f"{criterion} {explanation}".lower()
        if len(criterion) < 8:
            continue
        if any(keyword in text_blob for keyword in UNAVAILABLE_SIGNAL_KEYWORDS):
            continue
        if criterion in seen:
            continue
        seen.add(criterion)
        try:
            weight = int(round(float(item.get("weight", 1))))
        except (TypeError, ValueError):
            weight = 1
        sanitized.append(
            {
                "criterion": criterion,
                "weight": min(3, max(1, weight)),
                "explanation": explanation[:160],
            }
        )
        if len(sanitized) >= 6:
            break
    return sanitized


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
            record = row.to_dict()
            uuid = _extract_uuid(record)
            candidates = _extract_candidate_responses(record)
            if uuid and candidates:
                result[uuid] = candidates
        return result

    result: dict[str, list[str]] = {}
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_json_like(line, default={})
            if not isinstance(record, dict):
                continue
            uuid = _extract_uuid(record)
            candidates = _extract_candidate_responses(record)
            if uuid and candidates:
                result[uuid] = candidates
    return result


def _default_examples(schema_records: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in schema_records[:sample_size]:
        extra_info = _extract_extra_info(record)
        prompt_text = extra_info.get("prompt_text") or str(record.get("prompt", "")).strip()
        if not prompt_text:
            prompt_text = _extract_prompt_text(pd.Series(record))
        examples.append(
            {
                "prompt_text": _truncate_text(prompt_text, 160),
                "query_text": _truncate_text(extra_info.get("query_text", ""), 120),
                "interaction_type": extra_info.get("interaction_type", ""),
                "history_item_captions": _truncate_text_list(extra_info.get("history_item_captions", []), 3, 80),
                "history_ad_captions": _truncate_text_list(extra_info.get("history_ad_captions", []), 3, 80),
                "history_product_captions": _truncate_text_list(extra_info.get("history_product_captions", []), 3, 80),
                "ground_truth_captions": _truncate_text_list(extra_info.get("ground_truth_captions", []), 1, 80),
                "candidate_examples": _extract_candidate_examples(record, limit=4),
            }
        )
    return examples


def _generate_initial_rubric(
    client: Any | None,
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
    parsed = _sanitize_rubric_items(_parse_json_array_response(raw_response))
    if parsed:
        return parsed
    logger.warning(
        "proposer returned invalid initial rubric for %s, falling back to defaults; raw_response=%r",
        schema_id,
        str(raw_response)[:400],
    )
    return DEFAULT_RUBRICS.get(schema_id, DEFAULT_RUBRICS["unknown"])


def _refine_rubric_with_rtd(
    client: Any | None,
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
        refined = _sanitize_rubric_items(_parse_json_array_response(raw_response))
        if refined:
            current_rubric = refined
        else:
            logger.warning(
                "round %d RTD refinement failed for %s, keeping previous rubric; raw_response=%r",
                round_idx + 1,
                schema_id,
                str(raw_response)[:400],
            )
    return current_rubric


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_input_records(args.input_file)
    if not records:
        raise ValueError("input file contains no records")

    proposer_client = _load_client(
        args.proposer_base_url,
        args.proposer_model,
        backend=args.proposer_backend,
        max_new_tokens=args.proposer_max_new_tokens,
        device_map=args.proposer_device_map,
        torch_dtype=args.proposer_torch_dtype,
        attn_implementation=args.proposer_attn_implementation,
    )
    candidate_pool = _load_candidate_pool(args.candidate_pool_file)

    summary: dict[str, Any] = {}
    schema_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        extra_info = _extract_extra_info(record)
        schema_id = str(extra_info.get("schema_id", "unknown"))
        schema_records.setdefault(schema_id, []).append(record)

    for schema_id, schema_record_list in schema_records.items():
        sampled_records = random.sample(
            schema_record_list,
            k=min(len(schema_record_list), args.sample_size_per_schema),
        )
        examples = _default_examples(sampled_records, sample_size=min(len(sampled_records), 4))
        rubric = _generate_initial_rubric(proposer_client, schema_id, examples)

        if proposer_client is not None:
            first_record = sampled_records[0]
            context = _extract_extra_info(first_record)
            uuid = _extract_uuid(first_record)
            candidates = candidate_pool.get(uuid, []) or _extract_candidate_responses(first_record)
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
            "num_samples": int(len(sampled_records)),
            "rubric_path": str(rubric_path),
            "used_default": proposer_client is None,
            "used_candidate_pool": bool(candidate_pool) or bool(_extract_candidate_responses(sampled_records[0])),
        }
        logger.info("saved rubric for %s to %s", schema_id, rubric_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("saved rubric build summary to %s", summary_path)


if __name__ == "__main__":
    main()
