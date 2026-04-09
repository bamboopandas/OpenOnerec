from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from recipe.onerec.rubric_reward import get_judge_client, parse_json_like

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
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
)
SPECIFIC_TOPIC_KEYWORDS = (
    "搞笑",
    "舞蹈",
    "军事",
    "自拍",
    "穿搭",
    "服装",
    "尴尬",
    "节奏感",
    "校园",
    "文具",
)

ASPECT_VOCABS: dict[str, list[str]] = {
    "video": [
        "history_topic_match",
        "query_intent_match",
        "novelty_without_drift",
        "duplicate_penalty",
        "raw_rank_anchor_candidate",
    ],
    "ad": [
        "history_topic_match",
        "ad_interest_alignment",
        "query_intent_match",
        "duplicate_penalty",
        "raw_rank_anchor_candidate",
    ],
}

RULE_EVIDENCE_WHITELISTS: dict[str, list[str]] = {
    "video": ["query_text", "history_item_captions", "candidate_caption", "raw_rank"],
    "ad": ["query_text", "history_item_captions", "history_ad_captions", "candidate_caption", "raw_rank"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared and adaptive rerank rules from dev/test candidate pools")
    parser.add_argument("--dev_candidate_pool_file", required=True, help="Raw dev candidate_pool.jsonl")
    parser.add_argument("--test_candidate_pool_file", required=True, help="Raw test candidate_pool.jsonl")
    parser.add_argument("--task_name", required=True, choices=["video", "ad"], help="Task/schema name")
    parser.add_argument("--output_dir", required=True, help="Output directory for adaptive rule artifacts")
    parser.add_argument("--proposer_base_url", default="none", help="Optional OpenAI-compatible proposer endpoint")
    parser.add_argument("--proposer_model", required=True, help="Offline or remote proposer model")
    parser.add_argument("--proposer_backend", default="auto", help="auto | openai_compat | offline_hf")
    parser.add_argument("--proposer_max_new_tokens", type=int, default=1024, help="Max proposer generation tokens")
    parser.add_argument("--proposer_device_map", default="auto", help="Transformers device_map for offline proposer")
    parser.add_argument("--proposer_torch_dtype", default="bfloat16", help="Torch dtype for offline proposer")
    parser.add_argument("--proposer_attn_implementation", default="none", help="Optional attention implementation")
    parser.add_argument("--support_top_k", type=int, default=4, help="Top-k dev support pairs retrieved for each test sample")
    parser.add_argument("--candidate_top_k", type=int, default=6, help="Top-k raw candidates considered for negative mining/signatures")
    parser.add_argument("--max_support_pairs", type=int, default=0, help="Optional cap on dev support pairs")
    parser.add_argument("--rule_retry_count", type=int, default=1, help="Retry count when generated rules fail validation")
    parser.add_argument("--build_oracle_same_sample", action="store_true", help="Also build per-dev same-sample oracle rules for diagnostics")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_client(
    base_url: str,
    model: str,
    *,
    backend: str,
    max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
) -> Any:
    client = get_judge_client(
        judge_base_url=base_url,
        judge_model=model,
        timeout_s=60,
        judge_backend=backend,
        judge_max_new_tokens=max_new_tokens,
        judge_device_map=device_map,
        judge_torch_dtype=torch_dtype,
        judge_attn_implementation=attn_implementation,
    )
    if client is None:
        raise ValueError("failed to initialize proposer client")
    return client


def _load_candidate_pool_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_json_like(line, default={})
            if isinstance(record, dict):
                records.append(record)
    return records


def _compact_text(text: Any, max_chars: int = 72) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _semantic_token_set(text: Any) -> set[str]:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return set()
    tokens: set[str] = set()
    for chunk in TOKEN_PATTERN.findall(normalized):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) <= 2:
                tokens.add(chunk)
                continue
            for width in (2, 3):
                if len(chunk) < width:
                    continue
                for idx in range(len(chunk) - width + 1):
                    tokens.add(chunk[idx : idx + width])
        else:
            tokens.add(chunk)
    return tokens


def _token_overlap_score(left_text: Any, right_text: Any) -> float:
    left_tokens = _semantic_token_set(left_text)
    right_tokens = _semantic_token_set(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def _collect_unique_texts(values: list[Any], *, limit: int, max_chars: int = 72) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _compact_text(value, max_chars)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        collected.append(compact)
        if len(collected) >= limit:
            break
    return collected


def _primary_caption(candidate: dict[str, Any]) -> str:
    for item in candidate.get("predicted_items", []):
        caption = _compact_text(item.get("caption", ""), 96)
        if caption:
            return caption
    return ""


def _candidate_is_correct(candidate: dict[str, Any], ground_truth: str, ground_truth_pids: list[Any]) -> bool:
    predicted_pid = candidate.get("predicted_pid")
    if predicted_pid is not None and predicted_pid in set(ground_truth_pids or []):
        return True
    candidate_output = str(candidate.get("output", ""))
    return candidate_output and candidate_output in ground_truth


def _build_history_signatures(record: dict[str, Any], task_name: str) -> dict[str, Any]:
    extra_info = parse_json_like(record.get("extra_info"), default={})
    history_item = _collect_unique_texts(extra_info.get("history_item_captions", []) or [], limit=6, max_chars=56)
    history_ad = _collect_unique_texts(extra_info.get("history_ad_captions", []) or [], limit=4, max_chars=56)
    if task_name == "ad":
        history_signature = " | ".join(history_item + history_ad)
    else:
        history_signature = " | ".join(history_item)
    return {
        "history_item_captions": history_item,
        "history_ad_captions": history_ad,
        "history_signature": history_signature,
    }


def _build_candidate_signature(record: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    captions = _collect_unique_texts([_primary_caption(candidate) for candidate in (record.get("candidates") or [])[:top_k]], limit=top_k, max_chars=64)
    return {
        "candidate_captions": captions,
        "candidate_signature": " | ".join(captions),
    }


def _build_support_pair_records(records: list[dict[str, Any]], task_name: str, *, candidate_top_k: int, max_support_pairs: int) -> list[dict[str, Any]]:
    support_pairs: list[dict[str, Any]] = []
    for record in records:
        extra_info = parse_json_like(record.get("extra_info"), default={})
        ground_truth = str(record.get("ground_truth", ""))
        ground_truth_pids = list(extra_info.get("ground_truth_pids", []) or record.get("ground_truth_pids", []) or [])
        target_caption = _compact_text((extra_info.get("ground_truth_captions") or [""])[0], 96)
        if not target_caption:
            continue

        history_signatures = _build_history_signatures(record, task_name)
        candidate_signatures = _build_candidate_signature(record, top_k=candidate_top_k)
        top_candidates = list(record.get("candidates", []))[:candidate_top_k]
        negative_candidates = [candidate for candidate in top_candidates if not _candidate_is_correct(candidate, ground_truth, ground_truth_pids)]
        if not negative_candidates:
            continue

        negative_1 = negative_candidates[0]
        negative_2 = None
        remaining = negative_candidates[1:]
        if remaining:
            ranked_remaining = sorted(
                remaining,
                key=lambda candidate: (
                    -_token_overlap_score(target_caption, _primary_caption(candidate)),
                    int(candidate.get("raw_rank", 10**6)),
                ),
            )
            negative_2 = ranked_remaining[0]

        for negative_kind, negative_candidate in (("negative_1", negative_1), ("negative_2", negative_2)):
            if negative_candidate is None:
                continue
            negative_caption = _primary_caption(negative_candidate)
            if not negative_caption:
                continue
            support_pairs.append(
                {
                    "support_pair_id": f"{record.get('sample_id', '')}:{negative_kind}",
                    "sample_id": record.get("sample_id", ""),
                    "uuid": record.get("uuid", ""),
                    "task_name": task_name,
                    "query_text": _compact_text(extra_info.get("query_text", ""), 120),
                    "history_item_captions": history_signatures["history_item_captions"],
                    "history_ad_captions": history_signatures["history_ad_captions"],
                    "history_signature": history_signatures["history_signature"],
                    "candidate_signature": candidate_signatures["candidate_signature"],
                    "target_caption": target_caption,
                    "negative_caption": negative_caption,
                    "negative_kind": negative_kind,
                    "target_preferred": True,
                }
            )
            if max_support_pairs > 0 and len(support_pairs) >= max_support_pairs:
                return support_pairs
    return support_pairs


def _score_support_pair(task_name: str, test_record: dict[str, Any], support_pair: dict[str, Any]) -> tuple[float, float]:
    test_history_signatures = _build_history_signatures(test_record, task_name)
    test_candidate_signatures = _build_candidate_signature(test_record, top_k=6)

    if task_name == "video":
        score = (
            0.75 * _token_overlap_score(test_history_signatures["history_signature"], support_pair.get("history_signature", ""))
            + 0.25 * _token_overlap_score(test_candidate_signatures["candidate_signature"], support_pair.get("candidate_signature", ""))
        )
    else:
        item_score = _token_overlap_score(
            " | ".join(test_history_signatures["history_item_captions"]),
            " | ".join(support_pair.get("history_item_captions", [])),
        )
        ad_score = _token_overlap_score(
            " | ".join(test_history_signatures["history_ad_captions"]),
            " | ".join(support_pair.get("history_ad_captions", [])),
        )
        candidate_score = _token_overlap_score(test_candidate_signatures["candidate_signature"], support_pair.get("candidate_signature", ""))
        score = 0.55 * item_score + 0.25 * ad_score + 0.20 * candidate_score

    query_tiebreak = _token_overlap_score(
        parse_json_like(test_record.get("extra_info"), default={}).get("query_text", ""),
        support_pair.get("query_text", ""),
    )
    return score, query_tiebreak


def _retrieve_support_pairs(task_name: str, test_record: dict[str, Any], support_pairs: list[dict[str, Any]], *, support_top_k: int) -> list[dict[str, Any]]:
    scored_pairs: list[dict[str, Any]] = []
    for support_pair in support_pairs:
        score, query_tiebreak = _score_support_pair(task_name, test_record, support_pair)
        scored_pairs.append(
            {
                **support_pair,
                "retrieval_score": round(score, 6),
                "query_tiebreak": round(query_tiebreak, 6),
            }
        )
    scored_pairs.sort(
        key=lambda item: (
            -float(item["retrieval_score"]),
            -float(item["query_tiebreak"]),
            str(item.get("support_pair_id", "")),
        )
    )
    return scored_pairs[:support_top_k]


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
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    array_match = re.search(r"(\[\s*\{.*\}\s*\])", response, flags=re.DOTALL)
    if array_match:
        parsed = parse_json_like(array_match.group(1), default=None)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _build_aspect_extraction_prompt(task_name: str, support_pair: dict[str, Any]) -> str:
    aspect_vocab = ASPECT_VOCABS[task_name]
    payload = {
        "query_text": support_pair.get("query_text", ""),
        "history_item_captions": support_pair.get("history_item_captions", []),
        "history_ad_captions": support_pair.get("history_ad_captions", []),
        "target_caption": support_pair.get("target_caption", ""),
        "negative_caption": support_pair.get("negative_caption", ""),
    }
    return f"""你是一名推荐重排序规则归纳专家。

已知在下面这条监督对中，target item 应该排在 negative item 前面。
你的任务不是写规则，而是从固定词表中识别：哪几个抽象方面支持了“target > negative”。

要求：
1. `aspect_type` 只能从给定词表中选。
2. 具体主题词只能写进 `evidence`，不能把主题词写成方面名称。
3. 不允许输出规则句、权重或 tie 条件。
4. `aspect_type` 必须是抽象比较方面，不允许把“搞笑/舞蹈/军事/穿搭”这类具体主题词写成方面名称。
4. 返回 JSON 数组，每项必须包含：
   - `aspect_type`
   - `winner`
   - `evidence`
   - `strength`
5. `winner` 只能是 `target`。
6. `strength` 只能是 `weak`、`medium`、`strong`。

schema_id:
{task_name}

available_aspect_types:
{json.dumps(aspect_vocab, ensure_ascii=False, indent=2)}

support_pair:
{json.dumps(payload, ensure_ascii=False, indent=2)}

返回格式：
[
  {{
    "aspect_type": "history_topic_match",
    "winner": "target",
    "evidence": {{
      "history_signal": "...",
      "target_signal": "...",
      "negative_signal": "..."
    }},
    "strength": "strong"
  }}
]
"""


def _sanitize_aspect_records(task_name: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_aspects = set(ASPECT_VOCABS[task_name])
    sanitized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        aspect_type = str(item.get("aspect_type", "")).strip()
        if aspect_type not in allowed_aspects or aspect_type in seen:
            continue
        winner = str(item.get("winner", "")).strip().lower()
        if winner != "target":
            continue
        evidence = parse_json_like(item.get("evidence"), default={})
        if not isinstance(evidence, dict):
            continue
        strength = str(item.get("strength", "")).strip().lower()
        if strength not in {"weak", "medium", "strong"}:
            strength = "medium"
        sanitized.append(
            {
                "aspect_type": aspect_type,
                "winner": "target",
                "evidence": {
                    "history_signal": _compact_text(evidence.get("history_signal", ""), 96),
                    "target_signal": _compact_text(evidence.get("target_signal", ""), 96),
                    "negative_signal": _compact_text(evidence.get("negative_signal", ""), 96),
                },
                "strength": strength,
            }
        )
        seen.add(aspect_type)
    return sanitized


def _build_rule_synthesis_prompt(
    task_name: str,
    support_aspects: list[dict[str, Any]],
    current_signatures: dict[str, Any] | None,
    *,
    adaptive: bool,
    errors: list[str] | None = None,
) -> str:
    whitelist = RULE_EVIDENCE_WHITELISTS[task_name]
    current_payload = current_signatures or {}
    mode_text = "当前 test 样本生成一套专属规则" if adaptive else "当前 schema 生成一套共享规则"
    retry_text = ""
    if errors:
        retry_text = "\n上一次生成失败，失败原因如下：\n" + "\n".join(f"- {error}" for error in errors)
    return f"""你是一名推荐重排序规则设计专家，需要根据抽象判别方面证据，{mode_text}。

要求：
1. 输出必须是 JSON 数组。
2. 每条规则必须包含：
   - `rule_id`
   - `rule_text`
   - `evidence_source`
   - `decision_rule`
   - `tie_condition`
   - `weight`
3. `rule_id` 只能从固定 aspect 词表中选，不能自造。
4. `rule_text` 必须是陈述句，不能是问句，不能以“如何比较”开头。
5. `evidence_source` 只能从白名单中选。
6. `decision_rule` 必须显式比较候选 A 和候选 B。
7. `tie_condition` 必须是方面级、通用级条件，不能引用具体主题词。
8. `weight` 只能是 1/2/3，含义分别是：
   - 1 = 弱辅助规则
   - 2 = 常规主规则
   - 3 = 强主规则
9. 规则必须抽象，不能复述支持样本里的具体主题词，不能把“搞笑/舞蹈/军事/穿搭”等词写进 `rule_text`、`decision_rule`、`tie_condition`。
10. 禁止使用外部信号：销量、评分、价格、库存、评论数、点赞数、分享数、收藏数、图片质量、视频质量。
{retry_text}

schema_id:
{task_name}

allowed_rule_ids:
{json.dumps(ASPECT_VOCABS[task_name], ensure_ascii=False, indent=2)}

allowed_evidence_source:
{json.dumps(whitelist, ensure_ascii=False, indent=2)}

support_aspects:
{json.dumps(support_aspects, ensure_ascii=False, indent=2)}

current_sample_signatures:
{json.dumps(current_payload, ensure_ascii=False, indent=2)}

返回格式：
[
  {{
    "rule_id": "history_topic_match",
    "rule_text": "优先选择与用户最近观看主题更连贯的候选。",
    "evidence_source": ["history_item_captions", "candidate_caption"],
    "decision_rule": "比较候选 A 和 B 与最近观看主题的贴近程度，更贴近者胜。",
    "tie_condition": "若 A 和 B 在该方面证据都弱或差异不明显，则 tie。",
    "weight": 3
  }}
]
"""


def _is_question_style(text: str) -> bool:
    normalized = str(text or "").strip()
    return normalized.startswith("如何比较") or normalized.endswith("?") or normalized.endswith("？")


def _validate_rules(task_name: str, rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    allowed_rule_ids = set(ASPECT_VOCABS[task_name])
    allowed_evidence = set(RULE_EVIDENCE_WHITELISTS[task_name])
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            errors.append("rule must be a JSON object")
            continue
        rule_id = str(rule.get("rule_id", "")).strip()
        rule_text = str(rule.get("rule_text", "")).strip()
        evidence_source = list(rule.get("evidence_source", []))
        decision_rule = str(rule.get("decision_rule", "")).strip()
        tie_condition = str(rule.get("tie_condition", "")).strip()
        try:
            weight = int(round(float(rule.get("weight", 1))))
        except (TypeError, ValueError):
            weight = 1

        if rule_id not in allowed_rule_ids:
            errors.append(f"invalid rule_id: {rule_id}")
            continue
        if rule_id in seen_ids:
            errors.append(f"duplicate rule_id: {rule_id}")
            continue
        if _is_question_style(rule_text):
            errors.append(f"question-style rule_text: {rule_id}")
            continue
        if not decision_rule or "A" not in decision_rule or "B" not in decision_rule:
            errors.append(f"decision_rule must compare A and B: {rule_id}")
            continue
        if not tie_condition:
            errors.append(f"missing tie_condition: {rule_id}")
            continue
        if not evidence_source or any(str(source).strip() not in allowed_evidence for source in evidence_source):
            errors.append(f"illegal evidence_source: {rule_id}")
            continue
        text_blob = " ".join([rule_text, decision_rule, tie_condition]).lower()
        if any(keyword in text_blob for keyword in UNAVAILABLE_SIGNAL_KEYWORDS):
            errors.append(f"unavailable external signal used: {rule_id}")
            continue
        if any(keyword in text_blob for keyword in SPECIFIC_TOPIC_KEYWORDS):
            errors.append(f"specific topic keyword leaked into abstract rule: {rule_id}")
            continue
        validated.append(
            {
                "rule_id": rule_id,
                "rule_text": rule_text,
                "evidence_source": [str(source).strip() for source in evidence_source],
                "decision_rule": decision_rule,
                "tie_condition": tie_condition,
                "weight": min(3, max(1, weight)),
            }
        )
        seen_ids.add(rule_id)
    return validated, errors


def _extract_specific_topic_keywords(rules: list[dict[str, Any]]) -> int:
    text_blob = " ".join(
        " ".join(
            [
                str(rule.get("rule_text", "")),
                str(rule.get("decision_rule", "")),
                str(rule.get("tie_condition", "")),
            ]
        )
        for rule in rules
    )
    return int(any(keyword in text_blob for keyword in SPECIFIC_TOPIC_KEYWORDS))


def _generate_aspect_records(client: Any, task_name: str, support_pair: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    prompt = _build_aspect_extraction_prompt(task_name, support_pair)
    raw_response = client.generate(prompt)
    aspect_records = _sanitize_aspect_records(task_name, _parse_json_array_response(raw_response))
    return aspect_records, prompt, raw_response


def _generate_rules(
    client: Any,
    task_name: str,
    support_aspects: list[dict[str, Any]],
    current_signatures: dict[str, Any] | None,
    *,
    adaptive: bool,
    retry_count: int,
) -> tuple[list[dict[str, Any]], list[str], str, str]:
    attempt_errors: list[str] = []
    last_prompt = ""
    last_response = ""
    for _ in range(max(1, retry_count + 1)):
        prompt = _build_rule_synthesis_prompt(
            task_name,
            support_aspects,
            current_signatures,
            adaptive=adaptive,
            errors=attempt_errors,
        )
        raw_response = client.generate(prompt)
        rules, validation_errors = _validate_rules(task_name, _parse_json_array_response(raw_response))
        last_prompt = prompt
        last_response = raw_response
        if rules:
            return rules, validation_errors, last_prompt, last_response
        attempt_errors = validation_errors or ["empty generated rules"]
    return [], attempt_errors, last_prompt, last_response


def _summarize_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    valid = sum(1 for record in records if record.get("is_valid"))
    avg_rule_count = (sum(int(record.get("rule_count", 0)) for record in records) / total) if total else 0.0
    retry_rate = (sum(1 for record in records if int(record.get("retry_count", 0)) > 0) / total) if total else 0.0
    fallback_rate = (sum(1 for record in records if bool(record.get("used_fallback", False))) / total) if total else 0.0
    illegal_evidence_rate = (sum(1 for record in records if any("illegal evidence_source" in error for error in record.get("errors", []))) / total) if total else 0.0
    question_style_rule_rate = (sum(1 for record in records if any("question-style rule_text" in error for error in record.get("errors", []))) / total) if total else 0.0
    specific_topic_keyword_rate = (sum(int(record.get("specific_topic_keyword_flag", 0)) for record in records) / total) if total else 0.0
    return {
        "valid_rule_rate": valid / total if total else 0.0,
        "avg_rule_count": avg_rule_count,
        "rule_retry_rate": retry_rate,
        "fallback_rate": fallback_rate,
        "illegal_evidence_rate": illegal_evidence_rate,
        "question_style_rule_rate": question_style_rule_rate,
        "specific_topic_keyword_rate": specific_topic_keyword_rate,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proposer_client = _load_client(
        args.proposer_base_url,
        args.proposer_model,
        backend=args.proposer_backend,
        max_new_tokens=args.proposer_max_new_tokens,
        device_map=args.proposer_device_map,
        torch_dtype=args.proposer_torch_dtype,
        attn_implementation=args.proposer_attn_implementation,
    )

    dev_records = _load_candidate_pool_records(args.dev_candidate_pool_file)
    test_records = _load_candidate_pool_records(args.test_candidate_pool_file)
    support_pairs = _build_support_pair_records(
        dev_records,
        args.task_name,
        candidate_top_k=args.candidate_top_k,
        max_support_pairs=args.max_support_pairs,
    )
    _write_jsonl(output_dir / "support_pairs.jsonl", support_pairs)

    aspect_cache: dict[str, list[dict[str, Any]]] = {}
    aspect_records_output: list[dict[str, Any]] = []
    for support_pair in support_pairs:
        aspect_records, aspect_prompt, aspect_response = _generate_aspect_records(proposer_client, args.task_name, support_pair)
        aspect_cache[support_pair["support_pair_id"]] = aspect_records
        aspect_records_output.append(
            {
                "support_pair_id": support_pair["support_pair_id"],
                "sample_id": support_pair["sample_id"],
                "uuid": support_pair["uuid"],
                "aspect_records": aspect_records,
                "aspect_prompt": aspect_prompt,
                "aspect_response": aspect_response,
            }
        )
    _write_jsonl(output_dir / "aspect_evidence.jsonl", aspect_records_output)

    shared_support_aspects: list[dict[str, Any]] = []
    for support_pair in support_pairs:
        shared_support_aspects.extend(aspect_cache.get(support_pair["support_pair_id"], []))
    shared_rules, shared_errors, shared_prompt, shared_response = _generate_rules(
        proposer_client,
        args.task_name,
        shared_support_aspects,
        current_signatures={"mode": "shared_rules"},
        adaptive=False,
        retry_count=args.rule_retry_count,
    )
    shared_rules_dir = output_dir / "shared_rules"
    shared_rules_dir.mkdir(parents=True, exist_ok=True)
    _write_json(shared_rules_dir / f"{args.task_name}.json", shared_rules)

    retrieval_records: list[dict[str, Any]] = []
    generated_rule_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    oracle_generated_rule_records: list[dict[str, Any]] = []
    oracle_validation_records: list[dict[str, Any]] = []
    for test_record in test_records:
        retrieved_pairs = _retrieve_support_pairs(args.task_name, test_record, support_pairs, support_top_k=args.support_top_k)
        retrieval_records.append(
            {
                "sample_id": test_record.get("sample_id", ""),
                "uuid": test_record.get("uuid", ""),
                "task_name": args.task_name,
                "retrieved_support_pairs": [
                    {
                        "support_pair_id": pair["support_pair_id"],
                        "sample_id": pair["sample_id"],
                        "uuid": pair["uuid"],
                        "retrieval_score": pair["retrieval_score"],
                        "query_tiebreak": pair["query_tiebreak"],
                    }
                    for pair in retrieved_pairs
                ],
            }
        )
        support_aspects: list[dict[str, Any]] = []
        for pair in retrieved_pairs:
            support_aspects.extend(aspect_cache.get(pair["support_pair_id"], []))
        current_signatures = {
            **_build_history_signatures(test_record, args.task_name),
            **_build_candidate_signature(test_record, top_k=args.candidate_top_k),
            "query_text": _compact_text(parse_json_like(test_record.get("extra_info"), default={}).get("query_text", ""), 96),
        }
        adaptive_rules, validation_errors, rule_prompt, rule_response = _generate_rules(
            proposer_client,
            args.task_name,
            support_aspects,
            current_signatures=current_signatures,
            adaptive=True,
            retry_count=args.rule_retry_count,
        )
        generated_rules = list(adaptive_rules)
        generated_rule_count = len(generated_rules)
        used_fallback = False
        if not adaptive_rules:
            adaptive_rules = shared_rules
            used_fallback = True
        specific_topic_keyword_flag = _extract_specific_topic_keywords(generated_rules)
        generated_rule_records.append(
            {
                "sample_id": test_record.get("sample_id", ""),
                "uuid": test_record.get("uuid", ""),
                "task_name": args.task_name,
                "support_pair_ids": [pair["support_pair_id"] for pair in retrieved_pairs],
                "rules": adaptive_rules,
                "generated_rules": generated_rules,
                "used_fallback": used_fallback,
                "rule_prompt": rule_prompt,
                "rule_response": rule_response,
            }
        )
        validation_records.append(
            {
                "sample_id": test_record.get("sample_id", ""),
                "uuid": test_record.get("uuid", ""),
                "task_name": args.task_name,
                "is_valid": bool(generated_rules),
                "generated_rule_count": generated_rule_count,
                "rule_count": len(adaptive_rules),
                "used_fallback": used_fallback,
                "retry_count": min(args.rule_retry_count, 1 if validation_errors else 0),
                "errors": validation_errors,
                "specific_topic_keyword_flag": specific_topic_keyword_flag,
            }
        )

    _write_jsonl(output_dir / "support_retrieval.jsonl", retrieval_records)
    _write_jsonl(output_dir / "generated_rules.jsonl", generated_rule_records)
    _write_jsonl(output_dir / "rule_validation.jsonl", validation_records)

    oracle_summary: dict[str, Any] | None = None
    if args.build_oracle_same_sample:
        for dev_record in dev_records:
            sample_id = str(dev_record.get("sample_id", "")).strip()
            uuid = str(dev_record.get("uuid", "")).strip()
            own_pairs = [pair for pair in support_pairs if pair.get("sample_id") == sample_id or (uuid and pair.get("uuid") == uuid)]
            support_aspects: list[dict[str, Any]] = []
            for pair in own_pairs:
                support_aspects.extend(aspect_cache.get(pair["support_pair_id"], []))
            current_signatures = {
                **_build_history_signatures(dev_record, args.task_name),
                **_build_candidate_signature(dev_record, top_k=args.candidate_top_k),
                "query_text": _compact_text(parse_json_like(dev_record.get("extra_info"), default={}).get("query_text", ""), 96),
            }
            oracle_rules, validation_errors, rule_prompt, rule_response = _generate_rules(
                proposer_client,
                args.task_name,
                support_aspects,
                current_signatures=current_signatures,
                adaptive=True,
                retry_count=args.rule_retry_count,
            )
            generated_rules = list(oracle_rules)
            generated_rule_count = len(generated_rules)
            used_fallback = False
            if not oracle_rules:
                oracle_rules = shared_rules
                used_fallback = True
            specific_topic_keyword_flag = _extract_specific_topic_keywords(generated_rules)
            oracle_generated_rule_records.append(
                {
                    "sample_id": sample_id,
                    "uuid": uuid,
                    "task_name": args.task_name,
                    "support_pair_ids": [pair["support_pair_id"] for pair in own_pairs],
                    "rules": oracle_rules,
                    "generated_rules": generated_rules,
                    "used_fallback": used_fallback,
                    "rule_prompt": rule_prompt,
                    "rule_response": rule_response,
                }
            )
            oracle_validation_records.append(
                {
                    "sample_id": sample_id,
                    "uuid": uuid,
                    "task_name": args.task_name,
                    "is_valid": bool(generated_rules),
                    "generated_rule_count": generated_rule_count,
                    "rule_count": len(oracle_rules),
                    "used_fallback": used_fallback,
                    "retry_count": min(args.rule_retry_count, 1 if validation_errors else 0),
                    "errors": validation_errors,
                    "specific_topic_keyword_flag": specific_topic_keyword_flag,
                }
            )
        _write_jsonl(output_dir / "oracle_same_sample_rules.jsonl", oracle_generated_rule_records)
        _write_jsonl(output_dir / "oracle_rule_validation.jsonl", oracle_validation_records)
        oracle_summary = {
            "num_records": len(oracle_generated_rule_records),
            "validation": _summarize_validation(oracle_validation_records),
            "path": str((output_dir / "oracle_same_sample_rules.jsonl").resolve()),
        }

    summary = {
        "task_name": args.task_name,
        "num_dev_records": len(dev_records),
        "num_test_records": len(test_records),
        "num_support_pairs": len(support_pairs),
        "shared_rules": {
            "rule_count": len(shared_rules),
            "errors": shared_errors,
            "prompt": shared_prompt,
            "response": shared_response,
            "path": str((shared_rules_dir / f"{args.task_name}.json").resolve()),
        },
        "validation": _summarize_validation(validation_records),
    }
    if oracle_summary is not None:
        summary["oracle_same_sample"] = oracle_summary
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
