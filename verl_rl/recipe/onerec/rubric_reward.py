from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>(?:<s_d_(\d+)>)?")
SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")


DEFAULT_RUBRICS: dict[str, list[dict[str, Any]]] = {
    "video": [
        {"criterion": "推荐结果与用户最近观看内容的主题保持连贯。", "weight": 3, "explanation": "强调兴趣延续。"},
        {"criterion": "推荐结果不包含明显重复或无效的项目。", "weight": 2, "explanation": "避免重复和脏输出。"},
        {"criterion": "排序靠前的项目比排序靠后的项目更贴近用户历史兴趣。", "weight": 2, "explanation": "鼓励高质量 top-1。"},
    ],
    "ad": [
        {"criterion": "推荐广告与用户最近的视频兴趣或广告点击偏好相关。", "weight": 3, "explanation": "强调广告相关性。"},
        {"criterion": "推荐广告不与用户已表现出的偏好明显冲突。", "weight": 2, "explanation": "避免错配广告。"},
        {"criterion": "排序靠前的广告具备更强点击合理性。", "weight": 2, "explanation": "鼓励高质量 top-1。"},
    ],
    "product": [
        {"criterion": "推荐商品与用户视频兴趣或历史商品点击记录相关。", "weight": 3, "explanation": "强调跨域语义连通。"},
        {"criterion": "推荐商品列表没有重复，且主题不过度分散。", "weight": 2, "explanation": "控制列表质量。"},
        {"criterion": "排序靠前的商品最符合用户潜在购买意图。", "weight": 2, "explanation": "鼓励高质量 top-1。"},
    ],
    "interactive": [
        {"criterion": "推荐结果直接响应用户当前查询或需求。", "weight": 3, "explanation": "强调 query 对齐。"},
        {"criterion": "推荐结果同时兼顾用户画像与当前查询。", "weight": 3, "explanation": "兼顾长期兴趣和即时意图。"},
        {"criterion": "结果不重复，且首个结果最贴近当前需求。", "weight": 2, "explanation": "鼓励 top-1 质量。"},
    ],
    "label_cond::longview": [
        {"criterion": "推荐结果适合用户长时间观看，而非只具备短时吸引力。", "weight": 3, "explanation": "强调 longview。"},
        {"criterion": "推荐结果与用户既有兴趣保持一致。", "weight": 2, "explanation": "强调兴趣延续。"},
        {"criterion": "结果不重复且排序合理。", "weight": 2, "explanation": "控制输出质量。"},
    ],
    "label_cond::like": [
        {"criterion": "推荐结果具备高点赞潜力，而不只是泛相关。", "weight": 3, "explanation": "强调 like 倾向。"},
        {"criterion": "推荐结果与用户偏好一致。", "weight": 2, "explanation": "强调兴趣匹配。"},
        {"criterion": "结果不重复且排序合理。", "weight": 2, "explanation": "控制输出质量。"},
    ],
    "label_cond::follow": [
        {"criterion": "推荐结果更可能促使用户关注作者或账号。", "weight": 3, "explanation": "强调 follow 倾向。"},
        {"criterion": "推荐结果与用户稳定兴趣一致。", "weight": 2, "explanation": "强调长期偏好。"},
        {"criterion": "结果不重复且排序合理。", "weight": 2, "explanation": "控制输出质量。"},
    ],
    "label_cond::forward": [
        {"criterion": "推荐结果具备被用户转发分享的潜力。", "weight": 3, "explanation": "强调转发性。"},
        {"criterion": "推荐结果与用户兴趣和表达倾向一致。", "weight": 2, "explanation": "强调兴趣匹配。"},
        {"criterion": "结果不重复且排序合理。", "weight": 2, "explanation": "控制输出质量。"},
    ],
    "label_cond::not_interested": [
        {"criterion": "推荐结果更接近用户可能标记为不感兴趣的内容。", "weight": 3, "explanation": "强调 negative preference。"},
        {"criterion": "推荐结果与用户正向兴趣有明显区分。", "weight": 2, "explanation": "避免与正向兴趣混淆。"},
        {"criterion": "结果不重复且排序合理。", "weight": 2, "explanation": "控制输出质量。"},
    ],
}

DEFAULT_RUBRICS["label_cond"] = DEFAULT_RUBRICS["label_cond::like"]
DEFAULT_RUBRICS["unknown"] = DEFAULT_RUBRICS["video"]


def parse_json_like(raw_value: Any, default: Any) -> Any:
    if raw_value is None:
        return default
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return default
        for parser in (json.loads,):
            try:
                return parser(stripped)
            except Exception:
                continue
    return default


def extract_all_tuples(text: Any) -> list[tuple[str, ...]]:
    if not isinstance(text, str):
        return []
    matches = SLOT_PATTERN.findall(text)
    if not matches:
        return []
    normalized_matches: list[tuple[str, ...]] = []
    for match in matches:
        parts = tuple(part for part in match if part != "")
        if parts:
            normalized_matches.append(parts)
    return normalized_matches


def extract_sid_blocks(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    matches = SID_BLOCK_PATTERN.findall(text)
    if matches:
        return matches
    tuples = extract_all_tuples(text)
    sid_blocks: list[str] = []
    for item in tuples:
        if len(item) == 3:
            a, b, c = item
            sid_blocks.append(f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>")
        elif len(item) == 4:
            a, b, c, d = item
            sid_blocks.append(f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><s_d_{d}><|sid_end|>")
    return sid_blocks


def think_format_reward(prediction: str) -> float:
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0

    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")
    if end_idx < start_idx:
        return 0.0

    content = prediction[start_idx:end_idx]
    compact = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return 1.0 if len(compact) > 10 else 0.0


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = extract_all_tuples(prediction)
    gt_tuples = extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    total_reward = 0.0
    for pred_tuple in pred_tuples:
        max_score = 0.0
        for gt_tuple in gt_tuples:
            if pred_tuple == gt_tuple:
                max_score = max(max_score, 100.0)
            elif len(pred_tuple) >= 3 and len(gt_tuple) >= 3 and pred_tuple[:3] == gt_tuple[:3]:
                max_score = max(max_score, 10.0)
            elif len(pred_tuple) >= 2 and len(gt_tuple) >= 2 and pred_tuple[:2] == gt_tuple[:2]:
                max_score = max(max_score, 1.0)
        total_reward += max_score

    return total_reward / len(pred_tuples)


def hit_reward(prediction: str, ground_truth: str) -> float:
    prediction_after_think = extract_prediction_tail(prediction)
    pred_tuples = extract_all_tuples(prediction_after_think)
    gt_tuples = extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return len(pred_set & gt_set) / len(pred_tuples)


def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    prediction_after_think = extract_prediction_tail(prediction)
    pred_tuples = extract_all_tuples(prediction_after_think)
    gt_tuples = extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    return float(pred_tuples[0] in set(gt_tuples))


def pass_rate(prediction: str, ground_truth: str) -> float:
    pred_tuples = extract_all_tuples(prediction)
    gt_tuples = extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    return float(bool(set(pred_tuples) & set(gt_tuples)))


def extract_prediction_tail(prediction: str) -> str:
    if "</think>" in prediction and "<think>" in prediction:
        return prediction.split("</think>")[-1]
    return prediction


def normalize_prediction(prediction: str) -> str:
    sid_blocks = extract_sid_blocks(extract_prediction_tail(prediction))
    if sid_blocks:
        unique: list[str] = []
        seen: set[str] = set()
        for sid in sid_blocks:
            if sid in seen:
                continue
            seen.add(sid)
            unique.append(sid)
        return "".join(unique)
    return extract_prediction_tail(prediction).strip()


@dataclass
class JudgeResult:
    rubric_score: float
    criterion_results: list[dict[str, Any]]
    judge_reason: str


class SqliteJudgeCache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rubric_cache (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, cache_key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM rubric_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return parse_json_like(row[0], default=None)

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        value_json = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO rubric_cache(cache_key, value_json)
                VALUES(?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET value_json = excluded.value_json
                """,
                (cache_key, value_json),
            )
            self._conn.commit()


@lru_cache(maxsize=8)
def get_sqlite_cache(cache_path: str | None) -> Optional[SqliteJudgeCache]:
    if not cache_path or str(cache_path).lower() in {"none", "null", ""}:
        return None
    cache_path = os.path.expanduser(cache_path)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteJudgeCache(cache_path)


@lru_cache(maxsize=4)
def load_sidecar_index(path: str | None) -> dict[str, dict[str, Any]]:
    if not path or str(path).lower() in {"none", "null", ""}:
        return {}

    expanded_path = Path(os.path.expanduser(path))
    if not expanded_path.exists():
        logger.warning("sidecar index path does not exist: %s", expanded_path)
        return {}

    if expanded_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(expanded_path)
    else:
        with expanded_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        df = pd.DataFrame(data)

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row.get("sid", "")).strip()
        if not sid:
            continue
        lookup[sid] = {
            "pid": row.get("pid"),
            "caption": str(row.get("caption", "")).strip(),
            "mapping_file": row.get("mapping_file", ""),
        }
    return lookup


def sanitize_schema_name(schema_id: str) -> str:
    return schema_id.replace("::", "__")


def load_rubric(schema_id: str, rubric_dir: str | None) -> list[dict[str, Any]]:
    if rubric_dir and str(rubric_dir).lower() not in {"none", "null", ""}:
        rubric_path = Path(os.path.expanduser(rubric_dir)) / f"{sanitize_schema_name(schema_id)}.json"
        if rubric_path.exists():
            with rubric_path.open("r", encoding="utf-8") as handle:
                rubric = json.load(handle)
            if isinstance(rubric, list) and rubric:
                return rubric

    return DEFAULT_RUBRICS.get(schema_id) or DEFAULT_RUBRICS.get(schema_id.split("::")[0]) or DEFAULT_RUBRICS["unknown"]


class OpenAICompatJudgeClient:
    def __init__(self, base_url: str, model: str, timeout_s: int, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        if response and response.choices:
            content = response.choices[0].message.content
            if content:
                return content
        raise RuntimeError("judge returned empty response")


def _resolve_torch_dtype(dtype_name: str):
    import torch

    normalized = str(dtype_name or "bfloat16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {dtype_name}")


class OfflineHFJudgeClient:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_new_tokens: int = 768,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(torch_dtype),
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if attn_implementation and str(attn_implementation).lower() not in {"", "none", "null"}:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()
        self.input_device = self._resolve_input_device()

    def _resolve_input_device(self):
        import torch

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if device in {None, "cpu", "disk"}:
                    continue
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str):
                    return torch.device(device)
        return self.model.device

    def _render_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str) -> str:
        import torch

        rendered_prompt = self._render_prompt(prompt)
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {key: value.to(self.input_device) for key, value in model_inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, model_inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@lru_cache(maxsize=4)
def get_offline_hf_judge_client(
    judge_model: str,
    *,
    max_new_tokens: int = 768,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> OfflineHFJudgeClient:
    return OfflineHFJudgeClient(
        model_name_or_path=judge_model,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )


def get_judge_client(
    judge_base_url: str | None,
    judge_model: str | None,
    timeout_s: int,
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 768,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
) -> Optional[Any]:
    if not judge_model:
        return None
    if str(judge_model).lower() in {"none", "null", ""}:
        return None

    normalized_backend = str(judge_backend or "auto").strip().lower()
    normalized_base_url = str(judge_base_url or "").strip().lower()

    if normalized_backend == "auto":
        normalized_backend = "offline_hf" if normalized_base_url in {"", "none", "null"} else "openai_compat"

    try:
        if normalized_backend == "offline_hf":
            return get_offline_hf_judge_client(
                judge_model=judge_model,
                max_new_tokens=judge_max_new_tokens,
                device_map=judge_device_map,
                torch_dtype=judge_torch_dtype,
                attn_implementation=judge_attn_implementation,
            )

        if normalized_base_url in {"", "none", "null"}:
            return None

        return OpenAICompatJudgeClient(judge_base_url, judge_model, timeout_s)
    except Exception as exc:
        logger.warning("failed to initialize judge client: %s", exc)
        return None


def build_judge_prompt(
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
    predicted_items: list[dict[str, Any]],
    raw_prediction: str,
) -> str:
    context = {
        "task_name": extra_info.get("task_name", ""),
        "task_variant": extra_info.get("task_variant", ""),
        "query_text": extra_info.get("query_text", ""),
        "interaction_type": extra_info.get("interaction_type", ""),
        "user_profile_text": extra_info.get("user_profile_text", ""),
        "history_item_captions": extra_info.get("history_item_captions", []),
        "history_ad_captions": extra_info.get("history_ad_captions", []),
        "history_product_captions": extra_info.get("history_product_captions", []),
        "ground_truth_captions": extra_info.get("ground_truth_captions", []),
    }

    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    items_json = json.dumps(predicted_items, ensure_ascii=False, indent=2)

    return f"""你是一名推荐结果评审器，需要根据给定 rubric 评估候选推荐结果。

规则：
1. 逐条判断 rubric 是否满足，`satisfied` 只能是 true 或 false。
2. 只依据给定上下文和候选 item 语义描述判断，不要依赖 SID 字符串本身。
3. `rubric_score` 必须等于满足 criteria 的权重和 / 全部权重和，范围 [0, 1]。
4. `evidence` 和 `judge_reason` 必须尽量短，每条不超过 30 个字。
5. 返回纯 JSON，不要附加说明。

schema_id:
{schema_id}

rubric:
{rubric_json}

context:
{context_json}

candidate_items:
{items_json}

raw_model_response:
{raw_prediction}

返回格式：
{{
  "criterion_results": [
    {{
      "criterion": "...",
      "satisfied": true,
      "evidence": "..."
    }}
  ],
  "rubric_score": 0.0,
  "judge_reason": "..."
}}
"""


def parse_judge_response(raw_response: str, rubric: list[dict[str, Any]]) -> JudgeResult:
    response = raw_response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    parsed = parse_json_like(response, default=None)
    if not isinstance(parsed, dict):
        parsed = {}

    criterion_results = parsed.get("criterion_results", [])
    if not isinstance(criterion_results, list):
        criterion_results = []

    if not criterion_results:
        recovered_bools = re.findall(r'"satisfied"\s*:\s*(true|false)', response, flags=re.IGNORECASE)
        criterion_results = [
            {
                "criterion": criterion.get("criterion", ""),
                "satisfied": recovered_bools[idx].lower() == "true",
                "evidence": "",
            }
            for idx, criterion in enumerate(rubric)
            if idx < len(recovered_bools)
        ]

    normalized_results: list[dict[str, Any]] = []
    for idx, criterion in enumerate(rubric):
        criterion_text = criterion.get("criterion", "")
        weight = float(criterion.get("weight", 1))
        raw_result = criterion_results[idx] if idx < len(criterion_results) and isinstance(criterion_results[idx], dict) else {}
        normalized_results.append(
            {
                "criterion": raw_result.get("criterion", criterion_text),
                "satisfied": bool(raw_result.get("satisfied", False)),
                "weight": weight,
                "evidence": str(raw_result.get("evidence", "")),
            }
        )

    total_weight = sum(float(item.get("weight", 1.0)) for item in rubric) or 1.0
    satisfied_weight = sum(item["weight"] for item in normalized_results if item["satisfied"])
    rubric_score = parsed.get("rubric_score")
    if not isinstance(rubric_score, (int, float)):
        rubric_score_match = re.search(r'"rubric_score"\s*:\s*([0-9]*\.?[0-9]+)', response, flags=re.IGNORECASE)
        if rubric_score_match:
            try:
                rubric_score = float(rubric_score_match.group(1))
            except ValueError:
                rubric_score = None
    if not isinstance(rubric_score, (int, float)):
        rubric_score = satisfied_weight / total_weight

    rubric_score = max(0.0, min(1.0, float(rubric_score)))
    judge_reason = str(parsed.get("judge_reason", ""))
    if not judge_reason:
        judge_reason_match = re.search(r'"judge_reason"\s*:\s*"([^"]*)"', response, flags=re.IGNORECASE)
        if judge_reason_match:
            judge_reason = judge_reason_match.group(1)
    return JudgeResult(rubric_score=rubric_score, criterion_results=normalized_results, judge_reason=judge_reason)


def majority_vote_judge_results(results: list[JudgeResult], rubric: list[dict[str, Any]]) -> JudgeResult:
    if len(results) == 1:
        return results[0]

    aggregated_results: list[dict[str, Any]] = []
    for idx, criterion in enumerate(rubric):
        bool_votes = [bool(result.criterion_results[idx]["satisfied"]) for result in results if len(result.criterion_results) > idx]
        satisfied = Counter(bool_votes).most_common(1)[0][0] if bool_votes else False
        aggregated_results.append(
            {
                "criterion": criterion.get("criterion", ""),
                "satisfied": satisfied,
                "weight": float(criterion.get("weight", 1)),
                "evidence": "",
            }
        )

    total_weight = sum(float(item.get("weight", 1.0)) for item in rubric) or 1.0
    satisfied_weight = sum(item["weight"] for item in aggregated_results if item["satisfied"])
    rubric_score = satisfied_weight / total_weight

    reasons = [result.judge_reason for result in results if result.judge_reason]
    judge_reason = reasons[0] if reasons else ""
    return JudgeResult(rubric_score=rubric_score, criterion_results=aggregated_results, judge_reason=judge_reason)


def resolve_predicted_items(
    prediction: str,
    sidecar_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    sid_blocks = extract_sid_blocks(extract_prediction_tail(prediction))
    if not sid_blocks:
        return [], 1.0

    predicted_items: list[dict[str, Any]] = []
    unresolved = 0
    for sid in sid_blocks:
        record = sidecar_lookup.get(sid)
        if record is None:
            unresolved += 1
            predicted_items.append({"sid": sid, "caption": "", "pid": None})
            continue
        predicted_items.append(
            {
                "sid": sid,
                "caption": record.get("caption", ""),
                "pid": record.get("pid"),
            }
        )

    unresolved_ratio = unresolved / len(sid_blocks)
    return predicted_items, unresolved_ratio


def compute_objective_metrics(prediction: str, ground_truth: str) -> dict[str, float]:
    format_reward_value = think_format_reward(prediction)
    partial_hit_reward_value = partial_hit_reward(prediction, ground_truth)
    hit_reward_value = hit_reward(prediction, ground_truth)
    pass_rate_value = pass_rate(prediction, ground_truth)
    pass_at_1_value = first_sid_hit_reward(prediction, ground_truth)
    objective_anchor = (
        0.60 * pass_at_1_value
        + 0.20 * pass_rate_value
        + 0.15 * (partial_hit_reward_value / 100.0)
        + 0.05 * format_reward_value
    )
    return {
        "format_reward": format_reward_value,
        "partial_hit_reward": partial_hit_reward_value,
        "hit_reward": hit_reward_value,
        "pass_rate": pass_rate_value,
        "pass_at_1": pass_at_1_value,
        "objective_anchor": objective_anchor,
    }


def _judge_generate(judge_impl: Any, prompt: str) -> str:
    if hasattr(judge_impl, "generate"):
        return judge_impl.generate(prompt)
    if callable(judge_impl):
        return judge_impl(prompt)
    raise TypeError("judge_impl must be callable or expose a `generate` method")


def _cache_key(schema_id: str, extra_info: dict[str, Any], normalized_prediction: str) -> str:
    prompt_hash_payload = json.dumps(
        {
            "schema_id": schema_id,
            "prompt_text": extra_info.get("prompt_text", ""),
            "query_text": extra_info.get("query_text", ""),
            "interaction_type": extra_info.get("interaction_type", ""),
            "history_item_captions": extra_info.get("history_item_captions", []),
            "history_ad_captions": extra_info.get("history_ad_captions", []),
            "history_product_captions": extra_info.get("history_product_captions", []),
            "ground_truth_captions": extra_info.get("ground_truth_captions", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt_hash = sha256(prompt_hash_payload.encode("utf-8")).hexdigest()
    return f"{schema_id}:{prompt_hash}:{sha256(normalized_prediction.encode('utf-8')).hexdigest()}"


def _compute_rubric_score_for_sample(
    prediction: str,
    extra_info: dict[str, Any],
    sidecar_lookup: dict[str, dict[str, Any]],
    rubric_dir: str | None,
    judge_base_url: str | None,
    judge_model: str | None,
    cache_path: str | None,
    timeout_s: int,
    consensus_n: int,
    judge_backend: str,
    judge_max_new_tokens: int,
    judge_device_map: str,
    judge_torch_dtype: str,
    judge_attn_implementation: str | None,
    judge_impl: Any,
) -> dict[str, Any]:
    schema_id = str(extra_info.get("schema_id") or extra_info.get("task_name") or "unknown")
    normalized_prediction = normalize_prediction(prediction)
    predicted_items, unresolved_sid_ratio = resolve_predicted_items(prediction, sidecar_lookup)
    rubric = load_rubric(schema_id, rubric_dir)

    cache_key = _cache_key(schema_id=schema_id, extra_info=extra_info, normalized_prediction=normalized_prediction)
    cache = get_sqlite_cache(cache_path)
    if cache is not None:
        cached_value = cache.get(cache_key)
        if isinstance(cached_value, dict):
            cached_value["cache_hit"] = 1.0
            return cached_value

    if judge_impl is None:
        judge_impl = get_judge_client(
            judge_base_url,
            judge_model,
            timeout_s,
            judge_backend=judge_backend,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_device_map=judge_device_map,
            judge_torch_dtype=judge_torch_dtype,
            judge_attn_implementation=judge_attn_implementation,
        )

    if judge_impl is None or not predicted_items:
        fallback = {
            "rubric_score": 0.0,
            "criterion_results": [],
            "judge_reason": "judge unavailable or no semantic candidates",
            "unresolved_sid_ratio": unresolved_sid_ratio,
            "cache_hit": 0.0,
            "rubric_applied": 0.0,
        }
        if cache is not None:
            cache.set(cache_key, fallback)
        return fallback

    prompt = build_judge_prompt(
        schema_id=schema_id,
        rubric=rubric,
        extra_info=extra_info,
        predicted_items=predicted_items,
        raw_prediction=prediction,
    )

    results: list[JudgeResult] = []
    for _ in range(max(consensus_n, 1)):
        raw_response = _judge_generate(judge_impl, prompt)
        results.append(parse_judge_response(raw_response, rubric))
    result = majority_vote_judge_results(results, rubric)

    rubric_payload = {
        "rubric_score": result.rubric_score,
        "criterion_results": result.criterion_results,
        "judge_reason": result.judge_reason,
        "unresolved_sid_ratio": unresolved_sid_ratio,
        "cache_hit": 0.0,
        "rubric_applied": 1.0,
    }
    if cache is not None:
        cache.set(cache_key, rubric_payload)
    return rubric_payload


def compute_single_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None,
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_index_path: str | None = None,
    cache_path: str | None = None,
    timeout_s: int = 30,
    consensus_n: int = 1,
    reward_mode: str = "objective_rubric",
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 768,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
) -> dict[str, Any]:
    del data_source
    extra_info = parse_json_like(extra_info, default={})
    sidecar_lookup = load_sidecar_index(sidecar_index_path)
    objective_metrics = compute_objective_metrics(solution_str, ground_truth)
    reward_mode = str(reward_mode or "objective_rubric").strip().lower()

    if reward_mode == "objective_only":
        rubric_payload = {
            "rubric_score": 0.0,
            "criterion_results": [],
            "judge_reason": "rubric disabled by reward_mode=objective_only",
            "unresolved_sid_ratio": 0.0,
            "cache_hit": 0.0,
            "rubric_applied": 0.0,
        }
        final_score = objective_metrics["objective_anchor"]
    else:
        rubric_payload = _compute_rubric_score_for_sample(
            prediction=solution_str,
            extra_info=extra_info,
            sidecar_lookup=sidecar_lookup,
            rubric_dir=rubric_dir,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            cache_path=cache_path,
            timeout_s=timeout_s,
            consensus_n=consensus_n,
            judge_backend=judge_backend,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_device_map=judge_device_map,
            judge_torch_dtype=judge_torch_dtype,
            judge_attn_implementation=judge_attn_implementation,
            judge_impl=judge_impl,
        )
        if rubric_payload.get("rubric_applied", 0.0) >= 1.0:
            final_score = 0.50 * objective_metrics["objective_anchor"] + 0.50 * rubric_payload["rubric_score"]
        else:
            final_score = objective_metrics["objective_anchor"]

    return {
        "score": final_score,
        **objective_metrics,
        "rubric_score": rubric_payload["rubric_score"],
        "judge_reason": rubric_payload["judge_reason"],
        "unresolved_sid_ratio": rubric_payload["unresolved_sid_ratio"],
        "cache_hit": rubric_payload["cache_hit"],
        "rubric_applied": rubric_payload.get("rubric_applied", 0.0),
        "pred": normalize_prediction(solution_str),
    }


def compute_rubric_only_score(
    solution_str: str,
    extra_info: dict[str, Any] | None,
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_index_path: str | None = None,
    cache_path: str | None = None,
    timeout_s: int = 30,
    consensus_n: int = 1,
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 768,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
) -> dict[str, Any]:
    extra_info = parse_json_like(extra_info, default={})
    sidecar_lookup = load_sidecar_index(sidecar_index_path)
    rubric_payload = _compute_rubric_score_for_sample(
        prediction=solution_str,
        extra_info=extra_info,
        sidecar_lookup=sidecar_lookup,
        rubric_dir=rubric_dir,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        cache_path=cache_path,
        timeout_s=timeout_s,
        consensus_n=consensus_n,
        judge_backend=judge_backend,
        judge_max_new_tokens=judge_max_new_tokens,
        judge_device_map=judge_device_map,
        judge_torch_dtype=judge_torch_dtype,
        judge_attn_implementation=judge_attn_implementation,
        judge_impl=judge_impl,
    )
    return {
        "rubric_score": rubric_payload["rubric_score"],
        "criterion_results": rubric_payload["criterion_results"],
        "judge_reason": rubric_payload["judge_reason"],
        "unresolved_sid_ratio": rubric_payload["unresolved_sid_ratio"],
        "cache_hit": rubric_payload["cache_hit"],
        "rubric_applied": rubric_payload.get("rubric_applied", 0.0),
        "pred": normalize_prediction(solution_str),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    return compute_single_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score_batch(
    data_sources: Iterable[str],
    solution_strs: Iterable[str],
    ground_truths: Iterable[str],
    extra_infos: Iterable[dict[str, Any] | str | None],
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_index_path: str | None = None,
    cache_path: str | None = None,
    max_workers: int = 16,
    timeout_s: int = 30,
    consensus_n: int = 1,
    reward_mode: str = "objective_rubric",
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 768,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
) -> list[dict[str, Any]]:
    items = list(
        zip(
            list(data_sources),
            list(solution_strs),
            list(ground_truths),
            list(extra_infos),
            strict=True,
        )
    )

    def _run(item: tuple[str, str, str, dict[str, Any] | str | None]) -> dict[str, Any]:
        data_source, solution_str, ground_truth, extra_info = item
        return compute_single_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            rubric_dir=rubric_dir,
            sidecar_index_path=sidecar_index_path,
            cache_path=cache_path,
            timeout_s=timeout_s,
            consensus_n=consensus_n,
            reward_mode=reward_mode,
            judge_backend=judge_backend,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_device_map=judge_device_map,
            judge_torch_dtype=judge_torch_dtype,
            judge_attn_implementation=judge_attn_implementation,
            judge_impl=judge_impl,
        )

    if max_workers <= 1 or len(items) <= 1:
        return [_run(item) for item in items]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, items))
