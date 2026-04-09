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
from itertools import combinations
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

DEFAULT_USER_GOALS: dict[str, str] = {
    "video": "为该用户重排候选视频，优先更符合近期观看兴趣的内容。",
    "ad": "为该用户重排候选广告，优先更符合近期兴趣和点击倾向的内容。",
    "product": "为该用户重排候选商品，优先更符合近期兴趣和潜在购买意图的内容。",
    "unknown": "为该用户重排候选内容，优先更符合近期兴趣的内容。",
}

GENERIC_AUDIT_PHRASES = {
    "更贴近主题",
    "更相关",
    "证据不足",
    "更贴近历史兴趣",
    "按该规则更优",
}


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


@dataclass
class ListwiseJudgeResult:
    ranking: list[dict[str, Any]]
    judge_reason: str


@dataclass
class PairwiseJudgeResult:
    winner: str
    confidence: float
    judge_reason: str
    criterion_votes: list[dict[str, Any]]


@dataclass
class HistorySummaryResult:
    dominant_topics: list[str]
    recent_patterns: list[str]
    supporting_history_ids: list[int]
    history_evidence_snippets: list[dict[str, Any]]
    nonempty: bool


@dataclass
class SingleRuleJudgeResult:
    rule_id: str
    vote: str
    history_basis: str
    candidate_a_basis: str
    candidate_b_basis: str
    tie_analysis: str
    decision_rationale: str
    parse_ok: bool


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


def resolve_rubric(
    schema_id: str,
    rubric_dir: str | None,
    extra_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    adaptive_rules = parse_json_like((extra_info or {}).get("adaptive_rules"), default=[])
    if isinstance(adaptive_rules, list) and adaptive_rules:
        return adaptive_rules
    return load_rubric(schema_id, rubric_dir)


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


def _build_prompt_context(extra_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": extra_info.get("task_name", ""),
        "task_variant": extra_info.get("task_variant", ""),
        "query_text": extra_info.get("query_text", ""),
        "interaction_type": extra_info.get("interaction_type", ""),
        "user_profile_text": extra_info.get("user_profile_text", ""),
        "history_item_captions": extra_info.get("history_item_captions", []),
        "history_ad_captions": extra_info.get("history_ad_captions", []),
        "history_product_captions": extra_info.get("history_product_captions", []),
    }


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def _compact_text(text: Any, max_chars: int = 64) -> str:
    if text is None:
        return ""
    normalized = str(text).strip()
    if not normalized:
        return ""
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


def _top_overlap_matches(candidate_text: str, history_texts: list[str], *, top_n: int = 2) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for history_text in history_texts:
        if not history_text:
            continue
        overlap_score = _token_overlap_score(candidate_text, history_text)
        if overlap_score <= 0.0:
            continue
        scored.append(
            {
                "text": _compact_text(history_text, 56),
                "overlap": round(overlap_score, 4),
            }
        )
    scored.sort(key=lambda item: (-float(item["overlap"]), item["text"]))
    return scored[:top_n]


def _compress_history_texts(history_texts: list[str], *, limit: int = 8, max_chars: int = 56) -> list[str]:
    compressed: list[str] = []
    seen: set[str] = set()
    for history_text in history_texts:
        compact = _compact_text(history_text, max_chars)
        if not compact or compact in seen:
            continue
        compressed.append(compact)
        seen.add(compact)
        if len(compressed) >= limit:
            break
    return compressed


def _primary_candidate_caption(candidate: dict[str, Any]) -> str:
    for item in candidate.get("predicted_items", []):
        caption = _compact_text(item.get("caption", ""), 72)
        if caption:
            return caption
    return ""


def _extract_preference_summary(query_text: Any) -> str:
    text = _compact_text(query_text, 160)
    if not text:
        return ""
    summary_patterns = [
        r"(用户当前的偏好涉及[^。！？]*[。！？]?)",
        r"(当前偏好涉及[^。！？]*[。！？]?)",
        r"(偏好涉及[^。！？]*[。！？]?)",
    ]
    for pattern in summary_patterns:
        match = re.search(pattern, text)
        if match:
            return _compact_text(match.group(1), 96)
    return ""


def _resolve_preference_summary(extra_info: dict[str, Any]) -> str:
    direct_summary = _compact_text(extra_info.get("preference_summary_text", ""), 96)
    if direct_summary:
        return direct_summary
    return _extract_preference_summary(extra_info.get("query_text", ""))


def _normalize_vote(raw_vote: Any) -> str:
    mapping = {
        "a": "A",
        "candidate_a": "A",
        "left": "A",
        "1": "A",
        "b": "B",
        "candidate_b": "B",
        "right": "B",
        "2": "B",
        "tie": "tie",
        "equal": "tie",
        "same": "tie",
    }
    return mapping.get(str(raw_vote or "").strip().lower(), "tie")


def _user_goal_for_schema(schema_id: str) -> str:
    base_schema = str(schema_id or "unknown").split("::")[0]
    return DEFAULT_USER_GOALS.get(base_schema) or DEFAULT_USER_GOALS["unknown"]


def _nonempty_unique_texts(values: Iterable[Any], *, limit: int, max_chars: int) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        compact = _compact_text(normalized, max_chars)
        if compact in seen:
            continue
        seen.add(compact)
        texts.append(compact)
        if len(texts) >= limit:
            break
    return texts


def _build_history_records(extra_info: dict[str, Any], *, limit: int = 20, max_chars: int = 180) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    history_texts = list(extra_info.get("history_item_captions", []) or [])
    for history_text in history_texts:
        normalized = str(history_text or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append(
            {
                "history_id": len(records) + 1,
                "text": _compact_text(normalized, max_chars),
            }
        )
        if len(records) >= limit:
            break
    return records


def build_history_summary_prompt(
    schema_id: str,
    extra_info: dict[str, Any],
    history_records: list[dict[str, Any]],
) -> str:
    history_json = json.dumps(history_records, ensure_ascii=False, indent=2)
    return f"""你是一名推荐历史摘要器，需要把用户完整观看历史整理成可供重排判断使用的结构化摘要。

任务：
1. 只根据提供的历史片段总结，不要编造不存在的兴趣。
2. `dominant_topics` 表示最近主要兴趣主题，最多 3 条，每条 6 到 18 个字。
3. `recent_patterns` 表示最近行为模式或偏好倾向，最多 2 条，每条 8 到 24 个字。
4. `supporting_history_ids` 必须引用下方历史片段里的 `history_id`。
5. 如果历史本身很杂，也要如实说明，不要强行总结成单一主题。
6. 返回纯 JSON，不要附加说明。

schema_id:
{schema_id}

user_goal:
{_user_goal_for_schema(schema_id)}

history_records:
{history_json}

返回格式：
{{
  "dominant_topics": ["..."],
  "recent_patterns": ["..."],
  "supporting_history_ids": [1, 3, 5]
}}
"""


def parse_history_summary_response(
    raw_response: str,
    history_records: list[dict[str, Any]],
) -> HistorySummaryResult:
    parsed = parse_json_like(raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```"), default={})
    history_id_set = {int(item["history_id"]) for item in history_records}
    dominant_topics = _nonempty_unique_texts(parsed.get("dominant_topics", []) if isinstance(parsed, dict) else [], limit=3, max_chars=24)
    recent_patterns = _nonempty_unique_texts(parsed.get("recent_patterns", []) if isinstance(parsed, dict) else [], limit=2, max_chars=32)
    supporting_history_ids: list[int] = []
    raw_ids = parsed.get("supporting_history_ids", []) if isinstance(parsed, dict) else []
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                history_id = int(value)
            except (TypeError, ValueError):
                continue
            if history_id in history_id_set and history_id not in supporting_history_ids:
                supporting_history_ids.append(history_id)
    snippet_lookup = {int(item["history_id"]): item for item in history_records}
    history_evidence_snippets = [snippet_lookup[item_id] for item_id in supporting_history_ids if item_id in snippet_lookup]
    if not history_evidence_snippets:
        history_evidence_snippets = history_records[: min(len(history_records), 4)]
        supporting_history_ids = [int(item["history_id"]) for item in history_evidence_snippets]
    return HistorySummaryResult(
        dominant_topics=dominant_topics,
        recent_patterns=recent_patterns,
        supporting_history_ids=supporting_history_ids,
        history_evidence_snippets=history_evidence_snippets,
        nonempty=bool(dominant_topics or recent_patterns),
    )


def _history_summary_cache_key(schema_id: str, extra_info: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "schema_id": schema_id,
            "history_item_captions": extra_info.get("history_item_captions", []),
            "user_profile_text": extra_info.get("user_profile_text", ""),
            "prompt_text": extra_info.get("prompt_text", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"history_summary:{sha256(payload.encode('utf-8')).hexdigest()}"


def _build_history_summary_payload(
    schema_id: str,
    extra_info: dict[str, Any],
    *,
    cache_path: str | None,
    judge_impl: Any,
) -> dict[str, Any]:
    history_records = _build_history_records(extra_info)
    user_goal = _user_goal_for_schema(schema_id)
    if not history_records or judge_impl is None:
        return {
            "user_goal": user_goal,
            "history_summary": {
                "dominant_topics": [],
                "recent_patterns": [],
                "supporting_history_ids": [],
            },
            "history_evidence_snippets": history_records[:4],
            "history_summary_nonempty": False,
        }

    cache = get_sqlite_cache(cache_path)
    cache_key = _history_summary_cache_key(schema_id, extra_info)
    cached_value = cache.get(cache_key) if cache is not None else None
    if isinstance(cached_value, dict):
        history_summary = dict(cached_value.get("history_summary", {}))
        history_evidence_snippets = list(cached_value.get("history_evidence_snippets", []))
        return {
            "user_goal": user_goal,
            "history_summary": history_summary,
            "history_evidence_snippets": history_evidence_snippets,
            "history_summary_nonempty": bool(cached_value.get("history_summary_nonempty", False)),
        }

    prompt = build_history_summary_prompt(schema_id, extra_info, history_records)
    raw_response = _judge_generate(judge_impl, prompt)
    parsed = parse_history_summary_response(raw_response, history_records)
    payload = {
        "user_goal": user_goal,
        "history_summary": {
            "dominant_topics": parsed.dominant_topics,
            "recent_patterns": parsed.recent_patterns,
            "supporting_history_ids": parsed.supporting_history_ids,
        },
        "history_evidence_snippets": parsed.history_evidence_snippets,
        "history_summary_nonempty": parsed.nonempty,
    }
    if cache is not None:
        cache.set(
            cache_key,
            {
                "history_summary": payload["history_summary"],
                "history_evidence_snippets": payload["history_evidence_snippets"],
                "history_summary_nonempty": payload["history_summary_nonempty"],
            },
        )
    return payload


def _is_vague_audit_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if len(normalized) < 6:
        return True
    return normalized in GENERIC_AUDIT_PHRASES


def _mentions_hidden_identifier(text: str) -> bool:
    normalized = str(text or "").lower()
    banned_patterns = (
        "候选id",
        "candidate_id",
        "id推测",
        "根据id",
        "候选编号",
        "编号推测",
        "索引推测",
    )
    return any(pattern in normalized for pattern in banned_patterns)


def _build_single_rule_shared_criteria(
    schema_id: str,
    rubric: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_schema = str(schema_id or "unknown").split("::")[0]
    if base_schema != "video":
        return _build_pairwise_criteria(schema_id, rubric, {})

    shared_rules: list[dict[str, Any]] = []
    for item in rubric:
        rule_id = str(item.get("rule_id", "")).strip()
        if not rule_id:
            continue
        if rule_id == "history_topic_match":
            evidence_source = ["user_goal", "history_summary", "history_evidence_snippets", "candidate_caption"]
        elif rule_id == "query_intent_match":
            evidence_source = ["user_goal", "candidate_caption"]
        elif rule_id == "novelty_without_drift":
            evidence_source = ["user_goal", "history_summary", "history_evidence_snippets", "candidate_caption"]
        elif rule_id == "duplicate_penalty":
            evidence_source = ["candidate_caption", "candidate_pool_reference_captions"]
        else:
            evidence_source = ["user_goal", "history_summary", "history_evidence_snippets", "candidate_caption"]
        shared_rules.append(
            {
                "criterion_id": rule_id,
                "criterion": str(item.get("rule_text", "")).strip(),
                "weight": float(item.get("weight", 1)),
                "evidence_source": evidence_source,
                "decision_rule": str(item.get("decision_rule", "")).strip(),
                "tie_condition": str(item.get("tie_condition", "")).strip(),
            }
        )
    return shared_rules


def _candidate_caption_for_prompt(candidate: dict[str, Any], *, max_chars: int = 180) -> str:
    return _compact_text(_primary_candidate_caption(candidate), max_chars)


def _candidate_pool_reference_captions(
    candidate_states: list[dict[str, Any]],
    *,
    exclude_indices: set[int],
    limit: int = 4,
) -> list[str]:
    captions: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_states:
        if int(candidate.get("candidate_index", 0)) in exclude_indices:
            continue
        caption = _candidate_caption_for_prompt(candidate)
        if not caption or caption in seen:
            continue
        seen.add(caption)
        captions.append(caption)
        if len(captions) >= limit:
            break
    return captions


def build_single_rule_judge_prompt(
    schema_id: str,
    rule: dict[str, Any],
    *,
    history_payload: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    candidate_pool_captions: list[str],
    include_candidate_id: bool = False,
) -> str:
    evidence_source = list(rule.get("evidence_source", []))
    evidence_payload: dict[str, Any] = {}
    if "user_goal" in evidence_source:
        evidence_payload["user_goal"] = history_payload.get("user_goal", "")
    if "history_summary" in evidence_source:
        evidence_payload["history_summary"] = history_payload.get("history_summary", {})
    if "history_evidence_snippets" in evidence_source:
        evidence_payload["history_evidence_snippets"] = history_payload.get("history_evidence_snippets", [])
    if "candidate_pool_reference_captions" in evidence_source:
        evidence_payload["candidate_pool_reference_captions"] = candidate_pool_captions

    candidate_payload = {
        "candidate_a": {
            **({"candidate_id": int(candidate_a.get("candidate_index", 0))} if include_candidate_id else {}),
            "candidate_caption": _candidate_caption_for_prompt(candidate_a) or "无可用描述",
        },
        "candidate_b": {
            **({"candidate_id": int(candidate_b.get("candidate_index", 0))} if include_candidate_id else {}),
            "candidate_caption": _candidate_caption_for_prompt(candidate_b) or "无可用描述",
        },
    }
    evidence_json = json.dumps(evidence_payload, ensure_ascii=False, indent=2)
    candidate_json = json.dumps(candidate_payload, ensure_ascii=False, indent=2)
    return f"""你是一名推荐结果重排序评审器。当前只执行一条规则，只能依据给定证据判断候选 A 和候选 B。

要求：
1. 只能使用 `allowed_evidence` 中提供的字段，禁止引用未提供的信息。
2. `vote` 只能是 "A"、"B" 或 "tie"。
3. 必须分别写出：
   - `history_basis`
   - `candidate_a_basis`
   - `candidate_b_basis`
   - `tie_analysis`
   - `decision_rationale`
4. 不允许只写“更贴近主题”“证据不足”这种空话，必须明确说明当前看到的证据。
5. 如果按该规则无法明确区分 A 和 B，必须返回 `tie`。
6. 返回纯 JSON，不要附加说明。
7. {"不允许根据候选编号、候选ID、隐藏索引或任何未提供的系统信息进行推断。" if not include_candidate_id else "候选编号仅作为区分 A/B 的辅助标识，不能单独作为判断语义的依据。"}
8. 如果 `candidate_caption` 是“无可用描述”，必须把它视为缺少内容证据，{"不能从编号或顺序猜测语义。" if not include_candidate_id else "不能仅根据编号或顺序猜测语义。"}

schema_id:
{schema_id}

rule_id:
{rule.get("criterion_id", "")}

rule_text:
{rule.get("criterion", "")}

decision_rule:
{rule.get("decision_rule", "")}

tie_condition:
{rule.get("tie_condition", "")}

allowed_evidence:
{json.dumps(evidence_source, ensure_ascii=False)}

evidence:
{evidence_json}

candidates:
{candidate_json}

返回格式：
{{
  "rule_id": "{rule.get('criterion_id', '')}",
  "vote": "A",
  "history_basis": "...",
  "candidate_a_basis": "...",
  "candidate_b_basis": "...",
  "tie_analysis": "...",
  "decision_rationale": "..."
}}
"""


def parse_single_rule_judge_response(
    raw_response: str,
    rule_id: str,
    *,
    allow_hidden_identifier_mentions: bool = False,
) -> SingleRuleJudgeResult:
    response = raw_response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    parsed = parse_json_like(response, default={})
    if not isinstance(parsed, dict):
        parsed = {}

    resolved_rule_id = str(parsed.get("rule_id", rule_id)).strip() or rule_id
    vote = _normalize_vote(parsed.get("vote", parsed.get("winner", "tie")))
    history_basis = str(parsed.get("history_basis", "")).strip()
    candidate_a_basis = str(parsed.get("candidate_a_basis", "")).strip()
    candidate_b_basis = str(parsed.get("candidate_b_basis", "")).strip()
    tie_analysis = str(parsed.get("tie_analysis", "")).strip()
    decision_rationale = str(parsed.get("decision_rationale", "")).strip()
    parse_ok = (
        resolved_rule_id == rule_id
        and not _is_vague_audit_text(history_basis)
        and not _is_vague_audit_text(candidate_a_basis)
        and not _is_vague_audit_text(candidate_b_basis)
        and not _is_vague_audit_text(tie_analysis)
        and not _is_vague_audit_text(decision_rationale)
        and (allow_hidden_identifier_mentions or not _mentions_hidden_identifier(history_basis))
        and (allow_hidden_identifier_mentions or not _mentions_hidden_identifier(candidate_a_basis))
        and (allow_hidden_identifier_mentions or not _mentions_hidden_identifier(candidate_b_basis))
        and (allow_hidden_identifier_mentions or not _mentions_hidden_identifier(tie_analysis))
        and (allow_hidden_identifier_mentions or not _mentions_hidden_identifier(decision_rationale))
    )
    return SingleRuleJudgeResult(
        rule_id=resolved_rule_id,
        vote=vote,
        history_basis=history_basis,
        candidate_a_basis=candidate_a_basis,
        candidate_b_basis=candidate_b_basis,
        tie_analysis=tie_analysis,
        decision_rationale=decision_rationale,
        parse_ok=parse_ok,
    )


def _single_rule_cache_key(
    schema_id: str,
    extra_info: dict[str, Any],
    *,
    rule_id: str,
    left_prediction: str,
    right_prediction: str,
    include_candidate_id_prompt: bool,
) -> str:
    prompt_hash_payload = json.dumps(
        {
            "schema_id": schema_id,
            "rule_id": rule_id,
            "prompt_text": extra_info.get("prompt_text", ""),
            "history_item_captions": extra_info.get("history_item_captions", []),
            "user_profile_text": extra_info.get("user_profile_text", ""),
            "include_candidate_id_prompt": bool(include_candidate_id_prompt),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    pair_hash_payload = json.dumps([left_prediction, right_prediction], ensure_ascii=False, sort_keys=False)
    prompt_hash = sha256(prompt_hash_payload.encode("utf-8")).hexdigest()
    pair_hash = sha256(pair_hash_payload.encode("utf-8")).hexdigest()
    return f"single_rule:{schema_id}:{rule_id}:{prompt_hash}:{pair_hash}"


def _mirror_vote(vote: str) -> str:
    if vote == "A":
        return "B"
    if vote == "B":
        return "A"
    return "tie"


def _build_pairwise_criteria(
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
) -> list[dict[str, Any]]:
    if rubric and all(isinstance(item, dict) and str(item.get("rule_id", "")).strip() for item in rubric):
        typed_rules: list[dict[str, Any]] = []
        for item in rubric:
            typed_rules.append(
                {
                    "criterion_id": str(item.get("rule_id", "")).strip(),
                    "criterion": str(item.get("rule_text", "")).strip(),
                    "weight": float(item.get("weight", 1)),
                    "evidence_source": list(item.get("evidence_source", [])),
                    "decision_rule": str(item.get("decision_rule", "")).strip(),
                    "tie_condition": str(item.get("tie_condition", "")).strip(),
                }
            )
        return typed_rules

    base_schema = schema_id.split("::")[0]
    if base_schema == "product":
        preference_summary = _resolve_preference_summary(extra_info)
        criteria: list[dict[str, Any]] = []
        if preference_summary:
            criteria.append(
                {
                    "criterion_id": "query_summary_match",
                    "criterion": "优先选择与用户当前偏好摘要更贴近的候选。",
                    "weight": 4,
                    "evidence_source": ["context.preference_summary", "candidate_features.preference_summary_overlap"],
                    "decision_rule": "若 query 中存在“用户当前的偏好涉及...”这类摘要，应优先比较候选与该摘要的贴近程度。",
                }
            )
        criteria.extend(
            [
                {
                    "criterion_id": "history_product_match",
                    "criterion": "优先选择与用户历史商品兴趣更贴近的候选。",
                    "weight": 3,
                    "evidence_source": ["history_product_captions", "candidate_features.product_overlap"],
                    "decision_rule": "比较候选与历史商品标题的贴近程度，优先选择更像用户过往感兴趣商品的候选。",
                },
                {
                    "criterion_id": "history_video_bridge",
                    "criterion": "仅在商品证据不足时，参考用户最近观看主题。",
                    "weight": 1,
                    "evidence_source": ["history_item_captions", "candidate_features.video_overlap"],
                    "decision_rule": "视频主题只作弱补充，不能单独压过更贴近商品摘要或商品历史的候选。",
                },
                {
                    "criterion_id": "semantic_quality",
                    "criterion": "候选缺少 caption 时，应倾向于返回 tie，而不是凭空脑补。",
                    "weight": 2,
                    "evidence_source": ["candidate_features.primary_caption", "candidate_features.unresolved_sid_ratio"],
                    "decision_rule": "若某个候选缺少 caption 或语义不足，除非其他证据明显更强，否则返回 tie。",
                },
                {
                    "criterion_id": "raw_rank_anchor",
                    "criterion": "若上述证据接近，则保留原始排序更靠前的候选。",
                    "weight": 2,
                    "evidence_source": ["candidate_features.raw_rank"],
                    "decision_rule": "只有在 query 摘要、商品历史、视频桥接都无法明显区分时才用作 tie-break。",
                },
            ]
        )
        return criteria

    typed_criteria: list[dict[str, Any]] = []
    for index, item in enumerate(rubric, start=1):
        typed_criteria.append(
            {
                "criterion_id": str(item.get("criterion_id") or f"criterion_{index}"),
                "criterion": str(item.get("criterion", "")),
                "weight": float(item.get("weight", 1)),
                "evidence_source": ["context", "candidate_features"],
                "decision_rule": str(item.get("explanation", "")),
            }
        )
    return typed_criteria


def _build_pairwise_candidate_features(candidate: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
    primary_caption = _primary_candidate_caption(candidate)
    history_product_captions = list(extra_info.get("history_product_captions", []) or [])
    history_item_captions = list(extra_info.get("history_item_captions", []) or [])
    query_text = str(extra_info.get("query_text", "")).strip()
    preference_summary = _resolve_preference_summary(extra_info)

    product_matches = _top_overlap_matches(primary_caption, history_product_captions, top_n=2)
    video_matches = _top_overlap_matches(primary_caption, history_item_captions, top_n=2)

    return {
        "raw_rank": int(candidate.get("raw_rank", 0)),
        "primary_caption": primary_caption,
        "semantic_available": bool(primary_caption),
        "unresolved_sid_ratio": round(float(candidate.get("unresolved_sid_ratio", 1.0)), 4),
        "history_product_overlap": round(max((item["overlap"] for item in product_matches), default=0.0), 4),
        "history_video_overlap": round(max((item["overlap"] for item in video_matches), default=0.0), 4),
        "query_overlap": round(_token_overlap_score(primary_caption, query_text), 4),
        "preference_summary": preference_summary,
        "preference_summary_overlap": round(_token_overlap_score(primary_caption, preference_summary), 4),
        "top_product_matches": product_matches,
        "top_video_matches": video_matches,
    }


def build_judge_prompt(
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
    predicted_items: list[dict[str, Any]],
    raw_prediction: str,
) -> str:
    context = _build_prompt_context(extra_info)
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


def build_listwise_judge_prompt(
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    context_json = json.dumps(_build_prompt_context(extra_info), ensure_ascii=False, indent=2)
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)

    candidate_payload: list[dict[str, Any]] = []
    for candidate in candidates:
        semantic_items = []
        for item in candidate.get("predicted_items", []):
            semantic_items.append(
                {
                    "pid": item.get("pid"),
                    "caption": str(item.get("caption", "")).strip(),
                    "semantic_available": bool(str(item.get("caption", "")).strip()),
                }
            )
        candidate_payload.append(
            {
                "candidate_index": int(candidate.get("candidate_index", 0)),
                "raw_rank": int(candidate.get("raw_rank", 0)),
                "semantic_items": semantic_items,
                "unresolved_sid_ratio": float(candidate.get("unresolved_sid_ratio", 1.0)),
                "raw_model_response": str(candidate.get("raw_prediction", "")),
            }
        )

    candidates_json = json.dumps(candidate_payload, ensure_ascii=False, indent=2)
    return f"""你是一名推荐结果重排序评审器，需要根据给定 rubric 对整组候选推荐结果联合排序。

规则：
1. 必须联合比较整个候选池，而不是独立点评单个候选。
2. 只能依据给定上下文和候选 item 的语义描述判断，不要依赖 SID 字符串本身。
3. 如果某个候选缺少 caption 或语义不足，降低其置信度，不要自行脑补语义。
4. `ranking` 必须覆盖全部候选，`candidate_index` 不能重复。
5. `score` 范围必须是 [0, 1]，分数越高代表越应该排在前面。
6. `reason` 和 `judge_reason` 必须尽量短，每条不超过 20 个字。
7. 返回纯 JSON，不要附加说明。

schema_id:
{schema_id}

rubric:
{rubric_json}

context:
{context_json}

candidates:
{candidates_json}

返回格式：
{{
  "ranking": [
    {{
      "candidate_index": 1,
      "score": 0.82,
      "reason": "..."
    }}
  ],
  "judge_reason": "..."
}}
"""


def build_pairwise_judge_prompt(
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> str:
    preference_summary = _resolve_preference_summary(extra_info)
    typed_criteria = _build_pairwise_criteria(schema_id, rubric, extra_info)
    rules_json = json.dumps(typed_criteria, ensure_ascii=False, indent=2)

    compact_context = {
        **_build_prompt_context(extra_info),
        "history_item_captions": _compress_history_texts(extra_info.get("history_item_captions", []), limit=6, max_chars=48),
        "history_ad_captions": _compress_history_texts(extra_info.get("history_ad_captions", []), limit=4, max_chars=48),
        "history_product_captions": _compress_history_texts(extra_info.get("history_product_captions", []), limit=6, max_chars=48),
        "query_text": _compact_text(extra_info.get("query_text", ""), 96),
        "user_profile_text": _compact_text(extra_info.get("user_profile_text", ""), 64),
        "preference_summary": preference_summary,
    }
    context_json = json.dumps(compact_context, ensure_ascii=False, indent=2)

    def _candidate_payload(candidate: dict[str, Any], label: str) -> dict[str, Any]:
        semantic_items = []
        for item in candidate.get("predicted_items", [])[:2]:
            semantic_items.append(
                {
                    "pid": item.get("pid"),
                    "caption": _compact_text(item.get("caption", ""), 72),
                    "semantic_available": bool(str(item.get("caption", "")).strip()),
                }
            )
        return {
            "label": label,
            "candidate_index": int(candidate.get("candidate_index", 0)),
            "candidate_features": _build_pairwise_candidate_features(candidate, extra_info),
            "semantic_items": semantic_items,
        }

    candidate_json = json.dumps({"A": _candidate_payload(candidate_a, "A"), "B": _candidate_payload(candidate_b, "B")}, ensure_ascii=False, indent=2)

    return f"""你是一名推荐结果两两比较评审器，需要根据给定规则判断“候选 A”和“候选 B”谁更应该排在前面。

规则：
1. 只比较候选 A 和候选 B，返回更适合当前用户和当前任务的那个候选。
2. 必须逐条执行 `rules`，每条规则都返回一票 `A/B/tie`。
3. 对 product 任务，若 `context.preference_summary` 非空，应优先使用它和 `history_product_match` 作判断；`history_video_bridge` 只能作弱补充。
4. 若某个候选缺少 caption 或语义不足，不要因为另一候选“描述更完整”就直接判赢；除非 `query_summary_match` 或 `history_product_match` 明显更强，否则返回 tie。
5. `raw_rank_anchor` 只能在其他证据接近时使用，不能单独压过明显更相关的候选。
6. 只能依据给定上下文、候选 item 的语义描述和显式特征判断，不要依赖 SID 字符串本身。
7. `vote` 和 `final_vote` 只能是 "A"、"B" 或 "tie"。
8. 若某条规则证据不足，必须给该规则投 `tie`。
9. `evidence` 和 `short_reason` 必须尽量短，不超过 20 个字。
10. 返回纯 JSON，不要附加说明。

schema_id:
{schema_id}

rules:
{rules_json}

context:
{context_json}

candidates:
{candidate_json}

返回格式：
{{
  "rule_votes": [
    {{
      "rule_id": "history_topic_match",
      "vote": "A",
      "evidence": "A 更贴近最近观看主题"
    }}
  ],
  "final_vote": "A",
  "short_reason": "A 更贴近历史兴趣"
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


def parse_listwise_judge_response(raw_response: str, candidate_count: int) -> ListwiseJudgeResult:
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

    raw_ranking = parsed.get("ranking", [])
    if not isinstance(raw_ranking, list):
        raw_ranking = []

    normalized_ranking: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for entry in raw_ranking:
        if not isinstance(entry, dict):
            continue
        try:
            candidate_index = int(entry.get("candidate_index"))
        except (TypeError, ValueError):
            continue
        if candidate_index < 1 or candidate_index > candidate_count or candidate_index in seen_indices:
            continue
        score = entry.get("score")
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = max(0.0, 1.0 - ((len(normalized_ranking)) / max(candidate_count, 1)))
        normalized_ranking.append(
            {
                "candidate_index": candidate_index,
                "score": max(0.0, min(1.0, normalized_score)),
                "reason": str(entry.get("reason", "")),
            }
        )
        seen_indices.add(candidate_index)

    if not normalized_ranking:
        recovered_entries = re.findall(
            r'"candidate_index"\s*:\s*(\d+)(?:[^{}]*?"score"\s*:\s*([0-9]*\.?[0-9]+))?(?:[^{}]*?"reason"\s*:\s*"([^"]*)")?',
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for raw_index, raw_score, raw_reason in recovered_entries:
            candidate_index = int(raw_index)
            if candidate_index < 1 or candidate_index > candidate_count or candidate_index in seen_indices:
                continue
            try:
                normalized_score = float(raw_score)
            except (TypeError, ValueError):
                normalized_score = max(0.0, 1.0 - (len(normalized_ranking) / max(candidate_count, 1)))
            normalized_ranking.append(
                {
                    "candidate_index": candidate_index,
                    "score": max(0.0, min(1.0, normalized_score)),
                    "reason": str(raw_reason or ""),
                }
            )
            seen_indices.add(candidate_index)

    for candidate_index in range(1, candidate_count + 1):
        if candidate_index in seen_indices:
            continue
        normalized_ranking.append(
            {
                "candidate_index": candidate_index,
                "score": max(0.0, 1.0 - (len(normalized_ranking) / max(candidate_count, 1))),
                "reason": "",
            }
        )

    judge_reason = str(parsed.get("judge_reason", ""))
    if not judge_reason:
        judge_reason_match = re.search(r'"judge_reason"\s*:\s*"([^"]*)"', response, flags=re.IGNORECASE)
        if judge_reason_match:
            judge_reason = judge_reason_match.group(1)
    return ListwiseJudgeResult(ranking=normalized_ranking, judge_reason=judge_reason)


def parse_pairwise_judge_response(raw_response: str) -> PairwiseJudgeResult:
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

    raw_winner = str(parsed.get("final_vote", parsed.get("winner", ""))).strip().lower()
    winner_mapping = {
        "a": "A",
        "candidate_a": "A",
        "left": "A",
        "1": "A",
        "b": "B",
        "candidate_b": "B",
        "right": "B",
        "2": "B",
        "tie": "tie",
        "equal": "tie",
        "same": "tie",
    }
    winner = winner_mapping.get(raw_winner, "")
    if not winner:
        winner_match = re.search(r'"winner"\s*:\s*"([^"]+)"', response, flags=re.IGNORECASE)
        if winner_match:
            winner = winner_mapping.get(winner_match.group(1).strip().lower(), "")
    if winner not in {"A", "B", "tie"}:
        winner = "tie"

    confidence = 0.0 if winner == "tie" else 1.0

    judge_reason = str(parsed.get("short_reason", parsed.get("judge_reason", "")))
    if not judge_reason:
        judge_reason_match = re.search(r'"(?:short_reason|judge_reason)"\s*:\s*"([^"]*)"', response, flags=re.IGNORECASE)
        if judge_reason_match:
            judge_reason = judge_reason_match.group(1)

    raw_votes = parsed.get("rule_votes", parsed.get("criterion_votes", []))
    if not isinstance(raw_votes, list):
        raw_votes = []
    normalized_votes: list[dict[str, Any]] = []
    for vote in raw_votes:
        if not isinstance(vote, dict):
            continue
        vote_winner = winner_mapping.get(str(vote.get("vote", vote.get("winner", ""))).strip().lower(), "")
        if vote_winner not in {"A", "B", "tie"}:
            vote_winner = "tie"
        vote_confidence = 0.0 if vote_winner == "tie" else 1.0
        normalized_votes.append(
            {
                "criterion_id": str(vote.get("rule_id", vote.get("criterion_id", ""))).strip(),
                "winner": vote_winner,
                "confidence": max(0.0, min(1.0, vote_confidence)),
                "evidence": str(vote.get("evidence", "")).strip(),
            }
        )

    return PairwiseJudgeResult(
        winner=winner,
        confidence=confidence,
        judge_reason=judge_reason,
        criterion_votes=normalized_votes,
    )


def _aggregate_single_rule_votes(votes: list[dict[str, Any]], rules: list[dict[str, Any]]) -> PairwiseJudgeResult:
    weight_by_rule = {str(rule.get("criterion_id", "")): float(rule.get("weight", 1.0)) for rule in rules}
    a_weight = 0.0
    b_weight = 0.0
    for vote in votes:
        weight = weight_by_rule.get(str(vote.get("criterion_id", "")), 1.0)
        if vote.get("winner") == "A":
            a_weight += weight
        elif vote.get("winner") == "B":
            b_weight += weight
    if a_weight > b_weight:
        winner = "A"
    elif b_weight > a_weight:
        winner = "B"
    else:
        winner = "tie"
    judge_reason = ""
    for vote in votes:
        rationale = str(vote.get("decision_rationale", "")).strip()
        if rationale:
            judge_reason = rationale
            break
    return PairwiseJudgeResult(
        winner=winner,
        confidence=0.0 if winner == "tie" else 1.0,
        judge_reason=judge_reason,
        criterion_votes=votes,
    )


def _judge_single_rule_candidate_pair(
    *,
    schema_id: str,
    extra_info: dict[str, Any],
    rule: dict[str, Any],
    history_payload: dict[str, Any],
    candidate_left: dict[str, Any],
    candidate_right: dict[str, Any],
    candidate_subset: list[dict[str, Any]],
    cache_path: str | None,
    judge_impl: Any,
    include_candidate_id_prompt: bool = False,
) -> dict[str, Any]:
    normalized_left = normalize_prediction(candidate_left["raw_prediction"])
    normalized_right = normalize_prediction(candidate_right["raw_prediction"])
    cache_key = _single_rule_cache_key(
        schema_id,
        extra_info,
        rule_id=str(rule.get("criterion_id", "")),
        left_prediction=normalized_left,
        right_prediction=normalized_right,
        include_candidate_id_prompt=include_candidate_id_prompt,
    )
    cache = get_sqlite_cache(cache_path)
    cached_value = cache.get(cache_key) if cache is not None else None
    if isinstance(cached_value, dict):
        result = SingleRuleJudgeResult(
            rule_id=str(cached_value.get("rule_id", rule.get("criterion_id", ""))),
            vote=_normalize_vote(cached_value.get("vote", "tie")),
            history_basis=str(cached_value.get("history_basis", "")),
            candidate_a_basis=str(cached_value.get("candidate_a_basis", "")),
            candidate_b_basis=str(cached_value.get("candidate_b_basis", "")),
            tie_analysis=str(cached_value.get("tie_analysis", "")),
            decision_rationale=str(cached_value.get("decision_rationale", "")),
            parse_ok=bool(cached_value.get("parse_ok", True)),
        )
        return {
            "result": result,
            "judge_prompt": "",
            "raw_attempts": [],
            "cache_hit": 1.0,
            "retry_count": int(cached_value.get("retry_count", 0)),
            "forced_tie": bool(cached_value.get("forced_tie", False)),
        }

    candidate_pool_captions = _candidate_pool_reference_captions(
        candidate_subset,
        exclude_indices={int(candidate_left.get("candidate_index", 0)), int(candidate_right.get("candidate_index", 0))},
    )
    judge_prompt = build_single_rule_judge_prompt(
        schema_id=schema_id,
        rule=rule,
        history_payload=history_payload,
        candidate_a=candidate_left,
        candidate_b=candidate_right,
        candidate_pool_captions=candidate_pool_captions,
        include_candidate_id=include_candidate_id_prompt,
    )
    raw_attempts: list[str] = []
    retry_count = 0
    parsed_result: SingleRuleJudgeResult | None = None
    for attempt in range(2):
        raw_response = _judge_generate(judge_impl, judge_prompt)
        raw_attempts.append(raw_response)
        parsed = parse_single_rule_judge_response(
            raw_response,
            str(rule.get("criterion_id", "")),
            allow_hidden_identifier_mentions=include_candidate_id_prompt,
        )
        if parsed.parse_ok:
            parsed_result = parsed
            break
        retry_count = attempt + 1
    forced_tie = False
    if parsed_result is None:
        forced_tie = True
        parsed_result = SingleRuleJudgeResult(
            rule_id=str(rule.get("criterion_id", "")),
            vote="tie",
            history_basis="解析失败，强制 tie",
            candidate_a_basis=_candidate_caption_for_prompt(candidate_left, max_chars=80) or "候选 A 语义信息不足",
            candidate_b_basis=_candidate_caption_for_prompt(candidate_right, max_chars=80) or "候选 B 语义信息不足",
            tie_analysis="模型未按要求返回可审计字段，因此强制 tie。",
            decision_rationale="单规则输出不合格，按保守策略记为 tie。",
            parse_ok=False,
        )
    payload = {
        "rule_id": parsed_result.rule_id,
        "vote": parsed_result.vote,
        "history_basis": parsed_result.history_basis,
        "candidate_a_basis": parsed_result.candidate_a_basis,
        "candidate_b_basis": parsed_result.candidate_b_basis,
        "tie_analysis": parsed_result.tie_analysis,
        "decision_rationale": parsed_result.decision_rationale,
        "parse_ok": parsed_result.parse_ok,
        "retry_count": retry_count,
        "forced_tie": forced_tie,
    }
    if cache is not None:
        cache.set(cache_key, payload)
    return {
        "result": parsed_result,
        "judge_prompt": judge_prompt,
        "raw_attempts": raw_attempts,
        "cache_hit": 0.0,
        "retry_count": retry_count,
        "forced_tie": forced_tie,
    }


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
            "adaptive_rules": extra_info.get("adaptive_rules", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt_hash = sha256(prompt_hash_payload.encode("utf-8")).hexdigest()
    return f"{schema_id}:{prompt_hash}:{sha256(normalized_prediction.encode('utf-8')).hexdigest()}"


def _listwise_cache_key(schema_id: str, extra_info: dict[str, Any], normalized_predictions: list[str]) -> str:
    prompt_hash_payload = json.dumps(
        {
            "schema_id": schema_id,
            "prompt_text": extra_info.get("prompt_text", ""),
            "query_text": extra_info.get("query_text", ""),
            "interaction_type": extra_info.get("interaction_type", ""),
            "history_item_captions": extra_info.get("history_item_captions", []),
            "history_ad_captions": extra_info.get("history_ad_captions", []),
            "history_product_captions": extra_info.get("history_product_captions", []),
            "adaptive_rules": extra_info.get("adaptive_rules", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    ranking_hash_payload = json.dumps(normalized_predictions, ensure_ascii=False, sort_keys=False)
    prompt_hash = sha256(prompt_hash_payload.encode("utf-8")).hexdigest()
    ranking_hash = sha256(ranking_hash_payload.encode("utf-8")).hexdigest()
    return f"listwise:{schema_id}:{prompt_hash}:{ranking_hash}"


def _pairwise_cache_key(
    schema_id: str,
    extra_info: dict[str, Any],
    left_prediction: str,
    right_prediction: str,
) -> str:
    prompt_hash_payload = json.dumps(
        {
            "schema_id": schema_id,
            "prompt_text": extra_info.get("prompt_text", ""),
            "query_text": extra_info.get("query_text", ""),
            "interaction_type": extra_info.get("interaction_type", ""),
            "history_item_captions": extra_info.get("history_item_captions", []),
            "history_ad_captions": extra_info.get("history_ad_captions", []),
            "history_product_captions": extra_info.get("history_product_captions", []),
            "adaptive_rules": extra_info.get("adaptive_rules", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    pair_hash_payload = json.dumps([left_prediction, right_prediction], ensure_ascii=False, sort_keys=False)
    prompt_hash = sha256(prompt_hash_payload.encode("utf-8")).hexdigest()
    pair_hash = sha256(pair_hash_payload.encode("utf-8")).hexdigest()
    return f"pairwise:{schema_id}:{prompt_hash}:{pair_hash}"


def _pairwise_margin(result: PairwiseJudgeResult, criteria: list[dict[str, Any]], *, left_is_original_left: bool) -> float:
    criteria_lookup = {str(item.get("criterion_id", "")): item for item in criteria}
    weighted_margin = 0.0
    total_weight = 0.0
    for vote in result.criterion_votes:
        criterion_id = str(vote.get("criterion_id", "")).strip()
        criterion = criteria_lookup.get(criterion_id)
        if criterion is None:
            continue
        weight = float(criterion.get("weight", 1.0))
        total_weight += weight
        winner = vote.get("winner")
        if winner == "A":
            weighted_margin += weight
        elif winner == "B":
            weighted_margin -= weight
    if total_weight > 0:
        signed_margin = weighted_margin / total_weight
    else:
        if result.winner == "A":
            signed_margin = 1.0
        elif result.winner == "B":
            signed_margin = -1.0
        else:
            signed_margin = 0.0
    return signed_margin if left_is_original_left else -signed_margin


def _normalize_pairwise_scores(candidate_indices: list[int], aggregated_scores: dict[int, float]) -> dict[int, float]:
    if not candidate_indices:
        return {}
    score_values = [float(aggregated_scores.get(index, 0.0)) for index in candidate_indices]
    min_score = min(score_values)
    max_score = max(score_values)
    if max_score - min_score <= 1e-9:
        return {index: 0.5 for index in candidate_indices}
    return {
        index: (float(aggregated_scores.get(index, 0.0)) - min_score) / (max_score - min_score)
        for index in candidate_indices
    }


def _compare_pairwise_candidates(
    *,
    schema_id: str,
    rubric: list[dict[str, Any]],
    extra_info: dict[str, Any],
    candidate_left: dict[str, Any],
    candidate_right: dict[str, Any],
    cache_path: str | None,
    judge_impl: Any,
) -> dict[str, Any]:
    normalized_left = normalize_prediction(candidate_left["raw_prediction"])
    normalized_right = normalize_prediction(candidate_right["raw_prediction"])
    cache_key = _pairwise_cache_key(schema_id, extra_info, normalized_left, normalized_right)
    cache = get_sqlite_cache(cache_path)
    cached_value = cache.get(cache_key) if cache is not None else None

    if isinstance(cached_value, dict):
        result = PairwiseJudgeResult(
            winner=str(cached_value.get("winner", "tie")),
            confidence=float(cached_value.get("confidence", 0.0)),
            judge_reason=str(cached_value.get("judge_reason", "")),
            criterion_votes=list(cached_value.get("criterion_votes", [])),
        )
        return {
            "result": result,
            "judge_prompt": "",
            "judge_response": "",
            "cache_hit": 1.0,
            "rubric_applied": float(cached_value.get("rubric_applied", 1.0)),
        }

    left_semantic = any(item.get("caption") for item in candidate_left.get("predicted_items", []))
    right_semantic = any(item.get("caption") for item in candidate_right.get("predicted_items", []))
    if not left_semantic and not right_semantic:
        result = PairwiseJudgeResult(winner="tie", confidence=0.0, judge_reason="语义不足", criterion_votes=[])
        payload = {"winner": result.winner, "confidence": result.confidence, "judge_reason": result.judge_reason, "criterion_votes": [], "rubric_applied": 0.0}
        if cache is not None:
            cache.set(cache_key, payload)
        return {
            "result": result,
            "judge_prompt": "",
            "judge_response": "",
            "cache_hit": 0.0,
            "rubric_applied": 0.0,
        }

    judge_prompt = build_pairwise_judge_prompt(
        schema_id=schema_id,
        rubric=rubric,
        extra_info=extra_info,
        candidate_a=candidate_left,
        candidate_b=candidate_right,
    )
    raw_response = _judge_generate(judge_impl, judge_prompt)
    result = parse_pairwise_judge_response(raw_response)
    payload = {
        "winner": result.winner,
        "confidence": result.confidence,
        "judge_reason": result.judge_reason,
        "criterion_votes": result.criterion_votes,
        "rubric_applied": 1.0,
    }
    if cache is not None:
        cache.set(cache_key, payload)
    return {
        "result": result,
        "judge_prompt": judge_prompt,
        "judge_response": raw_response,
        "cache_hit": 0.0,
        "rubric_applied": 1.0,
    }


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
    rubric = resolve_rubric(schema_id, rubric_dir, extra_info)

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


def compute_listwise_rubric_rerank(
    predictions: list[str],
    extra_info: dict[str, Any] | None,
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_lookup: dict[str, dict[str, Any]] | None = None,
    cache_path: str | None = None,
    timeout_s: int = 30,
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 768,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
) -> dict[str, Any]:
    extra_info = parse_json_like(extra_info, default={})
    sidecar_lookup = sidecar_lookup or {}
    schema_id = str(extra_info.get("schema_id") or extra_info.get("task_name") or "unknown")
    rubric = resolve_rubric(schema_id, rubric_dir, extra_info)

    candidate_states: list[dict[str, Any]] = []
    normalized_predictions: list[str] = []
    for index, prediction in enumerate(predictions, start=1):
        predicted_items, unresolved_sid_ratio = resolve_predicted_items(prediction, sidecar_lookup)
        normalized_predictions.append(normalize_prediction(prediction))
        candidate_states.append(
            {
                "candidate_index": index,
                "raw_rank": index,
                "raw_prediction": prediction,
                "predicted_items": predicted_items,
                "predicted_pid": predicted_items[0].get("pid") if predicted_items else None,
                "unresolved_sid_ratio": unresolved_sid_ratio,
            }
        )

    cache_key = _listwise_cache_key(schema_id=schema_id, extra_info=extra_info, normalized_predictions=normalized_predictions)
    cache = get_sqlite_cache(cache_path)
    cached_value = cache.get(cache_key) if cache is not None else None
    if isinstance(cached_value, dict) and isinstance(cached_value.get("ranking"), list):
        listwise_result = ListwiseJudgeResult(
            ranking=list(cached_value.get("ranking", [])),
            judge_reason=str(cached_value.get("judge_reason", "")),
        )
        cache_hit = 1.0
    else:
        cache_hit = 0.0
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

        if judge_impl is None:
            listwise_result = ListwiseJudgeResult(
                ranking=[
                    {
                        "candidate_index": candidate["candidate_index"],
                        "score": max(0.0, 1.0 - ((candidate["candidate_index"] - 1) / max(len(candidate_states), 1))),
                        "reason": "",
                    }
                    for candidate in candidate_states
                ],
                judge_reason="judge unavailable",
            )
            rubric_applied = 0.0
            judge_prompt = ""
            raw_response = ""
        else:
            judge_prompt = build_listwise_judge_prompt(
                schema_id=schema_id,
                rubric=rubric,
                extra_info=extra_info,
                candidates=candidate_states,
            )
            raw_response = _judge_generate(judge_impl, judge_prompt)
            listwise_result = parse_listwise_judge_response(raw_response, len(candidate_states))
            rubric_applied = 1.0
            if cache is not None:
                cache.set(
                    cache_key,
                    {
                        "ranking": listwise_result.ranking,
                        "judge_reason": listwise_result.judge_reason,
                        "rubric_applied": rubric_applied,
                    },
                )

    if cache_hit >= 1.0:
        rubric_applied = float(cached_value.get("rubric_applied", 1.0))
        judge_prompt = ""
        raw_response = ""

    ranking_lookup = {
        int(entry.get("candidate_index", 0)): {
            "listwise_rank": rank,
            "rubric_score": max(0.0, min(1.0, float(entry.get("score", 0.0)))),
            "judge_reason": str(entry.get("reason", "")),
        }
        for rank, entry in enumerate(listwise_result.ranking, start=1)
        if 1 <= int(entry.get("candidate_index", 0)) <= len(candidate_states)
    }

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidate_states:
        ranking_entry = ranking_lookup.get(candidate["candidate_index"], {})
        listwise_rank = int(ranking_entry.get("listwise_rank", candidate["candidate_index"]))
        rubric_score = float(ranking_entry.get("rubric_score", max(0.0, 1.0 - ((listwise_rank - 1) / max(len(candidate_states), 1)))))
        scored_candidates.append(
            {
                "output": candidate["raw_prediction"],
                "raw_rank": candidate["raw_rank"],
                "predicted_pid": candidate["predicted_pid"],
                "predicted_items": candidate["predicted_items"],
                "rubric_score": rubric_score,
                "judge_reason": str(ranking_entry.get("judge_reason", "")),
                "unresolved_sid_ratio": float(candidate["unresolved_sid_ratio"]),
                "cache_hit": cache_hit,
                "rubric_applied": rubric_applied,
                "listwise_rank": listwise_rank,
            }
        )

    return {
        "schema_id": schema_id,
        "rubric": rubric,
        "candidates": scored_candidates,
        "judge_prompt": judge_prompt,
        "judge_response": raw_response,
        "judge_reason": listwise_result.judge_reason,
        "cache_hit": cache_hit,
        "rubric_applied": rubric_applied,
    }


def _compute_single_rule_shared_rerank(
    predictions: list[str],
    extra_info: dict[str, Any],
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_lookup: dict[str, dict[str, Any]] | None = None,
    cache_path: str | None = None,
    timeout_s: int = 30,
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 256,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
    history_summary_judge_impl: Any = None,
    pairwise_top_n: int = 8,
    include_candidate_id_prompt: bool = False,
) -> dict[str, Any]:
    sidecar_lookup = sidecar_lookup or {}
    schema_id = str(extra_info.get("schema_id") or extra_info.get("task_name") or "unknown")
    rubric = resolve_rubric(schema_id, rubric_dir, extra_info)
    single_rule_criteria = _build_single_rule_shared_criteria(schema_id, rubric)

    candidate_states: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        predicted_items, unresolved_sid_ratio = resolve_predicted_items(prediction, sidecar_lookup)
        candidate_states.append(
            {
                "candidate_index": index,
                "raw_rank": index,
                "raw_prediction": prediction,
                "predicted_items": predicted_items,
                "predicted_pid": predicted_items[0].get("pid") if predicted_items else None,
                "unresolved_sid_ratio": unresolved_sid_ratio,
            }
        )

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

    compare_count = min(max(int(pairwise_top_n), 0), len(candidate_states))
    history_payload = _build_history_summary_payload(
        schema_id,
        extra_info,
        cache_path=cache_path,
        judge_impl=history_summary_judge_impl if history_summary_judge_impl is not None else judge_impl,
    )
    history_summary_record = {
        "user_goal": history_payload.get("user_goal", ""),
        "history_summary": history_payload.get("history_summary", {}),
        "history_evidence_snippets": history_payload.get("history_evidence_snippets", []),
        "history_summary_nonempty": bool(history_payload.get("history_summary_nonempty", False)),
    }

    if compare_count < 2 or judge_impl is None:
        fallback_candidates: list[dict[str, Any]] = []
        for candidate in candidate_states:
            fallback_candidates.append(
                {
                    "output": candidate["raw_prediction"],
                    "raw_rank": candidate["raw_rank"],
                    "predicted_pid": candidate["predicted_pid"],
                    "predicted_items": candidate["predicted_items"],
                    "rubric_score": max(0.0, 1.0 - ((candidate["raw_rank"] - 1) / max(len(candidate_states), 1))),
                    "judge_reason": "",
                    "unresolved_sid_ratio": float(candidate["unresolved_sid_ratio"]),
                    "cache_hit": 0.0,
                    "rubric_applied": 0.0,
                    "pairwise_rank": candidate["raw_rank"],
                    "pairwise_score": 0.0,
                    "pairwise_structured_score": 0.0,
                }
            )
        return {
            "schema_id": schema_id,
            "rubric": rubric,
            "candidates": fallback_candidates,
            "pairwise_judgments": [],
            "judge_reason": "judge unavailable" if judge_impl is None else "pairwise_top_n < 2",
            "cache_hit": 0.0,
            "rubric_applied": 0.0,
            "pairwise_top_n": compare_count,
            "pairwise_raw_rank_anchor": 0.0,
            "typed_criteria": single_rule_criteria,
            "history_summary_record": history_summary_record,
            "single_rule_prompts": [],
            "single_rule_outputs": [],
            "single_rule_audit": [],
            "audit_metrics": {
                "expected_rule_calls": 0,
                "completed_rule_calls": 0,
                "retry_count": 0,
                "forced_tie_count": 0,
                "swap_consistent_rule_pairs": 0,
                "total_rule_pairs": 0,
                "history_summary_nonempty": bool(history_payload.get("history_summary_nonempty", False)),
            },
        }

    candidate_subset = candidate_states[:compare_count]
    aggregated_scores: dict[int, float] = {candidate["candidate_index"]: 0.0 for candidate in candidate_subset}
    pairwise_judgments: list[dict[str, Any]] = []
    single_rule_prompts: list[dict[str, Any]] = []
    single_rule_outputs: list[dict[str, Any]] = []
    single_rule_audit: list[dict[str, Any]] = []
    cache_hits: list[float] = []
    pair_counter = 0
    retry_count = 0
    forced_tie_count = 0
    completed_rule_calls = 0
    total_rule_pairs = 0
    swap_consistent_rule_pairs = 0

    for left_candidate, right_candidate in combinations(candidate_subset, 2):
        pair_counter += 1
        pair_numerator = 0.0
        total_weight = 0.0
        forward_votes: list[dict[str, Any]] = []
        backward_votes: list[dict[str, Any]] = []

        for rule in single_rule_criteria:
            total_rule_pairs += 1
            weight = float(rule.get("weight", 1.0))
            total_weight += weight
            forward_payload = _judge_single_rule_candidate_pair(
                schema_id=schema_id,
                extra_info=extra_info,
                rule=rule,
                history_payload=history_payload,
                candidate_left=left_candidate,
                candidate_right=right_candidate,
                candidate_subset=candidate_subset,
                cache_path=cache_path,
                judge_impl=judge_impl,
                include_candidate_id_prompt=include_candidate_id_prompt,
            )
            backward_payload = _judge_single_rule_candidate_pair(
                schema_id=schema_id,
                extra_info=extra_info,
                rule=rule,
                history_payload=history_payload,
                candidate_left=right_candidate,
                candidate_right=left_candidate,
                candidate_subset=candidate_subset,
                cache_path=cache_path,
                judge_impl=judge_impl,
                include_candidate_id_prompt=include_candidate_id_prompt,
            )
            forward_result = forward_payload["result"]
            backward_result = backward_payload["result"]
            swap_consistent = _mirror_vote(forward_result.vote) == backward_result.vote
            if swap_consistent:
                swap_consistent_rule_pairs += 1
                if forward_result.vote == "A":
                    pair_numerator += weight
                elif forward_result.vote == "B":
                    pair_numerator -= weight

            cache_hits.extend([float(forward_payload["cache_hit"]), float(backward_payload["cache_hit"])])
            retry_count += int(forward_payload["retry_count"]) + int(backward_payload["retry_count"])
            forced_tie_count += int(bool(forward_payload["forced_tie"])) + int(bool(backward_payload["forced_tie"]))
            completed_rule_calls += 2

            forward_vote_record = {
                "criterion_id": str(rule.get("criterion_id", "")),
                "winner": forward_result.vote,
                "confidence": 0.0 if forward_result.vote == "tie" else 1.0,
                "evidence": forward_result.decision_rationale,
                "history_basis": forward_result.history_basis,
                "candidate_a_basis": forward_result.candidate_a_basis,
                "candidate_b_basis": forward_result.candidate_b_basis,
                "tie_analysis": forward_result.tie_analysis,
                "decision_rationale": forward_result.decision_rationale,
                "swap_consistent": swap_consistent,
                "retry_count": int(forward_payload["retry_count"]),
                "forced_tie": bool(forward_payload["forced_tie"]),
            }
            backward_vote_record = {
                "criterion_id": str(rule.get("criterion_id", "")),
                "winner": backward_result.vote,
                "confidence": 0.0 if backward_result.vote == "tie" else 1.0,
                "evidence": backward_result.decision_rationale,
                "history_basis": backward_result.history_basis,
                "candidate_a_basis": backward_result.candidate_a_basis,
                "candidate_b_basis": backward_result.candidate_b_basis,
                "tie_analysis": backward_result.tie_analysis,
                "decision_rationale": backward_result.decision_rationale,
                "swap_consistent": swap_consistent,
                "retry_count": int(backward_payload["retry_count"]),
                "forced_tie": bool(backward_payload["forced_tie"]),
            }
            forward_votes.append(forward_vote_record)
            backward_votes.append(backward_vote_record)

            for swap_flag, payload, result, left_idx, right_idx in (
                (False, forward_payload, forward_result, int(left_candidate["candidate_index"]), int(right_candidate["candidate_index"])),
                (True, backward_payload, backward_result, int(right_candidate["candidate_index"]), int(left_candidate["candidate_index"])),
            ):
                single_rule_prompts.append(
                    {
                        "comparison_id": pair_counter,
                        "swap": swap_flag,
                        "rule_id": str(rule.get("criterion_id", "")),
                        "left_candidate_index": left_idx,
                        "right_candidate_index": right_idx,
                        "judge_prompt": payload.get("judge_prompt", ""),
                    }
                )
                single_rule_outputs.append(
                    {
                        "comparison_id": pair_counter,
                        "swap": swap_flag,
                        "rule_id": str(rule.get("criterion_id", "")),
                        "left_candidate_index": left_idx,
                        "right_candidate_index": right_idx,
                        "raw_attempts": list(payload.get("raw_attempts", [])),
                    }
                )
                single_rule_audit.append(
                    {
                        "comparison_id": pair_counter,
                        "swap": swap_flag,
                        "rule_id": str(rule.get("criterion_id", "")),
                        "left_candidate_index": left_idx,
                        "right_candidate_index": right_idx,
                        "vote": result.vote,
                        "history_basis": result.history_basis,
                        "candidate_a_basis": result.candidate_a_basis,
                        "candidate_b_basis": result.candidate_b_basis,
                        "tie_analysis": result.tie_analysis,
                        "decision_rationale": result.decision_rationale,
                        "parse_ok": bool(result.parse_ok),
                        "retry_count": int(payload.get("retry_count", 0)),
                        "forced_tie": bool(payload.get("forced_tie", False)),
                        "cache_hit": float(payload.get("cache_hit", 0.0)),
                        "swap_consistent": swap_consistent,
                    }
                )

        pair_margin = pair_numerator / total_weight if total_weight > 0 else 0.0
        aggregated_scores[left_candidate["candidate_index"]] += pair_margin
        aggregated_scores[right_candidate["candidate_index"]] -= pair_margin

        forward_pair_result = _aggregate_single_rule_votes(forward_votes, single_rule_criteria)
        backward_pair_result = _aggregate_single_rule_votes(backward_votes, single_rule_criteria)
        pairwise_judgments.extend(
            [
                {
                    "left_candidate_index": left_candidate["candidate_index"],
                    "right_candidate_index": right_candidate["candidate_index"],
                    "left_raw_rank": left_candidate["raw_rank"],
                    "right_raw_rank": right_candidate["raw_rank"],
                    "swap": False,
                    "winner": forward_pair_result.winner,
                    "confidence": forward_pair_result.confidence,
                    "judge_reason": forward_pair_result.judge_reason,
                    "rule_votes": forward_pair_result.criterion_votes,
                    "criterion_votes": forward_pair_result.criterion_votes,
                    "judge_prompt": "",
                    "judge_response": "",
                    "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
                    "rubric_applied": 1.0,
                    "aggregated_margin": pair_margin,
                },
                {
                    "left_candidate_index": right_candidate["candidate_index"],
                    "right_candidate_index": left_candidate["candidate_index"],
                    "left_raw_rank": right_candidate["raw_rank"],
                    "right_raw_rank": left_candidate["raw_rank"],
                    "swap": True,
                    "winner": backward_pair_result.winner,
                    "confidence": backward_pair_result.confidence,
                    "judge_reason": backward_pair_result.judge_reason,
                    "rule_votes": backward_pair_result.criterion_votes,
                    "criterion_votes": backward_pair_result.criterion_votes,
                    "judge_prompt": "",
                    "judge_response": "",
                    "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
                    "rubric_applied": 1.0,
                    "aggregated_margin": pair_margin,
                },
            ]
        )

    normalized_scores = _normalize_pairwise_scores(
        [candidate["candidate_index"] for candidate in candidate_subset],
        aggregated_scores,
    )
    ordered_subset = sorted(
        candidate_subset,
        key=lambda candidate: (
            -float(normalized_scores.get(candidate["candidate_index"], 0.5)),
            -float(aggregated_scores.get(candidate["candidate_index"], 0.0)),
            int(candidate["raw_rank"]),
        ),
    )
    rank_lookup = {candidate["candidate_index"]: rank for rank, candidate in enumerate(ordered_subset, start=1)}

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidate_states:
        candidate_index = candidate["candidate_index"]
        if candidate_index in rank_lookup:
            pairwise_rank = rank_lookup[candidate_index]
            pairwise_score = float(aggregated_scores.get(candidate_index, 0.0))
            structured_score = float(normalized_scores.get(candidate_index, 0.5))
            rubric_score = structured_score
        else:
            pairwise_rank = compare_count + (candidate["raw_rank"] - compare_count)
            pairwise_score = -1.0 - float(candidate["raw_rank"] - compare_count)
            fallback_scale = max(len(candidate_states) - compare_count, 1)
            structured_score = max(0.0, 0.49 - ((candidate["raw_rank"] - compare_count - 1) / fallback_scale) * 0.49)
            rubric_score = structured_score
        scored_candidates.append(
            {
                "output": candidate["raw_prediction"],
                "raw_rank": candidate["raw_rank"],
                "predicted_pid": candidate["predicted_pid"],
                "predicted_items": candidate["predicted_items"],
                "rubric_score": rubric_score,
                "judge_reason": "",
                "unresolved_sid_ratio": float(candidate["unresolved_sid_ratio"]),
                "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
                "rubric_applied": 1.0,
                "pairwise_rank": pairwise_rank,
                "pairwise_score": pairwise_score,
                "pairwise_structured_score": structured_score,
            }
        )

    pairwise_reason_lookup: dict[int, list[str]] = {}
    for record in pairwise_judgments:
        winner = record.get("winner")
        if winner == "A":
            winner_index = int(record["left_candidate_index"])
        elif winner == "B":
            winner_index = int(record["right_candidate_index"])
        else:
            continue
        reason = str(record.get("judge_reason", "")).strip()
        if not reason:
            continue
        pairwise_reason_lookup.setdefault(winner_index, []).append(reason)
    for candidate in scored_candidates:
        reasons = pairwise_reason_lookup.get(candidate["raw_rank"], [])
        candidate["judge_reason"] = reasons[0] if reasons else ""

    expected_rule_calls = len(list(combinations(candidate_subset, 2))) * len(single_rule_criteria) * 2
    return {
        "schema_id": schema_id,
        "rubric": rubric,
        "candidates": scored_candidates,
        "pairwise_judgments": pairwise_judgments,
        "judge_reason": "single-rule pairwise+swap aggregation",
        "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
        "rubric_applied": 1.0,
        "pairwise_top_n": compare_count,
        "pairwise_raw_rank_anchor": 0.0,
        "typed_criteria": single_rule_criteria,
        "history_summary_record": history_summary_record,
        "single_rule_prompts": single_rule_prompts,
        "single_rule_outputs": single_rule_outputs,
        "single_rule_audit": single_rule_audit,
        "audit_metrics": {
            "expected_rule_calls": expected_rule_calls,
            "completed_rule_calls": completed_rule_calls,
            "retry_count": retry_count,
            "forced_tie_count": forced_tie_count,
            "swap_consistent_rule_pairs": swap_consistent_rule_pairs,
            "total_rule_pairs": total_rule_pairs,
            "history_summary_nonempty": bool(history_payload.get("history_summary_nonempty", False)),
        },
    }


def compute_pairwise_rubric_rerank(
    predictions: list[str],
    extra_info: dict[str, Any] | None,
    *,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    rubric_dir: str | None = None,
    sidecar_lookup: dict[str, dict[str, Any]] | None = None,
    cache_path: str | None = None,
    timeout_s: int = 30,
    judge_backend: str = "auto",
    judge_max_new_tokens: int = 256,
    judge_device_map: str = "auto",
    judge_torch_dtype: str = "bfloat16",
    judge_attn_implementation: str | None = None,
    judge_impl: Any = None,
    history_summary_judge_impl: Any = None,
    pairwise_top_n: int = 8,
    pairwise_raw_rank_anchor: float = 0.15,
    pairwise_judge_mode: str = "multi_rule",
    single_rule_include_candidate_id: bool = False,
) -> dict[str, Any]:
    extra_info = parse_json_like(extra_info, default={})
    if str(pairwise_judge_mode).strip().lower() == "single_rule_audit":
        return _compute_single_rule_shared_rerank(
            predictions,
            extra_info,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            rubric_dir=rubric_dir,
            sidecar_lookup=sidecar_lookup,
            cache_path=cache_path,
            timeout_s=timeout_s,
            judge_backend=judge_backend,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_device_map=judge_device_map,
            judge_torch_dtype=judge_torch_dtype,
            judge_attn_implementation=judge_attn_implementation,
            judge_impl=judge_impl,
            history_summary_judge_impl=history_summary_judge_impl,
            pairwise_top_n=pairwise_top_n,
            include_candidate_id_prompt=single_rule_include_candidate_id,
        )
    sidecar_lookup = sidecar_lookup or {}
    schema_id = str(extra_info.get("schema_id") or extra_info.get("task_name") or "unknown")
    rubric = resolve_rubric(schema_id, rubric_dir, extra_info)

    candidate_states: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        predicted_items, unresolved_sid_ratio = resolve_predicted_items(prediction, sidecar_lookup)
        candidate_states.append(
            {
                "candidate_index": index,
                "raw_rank": index,
                "raw_prediction": prediction,
                "predicted_items": predicted_items,
                "predicted_pid": predicted_items[0].get("pid") if predicted_items else None,
                "unresolved_sid_ratio": unresolved_sid_ratio,
            }
        )

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

    compare_count = min(max(int(pairwise_top_n), 0), len(candidate_states))
    if compare_count < 2 or judge_impl is None:
        fallback_candidates: list[dict[str, Any]] = []
        for candidate in candidate_states:
            fallback_candidates.append(
                {
                    "output": candidate["raw_prediction"],
                    "raw_rank": candidate["raw_rank"],
                    "predicted_pid": candidate["predicted_pid"],
                    "predicted_items": candidate["predicted_items"],
                    "rubric_score": max(0.0, 1.0 - ((candidate["raw_rank"] - 1) / max(len(candidate_states), 1))),
                    "judge_reason": "",
                    "unresolved_sid_ratio": float(candidate["unresolved_sid_ratio"]),
                    "cache_hit": 0.0,
                    "rubric_applied": 0.0,
                    "pairwise_rank": candidate["raw_rank"],
                    "pairwise_score": 0.0,
                    "pairwise_structured_score": 0.0,
                }
            )
        return {
            "schema_id": schema_id,
            "rubric": rubric,
            "candidates": fallback_candidates,
            "pairwise_judgments": [],
            "judge_reason": "judge unavailable" if judge_impl is None else "pairwise_top_n < 2",
            "cache_hit": 0.0,
            "rubric_applied": 0.0,
            "pairwise_top_n": compare_count,
            "pairwise_raw_rank_anchor": pairwise_raw_rank_anchor,
        }

    candidate_subset = candidate_states[:compare_count]
    aggregated_scores: dict[int, float] = {candidate["candidate_index"]: 0.0 for candidate in candidate_subset}
    pairwise_judgments: list[dict[str, Any]] = []
    cache_hits: list[float] = []
    rubric_applied_flags: list[float] = []

    typed_criteria = _build_pairwise_criteria(schema_id, rubric, extra_info)

    for left_candidate, right_candidate in combinations(candidate_subset, 2):
        forward_payload = _compare_pairwise_candidates(
            schema_id=schema_id,
            rubric=rubric,
            extra_info=extra_info,
            candidate_left=left_candidate,
            candidate_right=right_candidate,
            cache_path=cache_path,
            judge_impl=judge_impl,
        )
        backward_payload = _compare_pairwise_candidates(
            schema_id=schema_id,
            rubric=rubric,
            extra_info=extra_info,
            candidate_left=right_candidate,
            candidate_right=left_candidate,
            cache_path=cache_path,
            judge_impl=judge_impl,
        )

        forward_result = forward_payload["result"]
        backward_result = backward_payload["result"]
        structured_margin = (
            _pairwise_margin(forward_result, typed_criteria, left_is_original_left=True)
            + _pairwise_margin(backward_result, typed_criteria, left_is_original_left=False)
        ) / 2.0
        aggregated_scores[left_candidate["candidate_index"]] += structured_margin
        aggregated_scores[right_candidate["candidate_index"]] -= structured_margin

        cache_hits.extend([float(forward_payload["cache_hit"]), float(backward_payload["cache_hit"])])
        rubric_applied_flags.extend([float(forward_payload["rubric_applied"]), float(backward_payload["rubric_applied"])])
        pairwise_judgments.extend(
            [
                {
                    "left_candidate_index": left_candidate["candidate_index"],
                    "right_candidate_index": right_candidate["candidate_index"],
                    "left_raw_rank": left_candidate["raw_rank"],
                    "right_raw_rank": right_candidate["raw_rank"],
                    "swap": False,
                    "winner": forward_result.winner,
                    "confidence": forward_result.confidence,
                    "judge_reason": forward_result.judge_reason,
                    "rule_votes": forward_result.criterion_votes,
                    "criterion_votes": forward_result.criterion_votes,
                    "judge_prompt": forward_payload["judge_prompt"],
                    "judge_response": forward_payload["judge_response"],
                    "cache_hit": float(forward_payload["cache_hit"]),
                    "rubric_applied": float(forward_payload["rubric_applied"]),
                    "aggregated_margin": structured_margin,
                },
                {
                    "left_candidate_index": right_candidate["candidate_index"],
                    "right_candidate_index": left_candidate["candidate_index"],
                    "left_raw_rank": right_candidate["raw_rank"],
                    "right_raw_rank": left_candidate["raw_rank"],
                    "swap": True,
                    "winner": backward_result.winner,
                    "confidence": backward_result.confidence,
                    "judge_reason": backward_result.judge_reason,
                    "rule_votes": backward_result.criterion_votes,
                    "criterion_votes": backward_result.criterion_votes,
                    "judge_prompt": backward_payload["judge_prompt"],
                    "judge_response": backward_payload["judge_response"],
                    "cache_hit": float(backward_payload["cache_hit"]),
                    "rubric_applied": float(backward_payload["rubric_applied"]),
                    "aggregated_margin": structured_margin,
                },
            ]
        )

    normalized_scores = _normalize_pairwise_scores(
        [candidate["candidate_index"] for candidate in candidate_subset],
        aggregated_scores,
    )
    anchored_scores: dict[int, float] = {}
    anchor_weight = max(0.0, min(0.4, float(pairwise_raw_rank_anchor)))
    for candidate in candidate_subset:
        candidate_index = candidate["candidate_index"]
        raw_rank_prior = max(0.0, 1.0 - ((candidate["raw_rank"] - 1) / max(len(candidate_states) - 1, 1)))
        structured_score = float(normalized_scores.get(candidate_index, 0.5))
        anchored_scores[candidate_index] = (1.0 - anchor_weight) * structured_score + anchor_weight * raw_rank_prior
    ordered_subset = sorted(
        candidate_subset,
        key=lambda candidate: (
            -float(anchored_scores.get(candidate["candidate_index"], 0.5)),
            -float(aggregated_scores.get(candidate["candidate_index"], 0.0)),
            int(candidate["raw_rank"]),
        ),
    )
    rank_lookup = {candidate["candidate_index"]: rank for rank, candidate in enumerate(ordered_subset, start=1)}

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidate_states:
        candidate_index = candidate["candidate_index"]
        if candidate_index in rank_lookup:
            pairwise_rank = rank_lookup[candidate_index]
            pairwise_score = float(aggregated_scores.get(candidate_index, 0.0))
            structured_score = float(normalized_scores.get(candidate_index, 0.5))
            rubric_score = float(anchored_scores.get(candidate_index, structured_score))
        else:
            pairwise_rank = compare_count + (candidate["raw_rank"] - compare_count)
            pairwise_score = -1.0 - float(candidate["raw_rank"] - compare_count)
            fallback_scale = max(len(candidate_states) - compare_count, 1)
            structured_score = max(0.0, 0.49 - ((candidate["raw_rank"] - compare_count - 1) / fallback_scale) * 0.49)
            rubric_score = structured_score
        scored_candidates.append(
            {
                "output": candidate["raw_prediction"],
                "raw_rank": candidate["raw_rank"],
                "predicted_pid": candidate["predicted_pid"],
                "predicted_items": candidate["predicted_items"],
                "rubric_score": rubric_score,
                "judge_reason": "",
                "unresolved_sid_ratio": float(candidate["unresolved_sid_ratio"]),
                "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
                "rubric_applied": float(max(rubric_applied_flags)) if rubric_applied_flags else 0.0,
                "pairwise_rank": pairwise_rank,
                "pairwise_score": pairwise_score,
                "pairwise_structured_score": structured_score,
            }
        )

    pairwise_reason_lookup: dict[int, list[str]] = {}
    for record in pairwise_judgments:
        winner = record.get("winner")
        if winner == "A":
            winner_index = int(record["left_candidate_index"])
        elif winner == "B":
            winner_index = int(record["right_candidate_index"])
        else:
            continue
        reason = str(record.get("judge_reason", "")).strip()
        if not reason:
            continue
        pairwise_reason_lookup.setdefault(winner_index, []).append(reason)

    for candidate in scored_candidates:
        reasons = pairwise_reason_lookup.get(candidate["raw_rank"], [])
        candidate["judge_reason"] = reasons[0] if reasons else ""

    return {
        "schema_id": schema_id,
        "rubric": rubric,
        "candidates": scored_candidates,
        "pairwise_judgments": pairwise_judgments,
        "judge_reason": "pairwise+swap aggregation",
        "cache_hit": float(sum(cache_hits) / len(cache_hits)) if cache_hits else 0.0,
        "rubric_applied": float(max(rubric_applied_flags)) if rubric_applied_flags else 0.0,
        "pairwise_top_n": compare_count,
        "pairwise_raw_rank_anchor": anchor_weight,
        "typed_criteria": typed_criteria,
    }


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
