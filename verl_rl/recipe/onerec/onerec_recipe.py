from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from recipe.onerec.rubric_reward import (
    compute_score as rubric_compute_score,
    compute_score_batch as rubric_compute_score_batch,
    parse_json_like,
)
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

__all__ = ["collate_fn", "OneRecDataset", "compute_score", "compute_score_batch"]

def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    non_tensors: dict[str, list[Any]] = defaultdict(list)

    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    batch: dict[str, Any] = {}
    for key, value in tensors.items():
        batch[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        batch[key] = np.array(value, dtype=object)

    return batch


class OneRecDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.enable_think = config.get("enable_think", True)
        self.enable_nonthink = config.get("enable_nonthink", False)
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed", None)

        self.use_force_prefix = config.get("use_force_prefix", False)
        self._FORCE_PREFIX_CONTENT = "<think>\n</think><|sid_begin|>"

        if self.enable_think and self.enable_nonthink:
            raise ValueError("enable_think and enable_nonthink cannot be both True") 

        self.num_workers = os.cpu_count()
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet: bool = False) -> None:
        from verl.utils.fs import copy_to_local

        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path

        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        dataframes: list[datasets.Dataset] = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)  # type: ignore[attr-defined]
        logger.info("dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {len(self.dataframe)}")

        self.dataframe = self.dataframe.map(
            self._extract_prompt_fields,
            num_proc=self.num_workers,
            desc="Extract prompts and reward annotations",
        )

        logger.info("processed dataset len: %s", len(self.dataframe))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        raw_messages = row.get("messages")
        if isinstance(raw_messages, str):
            messages = ast.literal_eval(raw_messages)
        else:
            messages = raw_messages or []

        clean_chats = [
            {
                "role": message.get("role"),
                "content": "".join(segment.get("text", "") for segment in message.get("content", []) if segment.get("type") == "text"),
            }
            for message in messages
        ]

        if not clean_chats:
            raise ValueError("Sample has empty messages; please check data integrity.")

        prompt_messages = clean_chats[:-1]

        # Append /think or /no_think suffix to user messages based on config
        if self.enable_think:
            for message in prompt_messages:
                if message["role"] == "user":
                    message["content"] = message["content"] + "/think"
        if self.enable_nonthink:
            for message in prompt_messages:
                if message["role"] == "user":
                    message["content"] = message["content"] + "/no_think"


        ground_truth_message = clean_chats[-1]["content"]
        metadata = parse_json_like(row.get("metadata"), default={})
        extra_info = parse_json_like(row.get("extra_info"), default={})
        if not isinstance(extra_info, dict):
            extra_info = {}
        if not extra_info:
            extra_info = {
                "task_name": str(row.get("source", "unknown")),
                "task_variant": str(row.get("source", "unknown")),
                "schema_id": str(row.get("source", "unknown")),
                "prompt_text": "\n".join(
                    message["content"] for message in prompt_messages if message.get("content")
                ),
                "query_text": "",
                "interaction_type": str(metadata.get("target_interaction", "")),
                "history_item_captions": [],
                "history_ad_captions": [],
                "history_product_captions": [],
                "ground_truth_sids": [],
                "ground_truth_captions": [],
                "context_version": "legacy",
                "metadata": metadata,
            }
        else:
            extra_info.setdefault("metadata", metadata)
            extra_info.setdefault(
                "prompt_text",
                "\n".join(message["content"] for message in prompt_messages if message.get("content")),
            )

        reward_payload = {
            "ground_truth": ground_truth_message,
            "style": "rubric" if extra_info else "rule",
            "schema_id": extra_info.get("schema_id", ""),
        }

        row[self.prompt_key] = prompt_messages
        row["reward_model"] = reward_payload
        row["extra_info"] = extra_info
        return row

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc_length(doc: dict[str, Any]) -> int:
                messages = self._build_messages(dict(doc))
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in doc.get(image_key, [])]
                videos = [process_video(video) for video in doc.get(video_key, [])]
                encoded = processor(text=[raw_prompt], images=images or None, videos=videos or None, return_tensors="pt")
                return int(encoded["input_ids"].shape[-1])

        else:

            def doc_length(doc: dict[str, Any]) -> int:
                messages = doc[prompt_key]
                return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length - 10,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
        )

        logger.info("filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
                parsed_segments = []
                for segment in segments:
                    if segment == "<image>":
                        parsed_segments.append({"type": "image"})
                    elif segment == "<video>":
                        parsed_segments.append({"type": "video"})
                    else:
                        parsed_segments.append({"type": "text", "text": segment})
                message["content"] = parsed_segments

        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        model_inputs: dict[str, Any] = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

            multi_modal_data: dict[str, Any] = {}

            images = None
            if self.image_key in row and row.get(self.image_key):
                images = [process_image(image) for image in row.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row and row.get(self.video_key):
                videos = [process_video(video) for video in row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            row["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                mm_inputs = dict(model_inputs)
                mm_inputs.pop("second_per_grid_ts", None)
                row["multi_modal_inputs"] = mm_inputs
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)

        row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        extra_info = parse_json_like(row.get("extra_info"), default={}) or {}
        row["extra_info"] = extra_info
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})


        if "source" in row or "data_source" in row:
            pass
        else:
            row["data_source"] = "unknown"
            logger.warning("No source/data_source field found for index %s, set to 'unknown'", row["index"])

        if self.need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", row["index"], row.get("data_source", row.get("source", "unknown")))

        return row

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        if self.truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.truncation == "right":
            return token_ids[: self.max_prompt_length]
        if self.truncation == "middle":
            left = self.max_prompt_length // 2
            right = self.max_prompt_length - left
            return token_ids[:left] + token_ids[-right:]
        if self.truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}. "
                "Consider increasingmax_prompt_length or enabling truncation."
            )
        raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def _extract_all_tuples(text: Any) -> list[tuple[str, str, str]]:
    if not isinstance(text, str):
        logger.warning("_extract_all_tuples received non-string input: %s", type(text))
        return []

    matches = SLOT_PATTERN.findall(text)
    return [tuple(match) for match in matches] if matches else []


def think_format_reward(prediction: str) -> float:
    """Check if prediction contains valid think format.

    Args:
        prediction: Model prediction text.

    Returns:
        1.0 if contains valid <think>...</think> with content length > 10, else 0.0.
    """
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0

    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")

    if end_idx < start_idx:
        return 0.0

    content = prediction[start_idx:end_idx]
    content_stripped = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    return 1.0 if len(content_stripped) > 10 else 0.0


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate hierarchical matching reward with partial match support.

    Args:
        prediction: Model prediction text, may contain multiple sids.
        ground_truth: Ground truth text, may contain multiple sids.

    Returns:
        Weighted match score:
        - Full match (s_a, s_b, s_c): 100 points
        - s_a and s_b match: 10 points
        - Only s_a match: 1 point
        - No match: 0 points
        Returns average score across all predicted sids.
    """
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)

    if not pred_tuples or not gt_tuples:
        return 0.0

    total_reward = 0.0

    # Find best match for each predicted sid and calculate score
    for pred_tuple in pred_tuples:
        max_score = 0.0
        
        for gt_tuple in gt_tuples:
            # Full match (s_a, s_b, s_c)
            if pred_tuple == gt_tuple:
                max_score = max(max_score, 100.0)
            # s_a and s_b match
            elif pred_tuple[:2] == gt_tuple[:2]:
                max_score = max(max_score, 10.0)
            # Only s_a match
            elif pred_tuple[0] == gt_tuple[0]:
                max_score = max(max_score, 1.0)
        
        total_reward += max_score
    
    # Return average score to avoid inflated scores with multiple predictions
    return total_reward / len(pred_tuples)

def hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate hit reward: intersection ratio between prediction and ground truth.

    Args:
        prediction: Model prediction text, may contain multiple sids.
        ground_truth: Ground truth text, may contain multiple sids.

    Returns:
        Hit reward: intersection count / prediction count.
    """
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0


    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return len(pred_set & gt_set) / len(pred_tuples)

def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate Pass@1 reward: whether the first sid after </think> hits ground truth.

    Args:
        prediction: Model prediction text.
        ground_truth: Ground truth text.

    Returns:
        1.0 if first sid is in ground truth, else 0.0.
    """
    # Extract content after </think>
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0

    pred_tuples = _extract_all_tuples(prediction)
    if not pred_tuples:
        return 0.0

    # Get the first predicted sid tuple
    first_pred_tuple = pred_tuples[0]

    gt_tuples = _extract_all_tuples(ground_truth)
    if not gt_tuples:
        return 0.0

    gt_set = set(gt_tuples)
    
    return float(first_pred_tuple in gt_set)

def pass_rate(prediction: str, ground_truth: str) -> float:
    """Calculate pass rate: whether prediction and ground truth have intersection.

    Args:
        prediction: Model prediction text, may contain multiple sids.
        ground_truth: Ground truth text, may contain multiple sids.

    Returns:
        1.0 if there is intersection, else 0.0.
    """
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    # Convert to set for intersection calculation
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    intersection_count = len(pred_set & gt_set)
    
    return float(intersection_count > 0)



def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    return rubric_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any] | str | None],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return rubric_compute_score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        **kwargs,
    )
