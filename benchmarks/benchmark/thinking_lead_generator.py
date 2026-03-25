from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from benchmark.base_generator import Generator, HfTransformersMixin
from benchmark.console import console, subhead_style_2
from benchmark.lead_utils import (
    INITIAL_LATENT_ENTROPY_RATIO_THRESHOLD,
    LeadModeState,
    build_full_vocab_truncated_mixture_embedding,
    update_lead_mode_state,
)


@dataclass(frozen=True)
class ThinkingCandidate:
    text: str
    avg_entropy: float
    latent_step_ratio: float
    switch_count: int
    triggered: bool
    thinking_tokens: int
    stopped_on_think: bool


def resolve_candidate_budget(candidate_budget: int) -> dict[str, int]:
    if candidate_budget == 32:
        return {
            "candidate_budget": 32,
            "num_return_thinking_sequences": 8,
            "num_beams": 4,
            "num_return_sequences": 32,
        }
    if candidate_budget == 128:
        return {
            "candidate_budget": 128,
            "num_return_thinking_sequences": 8,
            "num_beams": 16,
            "num_return_sequences": 128,
        }
    raise ValueError(f"Unsupported candidate_budget: {candidate_budget}")


def build_stage2_prompt(prompt_text: str, thinking_text: str, prompt_token: str) -> str:
    return f"{prompt_text}{thinking_text}</think>\n{prompt_token}"


def summarize_thinking_decode_stats(sample_decode_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not sample_decode_stats:
        return {
            "num_samples": 0,
            "mean_avg_entropy": 0.0,
            "latent_trigger_rate": 0.0,
            "mean_latent_step_ratio": 0.0,
            "mean_thinking_tokens": 0.0,
            "stop_on_think_rate": 0.0,
            "mean_stage1_time": 0.0,
            "mean_stage2_time": 0.0,
            "mean_switch_count": 0.0,
        }

    sample_values = list(sample_decode_stats.values())
    num_samples = len(sample_values)
    return {
        "num_samples": num_samples,
        "mean_avg_entropy": sum(item["mean_avg_entropy"] for item in sample_values) / num_samples,
        "latent_trigger_rate": sum(item["latent_trigger_rate"] for item in sample_values) / num_samples,
        "mean_latent_step_ratio": sum(item["mean_latent_step_ratio"] for item in sample_values) / num_samples,
        "mean_thinking_tokens": sum(item["mean_thinking_tokens"] for item in sample_values) / num_samples,
        "stop_on_think_rate": sum(item["stop_on_think_rate"] for item in sample_values) / num_samples,
        "mean_stage1_time": sum(item["stage1_time"] for item in sample_values) / num_samples,
        "mean_stage2_time": sum(item["stage2_time"] for item in sample_values) / num_samples,
        "mean_switch_count": sum(item["mean_switch_count"] for item in sample_values) / num_samples,
    }


class ThinkingLeadGenerator(HfTransformersMixin, Generator):
    """
    Apply LEAD only to stage-1 thinking generation.

    Stage-2 recommendation decoding remains vanilla HuggingFace generation starting from
    `</think>\\n<|sid_begin|>`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        stage1_decode_mode: str = "lead",
        latent_topk: int = 64,
        persistence_window: int = 3,
        max_switches: int = 5,
        initial_discrete_steps: Optional[int] = None,
        initial_entropy_ratio_threshold: float = INITIAL_LATENT_ENTROPY_RATIO_THRESHOLD,
        discrete_to_latent_margin: float = 0.0,
        latent_to_discrete_margin: float = 0.0,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        stage1_parallel_size: int = 2,
        stage2_batch_size: int = 2,
        **_: Any,
    ) -> None:
        super().__init__()
        if stage1_decode_mode not in {"vanilla", "lead"}:
            raise ValueError(f"Unsupported stage1_decode_mode: {stage1_decode_mode}")
        if latent_topk <= 0:
            raise ValueError("latent_topk must be positive")

        self.model_name = model_name_or_path
        self.stage1_decode_mode = stage1_decode_mode
        self.latent_topk = latent_topk
        self.persistence_window = persistence_window
        self.max_switches = max_switches
        self.initial_discrete_steps = persistence_window if initial_discrete_steps is None else max(0, initial_discrete_steps)
        self.initial_entropy_ratio_threshold = initial_entropy_ratio_threshold
        self.discrete_to_latent_margin = discrete_to_latent_margin
        self.latent_to_discrete_margin = latent_to_discrete_margin
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stage1_parallel_size = max(1, stage1_parallel_size)
        self.stage2_batch_size = max(1, stage2_batch_size)

        self.torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
        }.get(dtype, torch.bfloat16)

        console.print(f"Loading model from {model_name_or_path}...", style=subhead_style_2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        self.embedding_weight = self.model.get_input_embeddings().weight
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else self.pad_token_id
        self.stop_sequence = "</think>"
        self.stop_token_ids = self.tokenizer.encode(self.stop_sequence, add_special_tokens=False)
        self.num_params = sum(parameter.numel() for parameter in self.model.parameters())

        self.mfu_stats: dict[str, dict[str, list[float]]] = {}
        self.sample_decode_stats: dict[str, dict[str, Any]] = {}
        self.aggregate_decode_stats: dict[str, Any] = {}

    def _clone_cache(self, cache: Any) -> DynamicCache:
        if isinstance(cache, DynamicCache):
            return DynamicCache.from_legacy_cache(cache.to_legacy_cache())
        return DynamicCache.from_legacy_cache(cache)

    def _count_tokens(self, text: str) -> int:
        model_inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return int(model_inputs["input_ids"].shape[1])

    def _prepare_prompt_batch(self, prompt_text: str, batch_size: int) -> tuple[int, DynamicCache, torch.Tensor]:
        model_inputs = self.tokenizer(
            [prompt_text] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        if not isinstance(past_key_values, DynamicCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        prompt_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
        return prompt_len, past_key_values, outputs.logits[:, -1, :].detach()

    def _advance_with_embeddings(self, cache: DynamicCache, embeddings: torch.Tensor) -> tuple[DynamicCache, torch.Tensor]:
        cache_copy = self._clone_cache(cache)
        inputs_embeds = embeddings.to(self.embedding_weight.dtype).unsqueeze(1)

        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=cache_copy,
                use_cache=True,
                return_dict=True,
            )

        next_cache = outputs.past_key_values
        if not isinstance(next_cache, DynamicCache):
            next_cache = DynamicCache.from_legacy_cache(next_cache)

        return next_cache, outputs.logits[:, -1, :].detach()

    def _apply_sampling_constraints(
        self,
        logits: torch.Tensor,
        token_history: list[int],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> torch.Tensor:
        constrained = logits.clone()

        if repetition_penalty != 1.0 and token_history:
            seen_token_ids = set(token_history)
            for token_id in seen_token_ids:
                token_logit = constrained[token_id]
                constrained[token_id] = token_logit / repetition_penalty if token_logit > 0 else token_logit * repetition_penalty

        if (presence_penalty != 0.0 or frequency_penalty != 0.0) and token_history:
            counts = Counter(token_history)
            for token_id, token_count in counts.items():
                constrained[token_id] -= presence_penalty + frequency_penalty * token_count

        if temperature and temperature > 0:
            constrained = constrained / temperature

        if top_k and top_k > 0 and top_k < constrained.numel():
            threshold = torch.topk(constrained, k=top_k).values[-1]
            constrained = constrained.masked_fill(constrained < threshold, float("-inf"))

        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(constrained, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            constrained_mask = torch.zeros_like(constrained, dtype=torch.bool)
            constrained_mask.scatter_(0, sorted_indices, sorted_mask)
            constrained = constrained.masked_fill(constrained_mask, float("-inf"))

        return constrained

    def _sample_surface_token(
        self,
        raw_logits: torch.Tensor,
        token_history: list[int],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> tuple[int, float]:
        if temperature is not None and temperature <= 0:
            token_id = int(torch.argmax(raw_logits).item())
            log_prob = float(F.log_softmax(raw_logits.float(), dim=-1)[token_id].item())
            return token_id, log_prob

        filtered_logits = self._apply_sampling_constraints(
            logits=raw_logits.float(),
            token_history=token_history,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        token_probs = F.softmax(filtered_logits, dim=-1)
        token_id = int(torch.multinomial(token_probs, num_samples=1).item())
        log_prob = float(torch.log(token_probs[token_id].clamp_min(1e-12)).item())
        return token_id, log_prob

    def _token_ids_end_with_stop(self, token_ids: list[int]) -> bool:
        if not self.stop_token_ids or len(token_ids) < len(self.stop_token_ids):
            return False
        return token_ids[-len(self.stop_token_ids):] == self.stop_token_ids

    def _strip_stop_tokens(self, token_ids: list[int]) -> list[int]:
        if self._token_ids_end_with_stop(token_ids):
            return token_ids[: -len(self.stop_token_ids)]
        return token_ids

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _generate_thinking_batch(
        self,
        prompt_text: str,
        batch_size: int,
        max_new_thinking_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> tuple[list[ThinkingCandidate], int, int]:
        prompt_len, batch_cache, next_logits = self._prepare_prompt_batch(prompt_text, batch_size=batch_size)

        generated_token_ids: list[list[int]] = [[] for _ in range(batch_size)]
        entropy_history: list[list[float]] = [[] for _ in range(batch_size)]
        latent_steps = [0 for _ in range(batch_size)]
        switch_counts = [0 for _ in range(batch_size)]
        stopped_on_think = [False for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]
        states = [
            LeadModeState(mode="discrete", reference_entropy=None, steps_in_mode=self.initial_discrete_steps)
            for _ in range(batch_size)
        ]

        for _ in range(max_new_thinking_tokens):
            step_embeddings = []

            for batch_idx in range(batch_size):
                if finished[batch_idx]:
                    step_embeddings.append(self.embedding_weight[self.pad_token_id])
                    continue

                raw_logits = next_logits[batch_idx].float()
                raw_log_probs = F.log_softmax(raw_logits, dim=-1)
                raw_probs = raw_log_probs.exp()
                entropy = float(-(raw_probs * raw_log_probs).sum().item())
                entropy_history[batch_idx].append(entropy)

                state, transition_mode, switched = update_lead_mode_state(
                    state=states[batch_idx],
                    entropy=entropy,
                    max_entropy=math.log(float(raw_logits.numel())),
                    persistence_window=self.persistence_window,
                    max_switches=self.max_switches,
                    enable_lead=self.stage1_decode_mode == "lead",
                    initial_entropy_ratio_threshold=self.initial_entropy_ratio_threshold,
                    discrete_to_latent_margin=self.discrete_to_latent_margin,
                    latent_to_discrete_margin=self.latent_to_discrete_margin,
                )
                states[batch_idx] = state
                switch_counts[batch_idx] += int(switched)

                token_id, _ = self._sample_surface_token(
                    raw_logits=raw_logits,
                    token_history=generated_token_ids[batch_idx],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                generated_token_ids[batch_idx].append(token_id)

                if transition_mode == "latent":
                    latent_steps[batch_idx] += 1
                    next_embedding = build_full_vocab_truncated_mixture_embedding(
                        probs=raw_probs,
                        embedding_weight=self.embedding_weight,
                        topk=self.latent_topk,
                    )
                else:
                    next_embedding = self.embedding_weight[token_id]
                step_embeddings.append(next_embedding)

                if self._token_ids_end_with_stop(generated_token_ids[batch_idx]):
                    finished[batch_idx] = True
                    stopped_on_think[batch_idx] = True

            if all(finished):
                break

            embedding_batch = torch.stack(step_embeddings, dim=0)
            batch_cache, next_logits = self._advance_with_embeddings(batch_cache, embedding_batch)

        candidates = []
        total_output_tokens = 0
        for batch_idx in range(batch_size):
            stripped_token_ids = self._strip_stop_tokens(generated_token_ids[batch_idx])
            thinking_text = self._decode_token_ids(stripped_token_ids)
            thinking_token_count = len(stripped_token_ids)
            total_output_tokens += thinking_token_count

            avg_entropy = 0.0
            if entropy_history[batch_idx]:
                avg_entropy = float(sum(entropy_history[batch_idx]) / len(entropy_history[batch_idx]))
            candidates.append(
                ThinkingCandidate(
                    text=thinking_text,
                    avg_entropy=avg_entropy,
                    latent_step_ratio=float(latent_steps[batch_idx] / max(len(entropy_history[batch_idx]), 1)),
                    switch_count=switch_counts[batch_idx],
                    triggered=bool(switch_counts[batch_idx] > 0),
                    thinking_tokens=thinking_token_count,
                    stopped_on_think=stopped_on_think[batch_idx],
                )
            )

        return candidates, prompt_len, total_output_tokens

    def _generate_thinking_candidates(
        self,
        prompt_text: str,
        num_return_thinking_sequences: int,
        max_new_thinking_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> tuple[list[ThinkingCandidate], dict[str, float]]:
        start_time = time.time()
        all_candidates: list[ThinkingCandidate] = []
        prompt_len = 0
        total_output_tokens = 0

        remaining = num_return_thinking_sequences
        while remaining > 0:
            current_batch = min(self.stage1_parallel_size, remaining)
            batch_candidates, prompt_len, batch_output_tokens = self._generate_thinking_batch(
                prompt_text=prompt_text,
                batch_size=current_batch,
                max_new_thinking_tokens=max_new_thinking_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            all_candidates.extend(batch_candidates)
            total_output_tokens += batch_output_tokens
            remaining -= current_batch

        elapsed_sec = time.time() - start_time
        return all_candidates, {
            "input_tokens": prompt_len * num_return_thinking_sequences,
            "output_tokens": total_output_tokens,
            "elapsed_sec": elapsed_sec,
            "prompt_len": prompt_len,
        }

    def _sequence_scores_from_sampling(
        self,
        generated_ids: torch.Tensor,
        all_scores: tuple[torch.Tensor, ...],
        start_idx: int,
        end_idx: int,
    ) -> list[float]:
        if not all_scores:
            return []

        log_probs_by_step = [F.log_softmax(score.float(), dim=-1) for score in all_scores]
        sample_scores: list[float] = []
        for row_idx in range(start_idx, end_idx):
            total_score = 0.0
            for step_idx, log_probs in enumerate(log_probs_by_step):
                token_id = int(generated_ids[row_idx, step_idx].item())
                total_score += float(log_probs[row_idx, token_id].item())
            sample_scores.append(total_score)
        return sample_scores

    def _generate_standard(
        self,
        prompts: dict[str, str],
        **kwargs: Any,
    ) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, dict[str, list[float]]]]:
        results: dict[str, list[str]] = {}
        logprobs: dict[str, list[float]] = {}
        mfu_stats: dict[str, dict[str, list[float]]] = {}
        if not prompts:
            return results, logprobs, mfu_stats

        gen_kwargs, _ = self._build_sampling_params(**kwargs)
        gen_kwargs["pad_token_id"] = self.pad_token_id
        gen_kwargs["eos_token_id"] = self.eos_token_id
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True
        use_beam_search = "num_beams" in gen_kwargs
        per_prompt_returns = int(gen_kwargs.get("num_return_sequences", 1))

        prompt_items = list(prompts.items())
        for batch_start in range(0, len(prompt_items), self.stage2_batch_size):
            batch_items = prompt_items[batch_start : batch_start + self.stage2_batch_size]
            batch_ids = [sample_id for sample_id, _ in batch_items]
            batch_texts = [prompt_text for _, prompt_text in batch_items]

            model_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = model_inputs["input_ids"].to(self.device)
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            elapsed = time.time() - start_time

            sequences = outputs.sequences.detach().cpu()
            generated_ids = sequences[:, input_ids.shape[1] :]

            for sample_idx, sample_id in enumerate(batch_ids):
                start_idx = sample_idx * per_prompt_returns
                end_idx = start_idx + per_prompt_returns
                sample_generated_ids = generated_ids[start_idx:end_idx]
                results[sample_id] = [
                    self._decode_token_ids(token_row.tolist())
                    for token_row in sample_generated_ids
                ]

                if use_beam_search and hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                    logprobs[sample_id] = [
                        float(score)
                        for score in outputs.sequences_scores[start_idx:end_idx].detach().cpu().tolist()
                    ]
                else:
                    sample_scores = self._sequence_scores_from_sampling(
                        generated_ids=generated_ids,
                        all_scores=outputs.scores,
                        start_idx=start_idx,
                        end_idx=end_idx,
                    )
                    if sample_scores:
                        logprobs[sample_id] = sample_scores

                actual_prompt_tokens = int(attention_mask[sample_idx].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
                mfu_stats[sample_id] = {
                    "input_tokens": [actual_prompt_tokens],
                    "output_tokens": [int(sample_generated_ids.shape[1] * per_prompt_returns)],
                    "times": [elapsed / max(len(batch_ids), 1)],
                }

        return results, logprobs, mfu_stats

    def generate(
        self,
        prompts: dict[str, str],
        **kwargs: Any,
    ) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
        prompt_token = kwargs.get("prompt_token")
        enable_thinking = kwargs.get("enable_thinking", False)
        max_new_thinking_tokens = kwargs.get("max_new_thinking_tokens")
        num_return_thinking_sequences = int(kwargs.get("num_return_thinking_sequences", 1))
        total_num_return_sequences = int(kwargs.get("num_return_sequences", 1))
        num_beams = kwargs.get("num_beams")

        if not enable_thinking:
            raise ValueError("ThinkingLeadGenerator requires enable_thinking=True")
        if prompt_token is None:
            raise ValueError("ThinkingLeadGenerator requires prompt_token for stage-2 generation")
        if max_new_thinking_tokens is None:
            raise ValueError("ThinkingLeadGenerator requires max_new_thinking_tokens")

        use_beam_search = num_beams is not None
        if use_beam_search:
            beams_per_thinking = int(num_beams)
            expected_total = beams_per_thinking * num_return_thinking_sequences
            if expected_total != total_num_return_sequences:
                raise ValueError(
                    f"num_return_sequences ({total_num_return_sequences}) must equal "
                    f"num_return_thinking_sequences ({num_return_thinking_sequences}) * num_beams ({beams_per_thinking})"
                )
        else:
            beams_per_thinking = 1

        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.9))
        top_k = int(kwargs.get("top_k", -1))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        presence_penalty = float(kwargs.get("presence_penalty", 0.0))
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0))

        results: dict[str, list[str]] = {}
        logprobs: dict[str, list[float]] = {}
        self.mfu_stats = {}
        self.sample_decode_stats = {}

        total_samples = len(prompts)
        for idx, (sample_id, prompt_text) in enumerate(prompts.items(), start=1):
            thinking_candidates, stage1_stats = self._generate_thinking_candidates(
                prompt_text=prompt_text,
                num_return_thinking_sequences=num_return_thinking_sequences,
                max_new_thinking_tokens=int(max_new_thinking_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            stage2_prompts = {
                f"{sample_id}_thinking_{thinking_idx}": build_stage2_prompt(
                    prompt_text=prompt_text,
                    thinking_text=thinking_candidate.text,
                    prompt_token=prompt_token,
                )
                for thinking_idx, thinking_candidate in enumerate(thinking_candidates)
            }

            stage2_start_time = time.time()
            stage2_results, stage2_logprobs, stage2_mfu_stats = self._generate_standard(
                stage2_prompts,
                num_return_sequences=beams_per_thinking,
                num_beams=beams_per_thinking if use_beam_search else None,
                max_new_tokens=kwargs.get("max_new_tokens"),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                do_sample=not use_beam_search,
            )
            stage2_elapsed = time.time() - stage2_start_time

            final_generations: list[str] = []
            final_logprobs: list[float] = []
            stage2_input_tokens = 0
            stage2_output_tokens = 0
            for thinking_idx, thinking_candidate in enumerate(thinking_candidates):
                thinking_sample_id = f"{sample_id}_thinking_{thinking_idx}"
                sid_sequences = stage2_results.get(thinking_sample_id, [])
                for sid_sequence in sid_sequences:
                    final_generations.append(f"{thinking_candidate.text}</think>\n{prompt_token}{sid_sequence}")
                final_logprobs.extend(stage2_logprobs.get(thinking_sample_id, []))

                if thinking_sample_id in stage2_mfu_stats:
                    stage2_input_tokens += int(sum(stage2_mfu_stats[thinking_sample_id]["input_tokens"]))
                    stage2_output_tokens += int(sum(stage2_mfu_stats[thinking_sample_id]["output_tokens"]))

            results[sample_id] = final_generations
            if final_logprobs:
                logprobs[sample_id] = final_logprobs

            self.mfu_stats[sample_id] = {
                "input_tokens": [stage1_stats["input_tokens"], stage2_input_tokens],
                "output_tokens": [stage1_stats["output_tokens"], stage2_output_tokens],
                "times": [stage1_stats["elapsed_sec"], stage2_elapsed],
            }

            candidate_count = max(len(thinking_candidates), 1)
            self.sample_decode_stats[sample_id] = {
                "mean_avg_entropy": sum(candidate.avg_entropy for candidate in thinking_candidates) / candidate_count,
                "latent_trigger_rate": sum(float(candidate.triggered) for candidate in thinking_candidates) / candidate_count,
                "mean_latent_step_ratio": sum(candidate.latent_step_ratio for candidate in thinking_candidates) / candidate_count,
                "mean_thinking_tokens": sum(candidate.thinking_tokens for candidate in thinking_candidates) / candidate_count,
                "stop_on_think_rate": sum(float(candidate.stopped_on_think) for candidate in thinking_candidates) / candidate_count,
                "mean_switch_count": sum(candidate.switch_count for candidate in thinking_candidates) / candidate_count,
                "stage1_time": stage1_stats["elapsed_sec"],
                "stage2_time": stage2_elapsed,
                "prompt_length": stage1_stats["prompt_len"],
            }

            if idx % 25 == 0 or idx == total_samples:
                console.print(
                    f"[{idx}/{total_samples}] stage1_mode={self.stage1_decode_mode} stage1={stage1_stats['elapsed_sec']:.3f}s stage2={stage2_elapsed:.3f}s",
                    style=subhead_style_2,
                )

        self.aggregate_decode_stats = summarize_thinking_decode_stats(self.sample_decode_stats)
        return results, logprobs

    def write_decode_stats(
        self,
        output_dir: str | Path,
        task_name: str,
        candidate_budget: int,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "task_name": task_name,
            "stage1_decode_mode": self.stage1_decode_mode,
            "latent_topk": self.latent_topk,
            "persistence_window": self.persistence_window,
            "max_switches": self.max_switches,
            "initial_discrete_steps": self.initial_discrete_steps,
            "initial_entropy_ratio_threshold": self.initial_entropy_ratio_threshold,
            "discrete_to_latent_margin": self.discrete_to_latent_margin,
            "latent_to_discrete_margin": self.latent_to_discrete_margin,
            "candidate_budget": candidate_budget,
            **self.aggregate_decode_stats,
        }

        with (output_dir / "decode_stats_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary_payload, file, indent=2, ensure_ascii=False)

        with (output_dir / "decode_stats_samples.json").open("w", encoding="utf-8") as file:
            json.dump(self.sample_decode_stats, file, indent=2, ensure_ascii=False)
