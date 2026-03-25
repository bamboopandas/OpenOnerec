from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from benchmark.base_generator import Generator
from benchmark.console import console, subhead_style_2
from benchmark.lead_utils import (
    LeadModeState,
    build_truncated_mixture_embedding,
    truncate_and_normalize_probs,
    update_lead_mode_state,
)


@dataclass(frozen=True)
class StepAnalysis:
    entropy: float
    log_probs: torch.Tensor
    allowed_token_ids: torch.Tensor
    next_state: LeadModeState
    transition_mode: str
    switched: bool


@dataclass(frozen=True)
class PartialCandidate:
    tokens: tuple[int, ...]
    score: float
    state: LeadModeState
    entropy_history: tuple[float, ...]
    latent_steps: int
    switch_events: int
    logits: torch.Tensor


@dataclass(frozen=True)
class SecondStepCandidate:
    tokens: tuple[int, int]
    score: float
    state: LeadModeState
    entropy_history: tuple[float, float]
    latent_steps: int
    switch_events: int
    first_token: int
    transition_mode: str


VANILLA_STATE = LeadModeState(mode="discrete", reference_entropy=None, steps_in_mode=0, switch_count=0)


def summarize_decode_stats(sample_decode_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not sample_decode_stats:
        return {
            "num_samples": 0,
            "mean_avg_entropy": 0.0,
            "mean_latent_step_ratio": 0.0,
            "latent_trigger_rate": 0.0,
            "mean_switch_count": 0.0,
            "mean_elapsed_sec": 0.0,
        }

    sample_values = list(sample_decode_stats.values())
    num_samples = len(sample_values)
    mean_avg_entropy = sum(item["avg_entropy"] for item in sample_values) / num_samples
    mean_latent_step_ratio = sum(item["latent_step_ratio"] for item in sample_values) / num_samples
    latent_trigger_rate = sum(float(item.get("triggered", item["latent_step_ratio"] > 0.0)) for item in sample_values) / num_samples
    mean_switch_count = sum(item["switch_count"] for item in sample_values) / num_samples
    mean_elapsed_sec = sum(item["elapsed_sec"] for item in sample_values) / num_samples

    return {
        "num_samples": num_samples,
        "mean_avg_entropy": mean_avg_entropy,
        "mean_latent_step_ratio": mean_latent_step_ratio,
        "latent_trigger_rate": latent_trigger_rate,
        "mean_switch_count": mean_switch_count,
        "mean_elapsed_sec": mean_elapsed_sec,
    }


class LeadDecodingGenerator(Generator):
    """
    Recommendation-only decoder for OpenOneRec.

    The generator assumes prompts already include `<|sid_begin|>` and decodes exactly three
    itemic tokens `<s_a_*>`, `<s_b_*>`, `<s_c_*>`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        decode_mode: str = "lead",
        latent_topk: int = 64,
        persistence_window: int = 3,
        max_switches: int = 5,
        num_return_sequences: int = 32,
        max_new_tokens: int = 3,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if decode_mode not in {"vanilla", "lead", "lead_step0_rerank"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        if max_new_tokens != 3:
            raise ValueError("LeadDecodingGenerator only supports max_new_tokens=3 for s_a/s_b/s_c decoding")

        self.model_name = model_name_or_path
        self.decode_mode = decode_mode
        self.latent_topk = latent_topk
        self.persistence_window = persistence_window
        self.max_switches = max_switches
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = {
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_weight = self.model.get_input_embeddings().weight
        self.allowed_token_ids_by_step = self._build_allowed_token_ids_by_step()
        self.num_params = sum(parameter.numel() for parameter in self.model.parameters())

        self.mfu_stats: dict[str, dict[str, list[float]]] = {}
        self.sample_decode_stats: dict[str, dict[str, Any]] = {}
        self.aggregate_decode_stats: dict[str, Any] = {}

    def _build_allowed_token_ids_by_step(self) -> list[torch.Tensor]:
        vocab = self.tokenizer.get_vocab()
        token_prefixes = ("<s_a_", "<s_b_", "<s_c_")
        allowed_ids_by_step = []

        for prefix in token_prefixes:
            ids = sorted(token_id for token, token_id in vocab.items() if token.startswith(prefix))
            if not ids:
                raise RuntimeError(f"Tokenizer does not contain any tokens starting with {prefix}")
            allowed_ids_by_step.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        return allowed_ids_by_step

    def _clone_cache(self, cache: Any) -> DynamicCache:
        if isinstance(cache, DynamicCache):
            return DynamicCache.from_legacy_cache(cache.to_legacy_cache())
        return DynamicCache.from_legacy_cache(cache)

    def _prepare_prompt(self, prompt_text: str) -> tuple[int, DynamicCache, torch.Tensor]:
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
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

        return int(input_ids.shape[1]), past_key_values, outputs.logits[:, -1, :].detach().squeeze(0)

    def _advance_with_token(self, cache: DynamicCache, token_id: int) -> tuple[DynamicCache, torch.Tensor]:
        cache_copy = self._clone_cache(cache)
        input_ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache_copy,
                use_cache=True,
                return_dict=True,
            )

        next_cache = outputs.past_key_values
        if not isinstance(next_cache, DynamicCache):
            next_cache = DynamicCache.from_legacy_cache(next_cache)

        return next_cache, outputs.logits[:, -1, :].detach().squeeze(0)

    def _advance_with_embedding(self, cache: DynamicCache, embedding: torch.Tensor) -> tuple[DynamicCache, torch.Tensor]:
        cache_copy = self._clone_cache(cache)
        inputs_embeds = embedding.to(self.embedding_weight.dtype).view(1, 1, -1)

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

        return next_cache, outputs.logits[:, -1, :].detach().squeeze(0)

    def _analyze_step(
        self,
        logits: torch.Tensor,
        step_idx: int,
        state: LeadModeState,
        enable_lead: Optional[bool] = None,
    ) -> StepAnalysis:
        allowed_token_ids = self.allowed_token_ids_by_step[step_idx]
        allowed_logits = logits.index_select(0, allowed_token_ids).float()
        log_probs = F.log_softmax(allowed_logits, dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum().item())
        if enable_lead is None:
            enable_lead = self.decode_mode == "lead"
        next_state, transition_mode, switched = update_lead_mode_state(
            state=state,
            entropy=entropy,
            max_entropy=math.log(float(allowed_token_ids.numel())),
            persistence_window=self.persistence_window,
            max_switches=self.max_switches,
            enable_lead=enable_lead,
        )
        return StepAnalysis(
            entropy=entropy,
            log_probs=log_probs,
            allowed_token_ids=allowed_token_ids,
            next_state=next_state,
            transition_mode=transition_mode,
            switched=switched,
        )

    def _tokens_to_generation(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self.tokenizer.convert_ids_to_tokens(list(token_ids)))

    def _replay_first_step(
        self,
        root_cache: DynamicCache,
        root_analysis: StepAnalysis,
        first_token: int,
        replay_cache: dict[str, tuple[DynamicCache, torch.Tensor]],
        force_discrete: bool = False,
    ) -> tuple[DynamicCache, torch.Tensor]:
        replay_key = "step0:latent" if root_analysis.transition_mode == "latent" and not force_discrete else f"step0:{first_token}"
        if replay_key in replay_cache:
            return replay_cache[replay_key]

        if root_analysis.transition_mode == "latent" and not force_discrete:
            root_probs = root_analysis.log_probs.exp()
            mixture = build_truncated_mixture_embedding(
                probs=root_probs,
                allowed_token_ids=root_analysis.allowed_token_ids,
                embedding_weight=self.embedding_weight,
                topk=self.latent_topk,
            )
            result = self._advance_with_embedding(root_cache, mixture)
        else:
            result = self._advance_with_token(root_cache, first_token)

        replay_cache[replay_key] = result
        return result

    def _run_three_step_search(
        self,
        root_cache: DynamicCache,
        root_logits: torch.Tensor,
    ) -> tuple[list[str], list[float], dict[str, Any]]:
        root_enable_lead = self.decode_mode in {"lead", "lead_step0_rerank"}
        later_enable_lead = self.decode_mode == "lead"
        use_step0_rerank = self.decode_mode == "lead_step0_rerank"

        initial_state = LeadModeState(mode="discrete", reference_entropy=None, steps_in_mode=self.persistence_window)
        root_analysis = self._analyze_step(root_logits, step_idx=0, state=initial_state, enable_lead=root_enable_lead)

        topk_step0 = min(self.num_return_sequences, root_analysis.log_probs.numel())
        top_scores0, top_indices0 = torch.topk(root_analysis.log_probs, k=topk_step0)

        step0_shared_logits: Optional[torch.Tensor] = None
        step0_rerank_probs: Optional[torch.Tensor] = None
        if root_analysis.transition_mode == "latent":
            root_probs = root_analysis.log_probs.exp()
            root_mixture = build_truncated_mixture_embedding(
                probs=root_probs,
                allowed_token_ids=root_analysis.allowed_token_ids,
                embedding_weight=self.embedding_weight,
                topk=self.latent_topk,
            )
            _, latent_logits1 = self._advance_with_embedding(root_cache, root_mixture)
            if use_step0_rerank:
                rerank_analysis = self._analyze_step(
                    latent_logits1,
                    step_idx=1,
                    state=VANILLA_STATE,
                    enable_lead=False,
                )
                step0_rerank_probs = rerank_analysis.log_probs.exp()
            else:
                step0_shared_logits = latent_logits1

        partials: list[PartialCandidate] = []
        for rank in range(topk_step0):
            first_token = int(root_analysis.allowed_token_ids[top_indices0[rank]].item())
            score0 = float(top_scores0[rank].item())
            if step0_shared_logits is None:
                _, next_logits = self._advance_with_token(root_cache, first_token)
            else:
                next_logits = step0_shared_logits

            partial_state = root_analysis.next_state if later_enable_lead else VANILLA_STATE
            switch_events = int(root_analysis.switched)
            latent_steps = int(root_analysis.transition_mode == "latent" and later_enable_lead)
            if step0_rerank_probs is not None:
                step1_analysis = self._analyze_step(
                    next_logits,
                    step_idx=1,
                    state=VANILLA_STATE,
                    enable_lead=False,
                )
                score0 += float((step0_rerank_probs * step1_analysis.log_probs).sum().item())

            partials.append(
                PartialCandidate(
                    tokens=(first_token,),
                    score=score0,
                    state=partial_state,
                    entropy_history=(root_analysis.entropy,),
                    latent_steps=latent_steps,
                    switch_events=switch_events,
                    logits=next_logits,
                )
            )

        second_step_candidates: list[SecondStepCandidate] = []
        for partial in partials:
            analysis1 = self._analyze_step(
                partial.logits,
                step_idx=1,
                state=partial.state,
                enable_lead=later_enable_lead,
            )
            topk_step1 = min(self.num_return_sequences, analysis1.log_probs.numel())
            top_scores1, top_indices1 = torch.topk(analysis1.log_probs, k=topk_step1)
            for rank in range(topk_step1):
                second_token = int(analysis1.allowed_token_ids[top_indices1[rank]].item())
                score1 = partial.score + float(top_scores1[rank].item())
                second_step_candidates.append(
                    SecondStepCandidate(
                        tokens=(partial.tokens[0], second_token),
                        score=score1,
                        state=analysis1.next_state,
                        entropy_history=(partial.entropy_history[0], analysis1.entropy),
                        latent_steps=partial.latent_steps + int(analysis1.transition_mode == "latent"),
                        switch_events=partial.switch_events + int(analysis1.switched),
                        first_token=partial.tokens[0],
                        transition_mode=analysis1.transition_mode,
                    )
                )

        second_step_candidates.sort(key=lambda item: item.score, reverse=True)
        selected_second_steps = second_step_candidates[: self.num_return_sequences]

        replay_cache: dict[str, tuple[DynamicCache, torch.Tensor]] = {}
        step1_latent_cache: dict[str, torch.Tensor] = {}
        final_candidates: list[tuple[tuple[int, int, int], float, tuple[float, float, float], int, int]] = []

        for candidate in selected_second_steps:
            cache1, logits1 = self._replay_first_step(
                root_cache=root_cache,
                root_analysis=root_analysis,
                first_token=candidate.first_token,
                replay_cache=replay_cache,
                force_discrete=use_step0_rerank,
            )
            analysis1 = self._analyze_step(
                logits1,
                step_idx=1,
                state=candidate.state,
                enable_lead=later_enable_lead,
            )

            if candidate.transition_mode == "latent":
                latent_key = f"step1:latent:{candidate.first_token}"
                if latent_key not in step1_latent_cache:
                    probs1 = analysis1.log_probs.exp()
                    mixture1 = build_truncated_mixture_embedding(
                        probs=probs1,
                        allowed_token_ids=analysis1.allowed_token_ids,
                        embedding_weight=self.embedding_weight,
                        topk=self.latent_topk,
                    )
                    _, logits2 = self._advance_with_embedding(cache1, mixture1)
                    step1_latent_cache[latent_key] = logits2
                logits2 = step1_latent_cache[latent_key]
            else:
                _, logits2 = self._advance_with_token(cache1, candidate.tokens[1])

            analysis2 = self._analyze_step(
                logits2,
                step_idx=2,
                state=candidate.state,
                enable_lead=later_enable_lead,
            )
            topk_step2 = min(self.num_return_sequences, analysis2.log_probs.numel())
            top_scores2, top_indices2 = torch.topk(analysis2.log_probs, k=topk_step2)

            for rank in range(topk_step2):
                third_token = int(analysis2.allowed_token_ids[top_indices2[rank]].item())
                score2 = candidate.score + float(top_scores2[rank].item())
                final_candidates.append(
                    (
                        (candidate.tokens[0], candidate.tokens[1], third_token),
                        score2,
                        (candidate.entropy_history[0], candidate.entropy_history[1], analysis2.entropy),
                        candidate.latent_steps + int(analysis2.transition_mode == "latent"),
                        candidate.switch_events + int(analysis2.switched),
                    )
                )

        final_candidates.sort(key=lambda item: item[1], reverse=True)
        selected_finals = final_candidates[: self.num_return_sequences]

        generations = [self._tokens_to_generation(candidate[0]) for candidate in selected_finals]
        logprobs = [candidate[1] for candidate in selected_finals]

        best_entropy_history = selected_finals[0][2] if selected_finals else (0.0, 0.0, 0.0)
        best_latent_steps = selected_finals[0][3] if selected_finals else 0
        best_switch_events = selected_finals[0][4] if selected_finals else 0

        decode_stats = {
            "avg_entropy": float(sum(best_entropy_history) / len(best_entropy_history)) if selected_finals else 0.0,
            "latent_step_ratio": float(best_latent_steps / self.max_new_tokens) if self.max_new_tokens else 0.0,
            "switch_count": int(best_switch_events),
            "triggered": bool(best_switch_events > 0),
        }
        return generations, logprobs, decode_stats

    def _generate_standard(self, prompts: dict[str, str], **kwargs: Any) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, dict[str, list[float]]]]:
        num_return_sequences = kwargs.get("num_return_sequences", self.num_return_sequences)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        if num_return_sequences != self.num_return_sequences:
            self.num_return_sequences = num_return_sequences
        if max_new_tokens != self.max_new_tokens:
            raise ValueError("LeadDecodingGenerator only supports max_new_tokens=3")

        results: dict[str, list[str]] = {}
        logprobs: dict[str, list[float]] = {}
        self.mfu_stats = {}
        self.sample_decode_stats = {}

        total_samples = len(prompts)
        for idx, (sample_id, prompt_text) in enumerate(prompts.items(), start=1):
            start_time = time.time()
            prompt_len, root_cache, root_logits = self._prepare_prompt(prompt_text)
            generations, sample_logprobs, decode_stats = self._run_three_step_search(root_cache, root_logits)
            elapsed = time.time() - start_time

            results[sample_id] = generations
            logprobs[sample_id] = sample_logprobs
            self.mfu_stats[sample_id] = {
                "input_tokens": [prompt_len],
                "output_tokens": [self.max_new_tokens],
                "times": [elapsed],
            }
            self.sample_decode_stats[sample_id] = {
                **decode_stats,
                "elapsed_sec": elapsed,
                "prompt_length": prompt_len,
            }

            if idx % 100 == 0 or idx == total_samples:
                console.print(
                    f"[{idx}/{total_samples}] decode_mode={self.decode_mode} avg_time={elapsed:.3f}s",
                    style=subhead_style_2,
                )

        self.aggregate_decode_stats = summarize_decode_stats(self.sample_decode_stats)
        return results, logprobs, self.mfu_stats

    def write_decode_stats(self, output_dir: str | Path, task_name: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "task_name": task_name,
            "decode_mode": self.decode_mode,
            "latent_topk": self.latent_topk,
            "persistence_window": self.persistence_window,
            "max_switches": self.max_switches,
            **self.aggregate_decode_stats,
        }

        with (output_dir / "decode_stats_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary_payload, file, indent=2, ensure_ascii=False)

        with (output_dir / "decode_stats_samples.json").open("w", encoding="utf-8") as file:
            json.dump(self.sample_decode_stats, file, indent=2, ensure_ascii=False)
