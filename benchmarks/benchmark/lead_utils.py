from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


INITIAL_LATENT_ENTROPY_RATIO_THRESHOLD = 0.64


@dataclass(frozen=True)
class LeadModeState:
    mode: str = "discrete"
    reference_entropy: Optional[float] = None
    steps_in_mode: int = 0
    switch_count: int = 0


def truncate_and_normalize_probs(probs: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    topk = min(topk, probs.numel())
    if topk <= 0:
        raise ValueError("topk must be positive")
    top_probs, top_indices = torch.topk(probs, k=topk)
    norm_probs = top_probs / top_probs.sum().clamp_min(1e-12)
    return norm_probs, top_indices


def build_truncated_mixture_embedding(
    probs: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    norm_probs, top_indices = truncate_and_normalize_probs(probs, topk=topk)
    selected_token_ids = allowed_token_ids.index_select(0, top_indices)
    selected_embeddings = embedding_weight.index_select(0, selected_token_ids)
    mixture = (norm_probs.to(selected_embeddings.dtype).unsqueeze(-1) * selected_embeddings).sum(dim=0)
    return mixture


def build_full_vocab_truncated_mixture_embedding(
    probs: torch.Tensor,
    embedding_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    norm_probs, top_indices = truncate_and_normalize_probs(probs, topk=topk)
    selected_embeddings = embedding_weight.index_select(0, top_indices)
    mixture = (norm_probs.to(selected_embeddings.dtype).unsqueeze(-1) * selected_embeddings).sum(dim=0)
    return mixture


def update_lead_mode_state(
    state: LeadModeState,
    entropy: float,
    max_entropy: float,
    persistence_window: int,
    max_switches: int,
    enable_lead: bool,
    initial_entropy_ratio_threshold: float = INITIAL_LATENT_ENTROPY_RATIO_THRESHOLD,
    discrete_to_latent_margin: float = 0.0,
    latent_to_discrete_margin: float = 0.0,
) -> tuple[LeadModeState, str, bool]:
    entropy_ratio = entropy / max(max_entropy, 1e-12)

    if state.reference_entropy is None:
        if enable_lead and state.steps_in_mode >= persistence_window and entropy_ratio >= initial_entropy_ratio_threshold:
            switched_state = LeadModeState(
                mode="latent",
                reference_entropy=entropy,
                steps_in_mode=1,
                switch_count=state.switch_count + 1,
            )
            return switched_state, "latent", True

        initialized_state = LeadModeState(
            mode="discrete",
            reference_entropy=entropy,
            steps_in_mode=state.steps_in_mode + 1,
            switch_count=state.switch_count,
        )
        return initialized_state, "discrete", False

    if not enable_lead:
        vanilla_state = LeadModeState(
            mode="discrete",
            reference_entropy=state.reference_entropy,
            steps_in_mode=state.steps_in_mode + 1,
            switch_count=state.switch_count,
        )
        return vanilla_state, "discrete", False

    if state.switch_count >= max_switches:
        frozen_state = LeadModeState(
            mode=state.mode,
            reference_entropy=state.reference_entropy,
            steps_in_mode=state.steps_in_mode + 1,
            switch_count=state.switch_count,
        )
        return frozen_state, state.mode, False

    if state.mode == "discrete":
        if state.steps_in_mode >= persistence_window and entropy > state.reference_entropy + discrete_to_latent_margin:
            switched_state = LeadModeState(
                mode="latent",
                reference_entropy=entropy,
                steps_in_mode=1,
                switch_count=state.switch_count + 1,
            )
            return switched_state, "latent", True

        stayed_state = LeadModeState(
            mode="discrete",
            reference_entropy=state.reference_entropy,
            steps_in_mode=state.steps_in_mode + 1,
            switch_count=state.switch_count,
        )
        return stayed_state, "discrete", False

    if entropy < state.reference_entropy - latent_to_discrete_margin:
        switched_state = LeadModeState(
            mode="discrete",
            reference_entropy=entropy,
            steps_in_mode=1,
            switch_count=state.switch_count + 1,
        )
        return switched_state, "discrete", True

    stayed_state = LeadModeState(
        mode="latent",
        reference_entropy=state.reference_entropy,
        steps_in_mode=state.steps_in_mode + 1,
        switch_count=state.switch_count,
    )
    return stayed_state, "latent", False
