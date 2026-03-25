from __future__ import annotations

import torch

from benchmark.lead_decoding_generator import (
    LeadDecodingGenerator,
    LeadModeState,
    build_truncated_mixture_embedding,
    truncate_and_normalize_probs,
    update_lead_mode_state,
)


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self._inverse_vocab = {token_id: token for token, token_id in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._inverse_vocab[token_id] for token_id in token_ids]


def make_logits(vocab_size: int, scores: dict[int, float]) -> torch.Tensor:
    logits = torch.full((vocab_size,), -100.0, dtype=torch.float32)
    for token_id, value in scores.items():
        logits[token_id] = value
    return logits


def make_fake_generator(decode_mode: str = "lead") -> LeadDecodingGenerator:
    vocab = {}
    for idx in range(4):
        vocab[f"<s_a_{idx}>"] = idx
        vocab[f"<s_b_{idx}>"] = 10 + idx
        vocab[f"<s_c_{idx}>"] = 20 + idx

    generator = LeadDecodingGenerator.__new__(LeadDecodingGenerator)
    generator.decode_mode = decode_mode
    generator.latent_topk = 2
    generator.persistence_window = 3
    generator.max_switches = 5
    generator.num_return_sequences = 32
    generator.max_new_tokens = 3
    generator.device = "cpu"
    generator.tokenizer = FakeTokenizer(vocab)
    generator.embedding_weight = torch.eye(24, dtype=torch.float32)
    generator.allowed_token_ids_by_step = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([10, 11, 12, 13], dtype=torch.long),
        torch.tensor([20, 21, 22, 23], dtype=torch.long),
    ]

    def fake_advance_with_token(cache, token_id: int):
        if cache == "root":
            next_logits = make_logits(
                24,
                {
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                },
            )
            return ("after_first", token_id), next_logits

        if isinstance(cache, tuple) and cache[0] == "after_first":
            first_token = cache[1]
            base = 20 + ((first_token + token_id) % 4)
            next_logits = make_logits(
                24,
                {
                    base: 2.0,
                    20: 1.5,
                    21: 1.0,
                    22: 0.5,
                    23: 0.25,
                },
            )
            return ("after_second", first_token, token_id), next_logits

        raise AssertionError(f"Unexpected cache for token advance: {cache}")

    def fake_advance_with_embedding(cache, embedding: torch.Tensor):
        assert embedding.ndim == 1
        if cache == "root":
            next_logits = make_logits(
                24,
                {
                    10: 1.25,
                    11: 1.25,
                    12: 1.25,
                    13: 1.25,
                },
            )
            return ("after_first_latent",), next_logits

        if isinstance(cache, tuple) and cache[0] in {"after_first", "after_first_latent"}:
            next_logits = make_logits(
                24,
                {
                    20: 2.0,
                    21: 1.8,
                    22: 1.6,
                    23: 1.4,
                },
            )
            return ("after_second_latent",), next_logits

        raise AssertionError(f"Unexpected cache for latent advance: {cache}")

    generator._advance_with_token = fake_advance_with_token
    generator._advance_with_embedding = fake_advance_with_embedding
    return generator


def test_truncate_and_normalize_probs_sums_to_one():
    probs = torch.tensor([0.5, 0.3, 0.1, 0.1], dtype=torch.float32)
    norm_probs, indices = truncate_and_normalize_probs(probs, topk=2)

    assert indices.tolist() == [0, 1]
    assert torch.isclose(norm_probs.sum(), torch.tensor(1.0))


def test_build_truncated_mixture_embedding_uses_topk_support():
    probs = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32)
    allowed_token_ids = torch.tensor([2, 5, 7], dtype=torch.long)
    embedding_weight = torch.zeros((8, 3), dtype=torch.float32)
    embedding_weight[2] = torch.tensor([1.0, 0.0, 0.0])
    embedding_weight[5] = torch.tensor([0.0, 1.0, 0.0])
    embedding_weight[7] = torch.tensor([0.0, 0.0, 1.0])

    mixture = build_truncated_mixture_embedding(
        probs=probs,
        allowed_token_ids=allowed_token_ids,
        embedding_weight=embedding_weight,
        topk=2,
    )

    expected = torch.tensor([2 / 3, 1 / 3, 0.0], dtype=torch.float32)
    assert torch.allclose(mixture, expected, atol=1e-6)


def test_update_lead_mode_state_switches_in_both_directions():
    state = LeadModeState(mode="discrete", reference_entropy=0.5, steps_in_mode=3, switch_count=0)
    next_state, transition_mode, switched = update_lead_mode_state(
        state=state,
        entropy=1.2,
        max_entropy=1.5,
        persistence_window=3,
        max_switches=5,
        enable_lead=True,
    )

    assert switched is True
    assert transition_mode == "latent"
    assert next_state.mode == "latent"
    assert next_state.switch_count == 1

    back_state, back_mode, back_switched = update_lead_mode_state(
        state=next_state,
        entropy=0.3,
        max_entropy=1.5,
        persistence_window=3,
        max_switches=5,
        enable_lead=True,
    )

    assert back_switched is True
    assert back_mode == "discrete"
    assert back_state.mode == "discrete"
    assert back_state.switch_count == 2


def test_update_lead_mode_state_can_switch_on_initial_high_entropy_step():
    state = LeadModeState(mode="discrete", reference_entropy=None, steps_in_mode=3, switch_count=0)
    next_state, transition_mode, switched = update_lead_mode_state(
        state=state,
        entropy=0.9,
        max_entropy=1.3,
        persistence_window=3,
        max_switches=5,
        enable_lead=True,
    )

    assert switched is True
    assert transition_mode == "latent"
    assert next_state.mode == "latent"
    assert next_state.reference_entropy == 0.9
    assert next_state.switch_count == 1


def test_run_three_step_search_returns_32_candidates_with_lead_activity():
    generator = make_fake_generator(decode_mode="lead")
    root_logits = make_logits(
        24,
        {
            0: 4.0,
            1: 3.0,
            2: 2.0,
            3: 1.0,
        },
    )

    generations, logprobs, decode_stats = generator._run_three_step_search("root", root_logits)

    assert len(generations) == 32
    assert len(logprobs) == 32
    assert all(generation.startswith("<s_a_") for generation in generations)
    assert decode_stats["latent_step_ratio"] > 0.0


def test_run_three_step_search_step0_rerank_triggers_without_latent_rollout():
    generator = make_fake_generator(decode_mode="lead_step0_rerank")
    root_logits = make_logits(
        24,
        {
            0: 4.0,
            1: 3.0,
            2: 2.0,
            3: 1.0,
        },
    )

    generations, logprobs, decode_stats = generator._run_three_step_search("root", root_logits)

    assert len(generations) == 32
    assert len(logprobs) == 32
    assert decode_stats["triggered"] is True
    assert decode_stats["latent_step_ratio"] == 0.0
    assert decode_stats["switch_count"] > 0


def test_run_three_step_search_vanilla_has_zero_latent_ratio():
    generator = make_fake_generator(decode_mode="vanilla")
    root_logits = make_logits(
        24,
        {
            0: 4.0,
            1: 3.0,
            2: 2.0,
            3: 1.0,
        },
    )

    _, _, decode_stats = generator._run_three_step_search("root", root_logits)

    assert decode_stats["latent_step_ratio"] == 0.0
    assert decode_stats["switch_count"] == 0
