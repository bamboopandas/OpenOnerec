from __future__ import annotations

import torch

from benchmark.lead_utils import LeadModeState, update_lead_mode_state
from benchmark.thinking_lead_generator import (
    ThinkingCandidate,
    ThinkingLeadGenerator,
    build_stage2_prompt,
    resolve_candidate_budget,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self._id_to_text = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "<",
            5: ">",
            6: "_",
            7: "!",
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._id_to_text[token_id] for token_id in token_ids)


def make_fake_thinking_generator(
    stage1_decode_mode: str = "lead",
    initial_discrete_steps: int = 3,
    initial_entropy_ratio_threshold: float = 0.64,
    discrete_to_latent_margin: float = 0.0,
    latent_to_discrete_margin: float = 0.0,
) -> ThinkingLeadGenerator:
    generator = ThinkingLeadGenerator.__new__(ThinkingLeadGenerator)
    generator.stage1_decode_mode = stage1_decode_mode
    generator.latent_topk = 2
    generator.persistence_window = 3
    generator.max_switches = 5
    generator.initial_discrete_steps = initial_discrete_steps
    generator.initial_entropy_ratio_threshold = initial_entropy_ratio_threshold
    generator.discrete_to_latent_margin = discrete_to_latent_margin
    generator.latent_to_discrete_margin = latent_to_discrete_margin
    generator.stage1_parallel_size = 2
    generator.stage2_batch_size = 2
    generator.device = "cpu"
    generator.pad_token_id = 6
    generator.eos_token_id = 7
    generator.stop_token_ids = [4, 5]
    generator.embedding_weight = torch.eye(8, dtype=torch.float32)
    generator.tokenizer = FakeTokenizer()
    generator.embedding_history = []
    generator.mfu_stats = {}
    generator.sample_decode_stats = {}
    generator.aggregate_decode_stats = {}

    def fake_prepare_prompt_batch(prompt_text: str, batch_size: int):
        root_logits = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, -20.0, -20.0, -20.0, -20.0]
                for _ in range(batch_size)
            ],
            dtype=torch.float32,
        )
        return 7, {"step": 0, "batch_size": batch_size}, root_logits

    def fake_advance_with_embeddings(cache, embeddings: torch.Tensor):
        generator.embedding_history.append(embeddings.clone())
        step = cache["step"] + 1

        if step == 1:
            logits = torch.tensor(
                [
                    [0.2, 0.1, 3.0, 0.0, -20.0, -20.0, -20.0, -20.0]
                    for _ in range(cache["batch_size"])
                ],
                dtype=torch.float32,
            )
        elif step == 2:
            logits = torch.tensor(
                [
                    [-20.0, -20.0, 0.0, 0.0, 5.0, -20.0, -20.0, -20.0]
                    for _ in range(cache["batch_size"])
                ],
                dtype=torch.float32,
            )
        else:
            logits = torch.tensor(
                [
                    [-20.0, -20.0, 0.0, 0.0, -20.0, 5.0, -20.0, -20.0]
                    for _ in range(cache["batch_size"])
                ],
                dtype=torch.float32,
            )
        return {"step": step, "batch_size": cache["batch_size"]}, logits

    generator._prepare_prompt_batch = fake_prepare_prompt_batch
    generator._advance_with_embeddings = fake_advance_with_embeddings
    return generator


def test_resolve_candidate_budget_matches_expected_configs():
    assert resolve_candidate_budget(32) == {
        "candidate_budget": 32,
        "num_return_thinking_sequences": 8,
        "num_beams": 4,
        "num_return_sequences": 32,
    }
    assert resolve_candidate_budget(128) == {
        "candidate_budget": 128,
        "num_return_thinking_sequences": 8,
        "num_beams": 16,
        "num_return_sequences": 128,
    }


def test_build_stage2_prompt_uses_think_boundary():
    prompt = build_stage2_prompt("PROMPT", "reasoning", "<|sid_begin|>")
    assert prompt == "PROMPTreasoning</think>\n<|sid_begin|>"


def test_generate_thinking_candidates_vanilla_never_uses_latent():
    generator = make_fake_thinking_generator(stage1_decode_mode="vanilla")

    candidates, stats = generator._generate_thinking_candidates(
        prompt_text="PROMPT",
        num_return_thinking_sequences=2,
        max_new_thinking_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    assert len(candidates) == 2
    assert all(candidate.text == "AC" for candidate in candidates)
    assert all(candidate.latent_step_ratio == 0.0 for candidate in candidates)
    assert all(candidate.triggered is False for candidate in candidates)
    assert all(candidate.stopped_on_think is True for candidate in candidates)
    assert stats["input_tokens"] == 14
    assert stats["output_tokens"] == 4

    for embedding_batch in generator.embedding_history:
        assert torch.allclose(embedding_batch.max(dim=1).values, torch.ones(embedding_batch.shape[0]))


def test_generate_thinking_candidates_lead_uses_latent_embeddings():
    generator = make_fake_thinking_generator(stage1_decode_mode="lead")

    candidates, _ = generator._generate_thinking_candidates(
        prompt_text="PROMPT",
        num_return_thinking_sequences=2,
        max_new_thinking_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    assert len(candidates) == 2
    assert any(candidate.latent_step_ratio > 0.0 for candidate in candidates)
    assert any(candidate.triggered for candidate in candidates)
    assert any((embedding_batch.max(dim=1).values < 0.999).any().item() for embedding_batch in generator.embedding_history)


def test_generate_thinking_candidates_conservative_settings_delay_latent_switch():
    generator = make_fake_thinking_generator(
        stage1_decode_mode="lead",
        initial_discrete_steps=0,
        initial_entropy_ratio_threshold=0.99,
        discrete_to_latent_margin=10.0,
    )

    candidates, _ = generator._generate_thinking_candidates(
        prompt_text="PROMPT",
        num_return_thinking_sequences=2,
        max_new_thinking_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    assert len(candidates) == 2
    assert all(candidate.latent_step_ratio == 0.0 for candidate in candidates)
    assert all(candidate.triggered is False for candidate in candidates)
    for embedding_batch in generator.embedding_history:
        assert torch.allclose(embedding_batch.max(dim=1).values, torch.ones(embedding_batch.shape[0]))


def test_update_lead_mode_state_requires_true_warmup_before_switch():
    state = LeadModeState(mode="discrete", reference_entropy=None, steps_in_mode=0)

    state, mode, switched = update_lead_mode_state(
        state=state,
        entropy=1.0,
        max_entropy=2.0,
        persistence_window=3,
        max_switches=5,
        enable_lead=True,
        initial_entropy_ratio_threshold=0.75,
        discrete_to_latent_margin=0.0,
        latent_to_discrete_margin=0.0,
    )

    assert mode == "discrete"
    assert switched is False
    assert state.steps_in_mode == 1
    assert state.reference_entropy == 1.0

    state, mode, switched = update_lead_mode_state(
        state=state,
        entropy=1.3,
        max_entropy=2.0,
        persistence_window=3,
        max_switches=5,
        enable_lead=True,
        initial_entropy_ratio_threshold=0.75,
        discrete_to_latent_margin=0.0,
        latent_to_discrete_margin=0.0,
    )

    assert mode == "discrete"
    assert switched is False
    assert state.steps_in_mode == 2


def test_generate_builds_stage2_prompts_and_keeps_stage2_vanilla():
    generator = ThinkingLeadGenerator.__new__(ThinkingLeadGenerator)
    generator.stage1_decode_mode = "lead"
    generator.mfu_stats = {}
    generator.sample_decode_stats = {}
    generator.aggregate_decode_stats = {}

    def fake_generate_thinking_candidates(**kwargs):
        assert kwargs["num_return_thinking_sequences"] == 1
        return (
            [
                ThinkingCandidate(
                    text="reason",
                    avg_entropy=0.9,
                    latent_step_ratio=0.5,
                    switch_count=1,
                    triggered=True,
                    thinking_tokens=6,
                    stopped_on_think=True,
                )
            ],
            {
                "input_tokens": 5,
                "output_tokens": 6,
                "elapsed_sec": 0.1,
                "prompt_len": 5,
            },
        )

    def fake_generate_standard(prompts, **kwargs):
        assert prompts == {"sample_0_thinking_0": "PROMPTreason</think>\n<|sid_begin|>"}
        assert kwargs["num_beams"] == 4
        assert kwargs["num_return_sequences"] == 4
        return (
            {"sample_0_thinking_0": ["<s_a_1><s_b_1><s_c_1>"] * 4},
            {"sample_0_thinking_0": [0.1, 0.2, 0.3, 0.4]},
            {"sample_0_thinking_0": {"input_tokens": [9], "output_tokens": [12], "times": [0.2]}},
        )

    generator._generate_thinking_candidates = fake_generate_thinking_candidates
    generator._generate_standard = fake_generate_standard

    results, logprobs = generator.generate(
        {"sample_0": "PROMPT"},
        enable_thinking=True,
        prompt_token="<|sid_begin|>",
        max_new_thinking_tokens=16,
        num_return_thinking_sequences=1,
        num_return_sequences=4,
        num_beams=4,
        max_new_tokens=3,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    assert len(results["sample_0"]) == 4
    assert all(generation == "reason</think>\n<|sid_begin|><s_a_1><s_b_1><s_c_1>" for generation in results["sample_0"])
    assert logprobs["sample_0"] == [0.1, 0.2, 0.3, 0.4]
    assert generator.mfu_stats["sample_0"]["times"][0] == 0.1
    assert generator.mfu_stats["sample_0"]["times"][1] >= 0.0
