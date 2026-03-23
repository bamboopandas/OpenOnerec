from __future__ import annotations

from recipe.onerec.evaluate_beauty_proxy import evaluate_groups


def test_evaluate_groups_computes_beauty_proxy_metrics():
    sid1 = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    sid2 = "<|sid_begin|><s_a_1><s_b_1><s_c_2><|sid_end|>"
    groups = [
        {
            "input": "sample-1",
            "ground_truth": sid1,
            "outputs": [
                "<think>reason</think>" + sid1,
                "<think>reason</think>" + sid2,
            ],
            "rubric_scores": [0.9, 0.2],
            "objective_anchors": [1.0, 0.0],
            "unresolved_sid_ratios": [0.0, 0.0],
        },
        {
            "input": "sample-2",
            "ground_truth": sid2,
            "outputs": [
                "<think>reason</think>" + sid1,
                "<think>reason</think>" + sid2,
            ],
            "rubric_scores": [0.1, 0.8],
            "objective_anchors": [0.0, 1.0],
            "unresolved_sid_ratios": [0.0, 0.0],
        },
    ]

    summary, per_sample = evaluate_groups(groups, k=10, pass_ks=[1, 5, 10])

    assert len(per_sample) == 2
    assert summary["top1_hit"] == 0.5
    assert summary["pass@1"] == 0.5
    assert summary["pass@5"] == 1.0
    assert summary["pass@10"] == 1.0
    assert summary["beam_hit@10"] == 1.0
    assert summary["recall@10"] == 1.0
    assert round(summary["ndcg@10"], 6) == round((1.0 + (1.0 / 1.584962500721156)) / 2.0, 6)
    assert summary["hit_minus_miss_rubric_gap"] > 0.0


def test_evaluate_groups_supports_tuple_style_sid_outputs():
    sid1 = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    sid2 = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"
    groups = [
        {
            "input": "sample",
            "ground_truth": sid1,
            "outputs": [
                "<s_a_2><s_b_2><s_c_2>",
                "<s_a_1><s_b_1><s_c_1>",
            ],
            "rubric_scores": [0.1, 0.9],
        }
    ]

    summary, _ = evaluate_groups(groups, k=10, pass_ks=[1, 2])

    assert summary["pass@1"] == 0.0
    assert summary["pass@2"] == 1.0
