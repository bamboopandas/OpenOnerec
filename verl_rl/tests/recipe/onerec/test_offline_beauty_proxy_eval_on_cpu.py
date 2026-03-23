from __future__ import annotations

import json

import pandas as pd

from recipe.onerec.offline_beauty_proxy_eval import PromptExample, evaluate_with_judge


class FakeRubricJudge:
    def generate(self, prompt: str) -> str:
        if "correct target item" in prompt:
            return json.dumps(
                {
                    "criterion_results": [
                        {"criterion": "c1", "satisfied": True, "evidence": "match"},
                        {"criterion": "c2", "satisfied": True, "evidence": "match"},
                    ],
                    "rubric_score": 1.0,
                    "judge_reason": "good",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "criterion_results": [
                    {"criterion": "c1", "satisfied": False, "evidence": "miss"},
                    {"criterion": "c2", "satisfied": False, "evidence": "miss"},
                ],
                "rubric_score": 0.0,
                "judge_reason": "bad",
            },
            ensure_ascii=False,
        )


def test_evaluate_with_judge_reports_coverage_and_rerank_metrics(tmp_path):
    sidecar_path = tmp_path / "sidecar.parquet"
    rubric_dir = tmp_path / "rubrics"
    output_dir = tmp_path / "offline_eval"
    rubric_dir.mkdir()

    sid_wrong = "<|sid_begin|><s_a_1><s_b_1><s_c_1><s_d_1><|sid_end|>"
    sid_hit = "<|sid_begin|><s_a_1><s_b_1><s_c_1><s_d_2><|sid_end|>"
    sid_other = "<|sid_begin|><s_a_1><s_b_1><s_c_1><s_d_3><|sid_end|>"

    pd.DataFrame(
        [
            {"sid": sid_wrong, "pid": 1, "caption": "wrong item", "title": "wrong item", "categories": "Beauty > Makeup", "mapping_file": "beauty"},
            {"sid": sid_hit, "pid": 2, "caption": "correct target item", "title": "correct target item", "categories": "Beauty > Makeup", "mapping_file": "beauty"},
            {"sid": sid_other, "pid": 3, "caption": "other item", "title": "other item", "categories": "Beauty > Makeup", "mapping_file": "beauty"},
        ]
    ).to_parquet(sidecar_path, index=False)

    with (rubric_dir / "product.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"criterion": "c1", "weight": 1, "explanation": "e1"},
                {"criterion": "c2", "weight": 1, "explanation": "e2"},
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    examples = [
        PromptExample(
            uuid="ex-1",
            prompt_messages=[],
            ground_truth=sid_hit,
            extra_info={
                "schema_id": "product",
                "prompt_text": "prompt-1",
                "history_item_captions": ["history"],
                "history_product_captions": ["history"],
                "ground_truth_sids": [sid_hit],
                "ground_truth_captions": ["correct target item"],
            },
            raw_messages=[],
        ),
        PromptExample(
            uuid="ex-2",
            prompt_messages=[],
            ground_truth=sid_hit,
            extra_info={
                "schema_id": "product",
                "prompt_text": "prompt-2",
                "history_item_captions": ["history"],
                "history_product_captions": ["history"],
                "ground_truth_sids": [sid_hit],
                "ground_truth_captions": ["correct target item"],
            },
            raw_messages=[],
        ),
    ]

    candidate_pool = {
        "ex-1": {
            "uuid": "ex-1",
            "ground_truth": sid_hit,
            "extra_info": examples[0].extra_info,
            "prompt_messages": [],
            "candidates": [
                {"model_name": "retrieval", "strategy": "raw", "output": sid_wrong, "retrieval_score": 10.0},
                {"model_name": "retrieval", "strategy": "raw", "output": sid_hit, "retrieval_score": 9.0},
            ],
        },
        "ex-2": {
            "uuid": "ex-2",
            "ground_truth": sid_hit,
            "extra_info": examples[1].extra_info,
            "prompt_messages": [],
            "candidates": [
                {"model_name": "retrieval", "strategy": "raw", "output": sid_other, "retrieval_score": 8.0},
            ],
        },
    }

    result = evaluate_with_judge(
        examples,
        candidate_pool,
        judge_model_path="fake-judge",
        rubric_dir=str(rubric_dir),
        sidecar_index_path=str(sidecar_path),
        output_dir=str(output_dir),
        cache_path=str(output_dir / "judge_cache.sqlite"),
        k=10,
        judge_max_new_tokens=64,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=None,
        judge_impl=FakeRubricJudge(),
    )

    assert result["num_total_examples"] == 2
    assert result["num_rerank_examples"] == 1
    assert result["candidate_coverage"]["pass@10"] == 0.5
    assert result["candidate_coverage"]["pass@20"] == 0.5
    assert result["raw_ranking"]["pass@1"] == 0.0
    assert result["raw_ranking"]["pass@5"] == 1.0
    assert result["rubric_rerank"]["pass@1"] == 1.0
    assert result["rubric_rerank"]["pass@5"] == 1.0
    assert result["delta"]["pass@1"] == 1.0
    assert result["rubric_diagnostics"]["mean_hit_rubric_score"] > result["rubric_diagnostics"]["mean_miss_rubric_score"]
    assert (output_dir / "rubric_snapshot.json").exists()
    assert (output_dir / "judge_prompts.jsonl").exists()
