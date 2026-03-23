from __future__ import annotations

import json

import pandas as pd

from recipe.onerec.rubric_reward import compute_rubric_only_score, compute_score_batch


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "candidate_items" in prompt
        return json.dumps(
            {
                "criterion_results": [
                    {"criterion": "c1", "satisfied": True, "evidence": "ok"},
                    {"criterion": "c2", "satisfied": False, "evidence": "skip"},
                    {"criterion": "c3", "satisfied": True, "evidence": "ok"},
                ],
                "rubric_score": 0.75,
                "judge_reason": "looks relevant",
            },
            ensure_ascii=False,
        )


def test_compute_score_batch_with_cache_and_sidecar(tmp_path):
    sidecar_path = tmp_path / "sidecar_index.parquet"
    cache_path = tmp_path / "reward_cache.sqlite"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    pd.DataFrame(
        [
            {
                "sid": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
                "pid": 1,
                "caption": "动作电影剪辑",
                "mapping_file": "video_pid2sid.parquet",
            }
        ]
    ).to_parquet(sidecar_path, index=False)

    with (rubric_dir / "video.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"criterion": "c1", "weight": 1, "explanation": "e1"},
                {"criterion": "c2", "weight": 1, "explanation": "e2"},
                {"criterion": "c3", "weight": 2, "explanation": "e3"},
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    extra_info = {
        "schema_id": "video",
        "task_name": "video",
        "task_variant": "video",
        "prompt_text": "根据历史推荐视频",
        "history_item_captions": ["动作电影剪辑"],
        "history_ad_captions": [],
        "history_product_captions": [],
        "ground_truth_captions": ["动作电影剪辑"],
        "query_text": "",
        "interaction_type": "",
    }
    prediction = "<think>用户喜欢动作内容，所以优先推荐同主题视频。</think><|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
    ground_truth = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"

    judge = FakeJudge()
    first_results = compute_score_batch(
        data_sources=["RecIF_VideoRec"],
        solution_strs=[prediction],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        judge_impl=judge,
        rubric_dir=str(rubric_dir),
        sidecar_index_path=str(sidecar_path),
        cache_path=str(cache_path),
        max_workers=1,
    )
    assert judge.calls == 1
    assert len(first_results) == 1
    first = first_results[0]
    assert first["rubric_score"] == 0.75
    assert first["objective_anchor"] == 1.0
    assert first["score"] == 0.875
    assert first["cache_hit"] == 0.0
    assert first["unresolved_sid_ratio"] == 0.0
    assert first["pred"] == ground_truth

    second_results = compute_score_batch(
        data_sources=["RecIF_VideoRec"],
        solution_strs=[prediction],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        judge_impl=judge,
        rubric_dir=str(rubric_dir),
        sidecar_index_path=str(sidecar_path),
        cache_path=str(cache_path),
        max_workers=1,
    )
    assert judge.calls == 1
    assert second_results[0]["cache_hit"] == 1.0
    assert second_results[0]["score"] == first["score"]


def test_compute_score_batch_handles_unresolved_sid_without_judge(tmp_path):
    sidecar_path = tmp_path / "empty_sidecar.parquet"
    pd.DataFrame(columns=["sid", "pid", "caption", "mapping_file"]).to_parquet(sidecar_path, index=False)

    result = compute_score_batch(
        data_sources=["RecIF_VideoRec"],
        solution_strs=["<think>test test test test test</think><|sid_begin|><s_a_9><s_b_9><s_c_9><|sid_end|>"],
        ground_truths=["<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"],
        extra_infos=[{"schema_id": "video", "prompt_text": "prompt"}],
        judge_base_url="none",
        judge_model="none",
        sidecar_index_path=str(sidecar_path),
        max_workers=1,
    )[0]

    assert result["rubric_score"] == 0.0
    assert result["unresolved_sid_ratio"] == 1.0
    assert result["cache_hit"] == 0.0


def test_compute_score_batch_supports_objective_only_mode(tmp_path):
    sidecar_path = tmp_path / "sidecar_index.parquet"
    pd.DataFrame(
        [
            {
                "sid": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
                "pid": 1,
                "caption": "动作电影剪辑",
                "mapping_file": "video_pid2sid.parquet",
            }
        ]
    ).to_parquet(sidecar_path, index=False)

    judge = FakeJudge()
    result = compute_score_batch(
        data_sources=["RecIF_VideoRec"],
        solution_strs=["<think>test test test test test</think><|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"],
        ground_truths=["<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"],
        extra_infos=[{"schema_id": "video", "prompt_text": "prompt"}],
        sidecar_index_path=str(sidecar_path),
        reward_mode="objective_only",
        judge_impl=judge,
        max_workers=1,
    )[0]

    assert judge.calls == 0
    assert result["score"] == result["objective_anchor"]
    assert result["rubric_score"] == 0.0
    assert result["rubric_applied"] == 0.0


def test_compute_rubric_only_score_skips_objective_metrics(tmp_path):
    sidecar_path = tmp_path / "sidecar_index.parquet"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    pd.DataFrame(
        [
            {
                "sid": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
                "pid": 1,
                "caption": "动作电影剪辑",
                "mapping_file": "video_pid2sid.parquet",
            }
        ]
    ).to_parquet(sidecar_path, index=False)
    with (rubric_dir / "video.json").open("w", encoding="utf-8") as handle:
        json.dump([{"criterion": "c1", "weight": 1, "explanation": "e1"}], handle, ensure_ascii=False)

    judge = FakeJudge()
    result = compute_rubric_only_score(
        solution_str="<think>test test test test test</think><|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
        extra_info={"schema_id": "video", "prompt_text": "prompt"},
        sidecar_index_path=str(sidecar_path),
        rubric_dir=str(rubric_dir),
        judge_impl=judge,
    )

    assert judge.calls == 1
    assert result["rubric_score"] == 0.75
    assert "score" not in result
    assert "objective_anchor" not in result
