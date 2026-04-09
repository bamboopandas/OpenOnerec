from __future__ import annotations

import json
import re

import pandas as pd

from recipe.onerec.rubric_reward import (
    build_single_rule_judge_prompt,
    build_pairwise_judge_prompt,
    compute_pairwise_rubric_rerank,
    compute_rubric_only_score,
    compute_score_batch,
    parse_listwise_judge_response,
    parse_pairwise_judge_response,
)


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


class FakePairwiseJudge:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        a_hit = bool(re.search(r'"A".*?命中商品标题', prompt, flags=re.DOTALL))
        b_hit = bool(re.search(r'"B".*?命中商品标题', prompt, flags=re.DOTALL))
        if a_hit and not b_hit:
            winner = "A"
        elif b_hit and not a_hit:
            winner = "B"
        else:
            winner = "tie"
        return json.dumps(
            {
                "rule_votes": [
                    {"rule_id": "history_product_match", "vote": winner, "evidence": "商品更像历史"},
                    {"rule_id": "history_video_bridge", "vote": winner, "evidence": "视频主题更近"},
                    {"rule_id": "semantic_quality", "vote": "tie", "evidence": ""},
                    {"rule_id": "raw_rank_anchor", "vote": "tie", "evidence": ""},
                ],
                "final_vote": winner,
                "short_reason": "pairwise",
            },
            ensure_ascii=False,
        )


class FakeSingleRuleJudge:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if "推荐历史摘要器" in prompt:
            return json.dumps(
                {
                    "dominant_topics": ["校园互动内容", "人物轻剧情节"],
                    "recent_patterns": ["近期偏好轻松人物视频"],
                    "supporting_history_ids": [1, 2],
                },
                ensure_ascii=False,
            )
        a_hit = "命中视频标题" in prompt and '"candidate_a"' in prompt
        b_hit = "命中视频标题" in prompt and '"candidate_b"' in prompt
        if a_hit and not b_hit:
            vote = "A"
        elif b_hit and not a_hit:
            vote = "B"
        else:
            vote = "tie"
        return json.dumps(
            {
                "rule_id": "history_topic_match" if "history_topic_match" in prompt else "query_intent_match",
                "vote": vote,
                "history_basis": "最近历史主要是校园互动和轻松人物内容。",
                "candidate_a_basis": "A 是命中视频标题，和历史更连贯。" if vote == "A" else "A 与历史主题接近程度一般。",
                "candidate_b_basis": "B 是普通视频标题，和历史较远。" if vote == "A" else "B 与历史主题接近程度一般。",
                "tie_analysis": "如果两者都只弱相关才应 tie；当前差异足够明显。" if vote != "tie" else "两者都没有明显优势，因此 tie。",
                "decision_rationale": "按该规则 A 更优。" if vote == "A" else ("按该规则 B 更优。" if vote == "B" else "按该规则应返回 tie。"),
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


def test_parse_listwise_judge_response_recovers_partial_ranking():
    raw_response = """
    {
      "ranking": [
        {"candidate_index": 2, "score": 0.9, "reason": "更相关"},
        {"candidate_index": 1, "score": 0.3, "reason": "次之"}
      ],
      "judge_reason": "ok"
    """

    result = parse_listwise_judge_response(raw_response, candidate_count=3)

    assert [item["candidate_index"] for item in result.ranking] == [2, 1, 3]
    assert result.ranking[0]["score"] == 0.9
    assert result.judge_reason == "ok"


def test_parse_pairwise_judge_response_recovers_winner_and_votes():
    raw_response = """
    ```json
    {
      "rule_votes": [
        {"rule_id": "history_product_match", "vote": "B", "evidence": "更像历史商品"}
      ],
      "final_vote": "B",
      "short_reason": "更符合历史兴趣"
    }
    ```
    """

    result = parse_pairwise_judge_response(raw_response)

    assert result.winner == "B"
    assert result.confidence == 1.0
    assert result.judge_reason == "更符合历史兴趣"
    assert result.criterion_votes[0]["criterion_id"] == "history_product_match"
    assert result.criterion_votes[0]["winner"] == "B"


def test_build_pairwise_judge_prompt_includes_preference_summary_for_product():
    prompt = build_pairwise_judge_prompt(
        schema_id="product",
        rubric=[{"criterion": "c1", "weight": 1, "explanation": "e1"}],
        extra_info={
            "task_name": "product",
            "query_text": "以下是用户的观看和购物记录。用户当前的偏好涉及学生, 学习, 文具等主题。基于以上记录，推荐用户可能感兴趣并点击的商品。",
            "history_item_captions": ["校园学习视频"],
            "history_product_captions": ["学生文具套装"],
            "history_ad_captions": [],
            "user_profile_text": "你是一个跨域推荐专家。",
        },
        candidate_a={
            "candidate_index": 1,
            "raw_rank": 1,
            "predicted_items": [{"pid": 1, "caption": "学生文具礼盒"}],
            "unresolved_sid_ratio": 0.0,
        },
        candidate_b={
            "candidate_index": 2,
            "raw_rank": 2,
            "predicted_items": [{"pid": 2, "caption": "校园恋爱短视频"}],
            "unresolved_sid_ratio": 0.0,
        },
    )

    assert '"criterion_id": "query_summary_match"' in prompt
    assert '"preference_summary": "用户当前的偏好涉及学生, 学习, 文具等主题。"' in prompt
    assert '"A"' in prompt and '"B"' in prompt


def test_compute_pairwise_rubric_rerank_uses_swap_and_aggregation(tmp_path):
    sidecar_path = tmp_path / "sidecar_index.parquet"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"
    third_sid = "<|sid_begin|><s_a_3><s_b_3><s_c_3><|sid_end|>"

    pd.DataFrame(
        [
            {"sid": hit_sid, "pid": 101, "caption": "命中商品标题", "mapping_file": "product_pid2sid.parquet"},
            {"sid": miss_sid, "pid": 102, "caption": "无关商品标题", "mapping_file": "product_pid2sid.parquet"},
            {"sid": third_sid, "pid": 103, "caption": "普通商品标题", "mapping_file": "product_pid2sid.parquet"},
        ]
    ).to_parquet(sidecar_path, index=False)
    with (rubric_dir / "product.json").open("w", encoding="utf-8") as handle:
        json.dump([{"criterion": "c1", "weight": 1, "explanation": "e1"}], handle, ensure_ascii=False)

    result = compute_pairwise_rubric_rerank(
        predictions=[miss_sid, hit_sid, third_sid],
        extra_info={
            "schema_id": "product",
            "task_name": "product",
            "prompt_text": "推荐商品",
            "history_item_captions": ["护肤视频"],
            "history_product_captions": ["历史商品标题"],
        },
        sidecar_lookup={
            hit_sid: {"pid": 101, "caption": "命中商品标题"},
            miss_sid: {"pid": 102, "caption": "无关商品标题"},
            third_sid: {"pid": 103, "caption": "普通商品标题"},
        },
        rubric_dir=str(rubric_dir),
        judge_impl=FakePairwiseJudge(),
        pairwise_top_n=3,
    )

    reranked = sorted(result["candidates"], key=lambda item: (item["pairwise_rank"], item["raw_rank"]))
    assert reranked[0]["output"] == hit_sid
    assert len(result["pairwise_judgments"]) == 6
    assert any(record["swap"] for record in result["pairwise_judgments"])
    assert result["typed_criteria"][0]["criterion_id"] == "history_product_match"


def test_compute_pairwise_rubric_rerank_single_rule_audit_mode(tmp_path):
    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"

    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()
    with (rubric_dir / "video.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "rule_id": "history_topic_match",
                    "rule_text": "优先选择与用户最近观看主题更连贯的候选。",
                    "evidence_source": ["history_item_captions", "candidate_caption"],
                    "decision_rule": "比较候选 A 和 B 与最近观看主题的贴近程度，更贴近者胜。",
                    "tie_condition": "若 A 和 B 在该方面证据都弱或差异不明显，则 tie。",
                    "weight": 3,
                },
                {
                    "rule_id": "query_intent_match",
                    "rule_text": "优先选择符合用户当前查询意图的候选。",
                    "evidence_source": ["query_text", "candidate_caption"],
                    "decision_rule": "比较候选 A 和 B 是否符合用户当前查询意图，更符合者胜。",
                    "tie_condition": "若 A 和 B 在该方面证据都弱或差异不明显，则 tie。",
                    "weight": 2,
                },
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    result = compute_pairwise_rubric_rerank(
        predictions=[miss_sid, hit_sid],
        extra_info={
            "schema_id": "video",
            "task_name": "video",
            "prompt_text": "后续视频：",
            "user_profile_text": "分析用户近期观看兴趣。",
            "history_item_captions": ["校园互动视频", "轻松人物剧情视频"],
        },
        sidecar_lookup={
            hit_sid: {"pid": 101, "caption": "命中视频标题"},
            miss_sid: {"pid": 102, "caption": "普通视频标题"},
        },
        rubric_dir=str(rubric_dir),
        judge_impl=FakeSingleRuleJudge(),
        pairwise_top_n=2,
        pairwise_judge_mode="single_rule_audit",
    )

    reranked = sorted(result["candidates"], key=lambda item: (item["pairwise_rank"], item["raw_rank"]))
    assert reranked[0]["output"] == hit_sid
    assert result["history_summary_record"]["history_summary_nonempty"] is True
    assert len(result["single_rule_prompts"]) == 4
    assert len(result["single_rule_outputs"]) == 4
    assert len(result["single_rule_audit"]) == 4
    assert result["audit_metrics"]["expected_rule_calls"] == 4
    assert result["audit_metrics"]["completed_rule_calls"] == 4
    assert all("candidate_id" not in record["judge_prompt"] for record in result["single_rule_prompts"])


def test_build_single_rule_judge_prompt_can_toggle_candidate_id():
    rule = {
        "criterion_id": "history_topic_match",
        "criterion": "优先选择与用户最近观看主题更连贯的候选。",
        "weight": 3,
        "evidence_source": ["user_goal", "history_summary", "history_evidence_snippets", "candidate_caption"],
        "decision_rule": "比较候选 A 和 B 与最近观看主题的贴近程度，更贴近者胜。",
        "tie_condition": "若 A 和 B 在该方面证据都弱或差异不明显，则 tie。",
    }
    history_payload = {
        "user_goal": "为该用户重排候选视频，优先更符合近期观看兴趣的内容。",
        "history_summary": {"dominant_topics": ["校园互动"], "recent_patterns": ["轻松人物内容"], "supporting_history_ids": [1]},
        "history_evidence_snippets": [{"history_id": 1, "text": "校园互动视频"}],
    }
    candidate_a = {"candidate_index": 1, "predicted_items": [{"caption": "命中视频标题"}]}
    candidate_b = {"candidate_index": 2, "predicted_items": [{"caption": "普通视频标题"}]}

    prompt_without_id = build_single_rule_judge_prompt(
        "video",
        rule,
        history_payload=history_payload,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        candidate_pool_captions=[],
        include_candidate_id=False,
    )
    prompt_with_id = build_single_rule_judge_prompt(
        "video",
        rule,
        history_payload=history_payload,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        candidate_pool_captions=[],
        include_candidate_id=True,
    )

    assert "candidate_id" not in prompt_without_id
    assert "candidate_id" in prompt_with_id
