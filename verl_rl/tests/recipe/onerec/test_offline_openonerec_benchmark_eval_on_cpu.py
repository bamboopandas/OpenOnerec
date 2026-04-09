from __future__ import annotations

import json
import re
import sys

import pandas as pd

from recipe.onerec import merge_openonerec_benchmark_rerank as merge_eval
from recipe.onerec import offline_openonerec_benchmark_eval as benchmark_eval


class FakeJudge:
    def generate(self, prompt: str) -> str:
        if "推荐历史摘要器" in prompt:
            return json.dumps(
                {
                    "dominant_topics": ["校园互动内容"],
                    "recent_patterns": ["近期偏好轻松人物视频"],
                    "supporting_history_ids": [1],
                },
                ensure_ascii=False,
            )
        assert '"A"' in prompt and '"B"' in prompt
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
                    {"rule_id": "history_product_match", "vote": winner, "evidence": "更像历史商品"},
                    {"rule_id": "history_video_bridge", "vote": winner, "evidence": "视频主题接近"},
                    {"rule_id": "semantic_quality", "vote": "tie", "evidence": ""},
                    {"rule_id": "raw_rank_anchor", "vote": "tie", "evidence": ""},
                ],
                "final_vote": winner,
                "short_reason": "pairwise",
            },
            ensure_ascii=False,
        )


class FakeVideoSingleRuleJudge:
    def generate(self, prompt: str) -> str:
        if "推荐历史摘要器" in prompt:
            return json.dumps(
                {
                    "dominant_topics": ["校园互动内容"],
                    "recent_patterns": ["近期偏好轻松人物视频"],
                    "supporting_history_ids": [1, 2],
                },
                ensure_ascii=False,
            )
        rule_id_match = re.search(r"rule_id:\s*\n([^\n]+)", prompt)
        rule_id = rule_id_match.group(1).strip() if rule_id_match else "history_topic_match"
        a_hit = bool(re.search(r'"candidate_a".*?命中视频标题', prompt, flags=re.DOTALL))
        b_hit = bool(re.search(r'"candidate_b".*?命中视频标题', prompt, flags=re.DOTALL))
        if a_hit and not b_hit:
            vote = "A"
        elif b_hit and not a_hit:
            vote = "B"
        else:
            vote = "tie"
        return json.dumps(
            {
                "rule_id": rule_id,
                "vote": vote,
                "history_basis": "最近历史主要是校园互动和轻松人物内容。",
                "candidate_a_basis": "A 与校园互动主题更接近。",
                "candidate_b_basis": "B 与历史主题更远。",
                "tie_analysis": "当前差异足够明显，因此不是 tie。" if vote != "tie" else "两者差异不明显，因此 tie。",
                "decision_rationale": "按该规则 A 更优。" if vote == "A" else ("按该规则 B 更优。" if vote == "B" else "按该规则返回 tie。"),
            },
            ensure_ascii=False,
        )


def test_offline_openonerec_benchmark_eval_reranks_raw_candidates(tmp_path):
    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"

    task_data_file = tmp_path / "product_test.parquet"
    generation_file = tmp_path / "test_generated.json"
    sidecar_path = tmp_path / "sidecar.parquet"
    caption_path = tmp_path / "pid2caption.parquet"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    pd.DataFrame(
        [
            {
                "hist_longview": [11],
                "hist_goods": [21],
                "metadata": json.dumps({"answer": hit_sid, "uid": 1, "uuid": "u1", "answer_iid": [31]}, ensure_ascii=False),
                "messages": json.dumps(
                    [
                        {"role": "system", "content": [{"type": "text", "text": "你是推荐专家"}]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "用户观看过视频并点击过商品。用户当前的偏好涉及学习, 文具, 校园等主题。请推荐下一个商品。",
                                }
                            ],
                        },
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    ).to_parquet(task_data_file, index=False)

    with generation_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "fake-model",
                "samples": {
                    "0": {
                        "prompt": "fake prompt",
                        "ground_truth": hit_sid,
                        "metadata": {"row_index": 0, "uuid": "u1", "answer_iid": [31]},
                        "generations": [miss_sid, hit_sid],
                    }
                },
            },
            handle,
            ensure_ascii=False,
        )

    pd.DataFrame(
        [
            {"sid": hit_sid, "pid": 31, "caption": "命中商品标题", "mapping_file": "product_pid2sid.parquet"},
            {"sid": miss_sid, "pid": 32, "caption": "无关商品标题", "mapping_file": "product_pid2sid.parquet"},
        ]
    ).to_parquet(sidecar_path, index=False)
    pd.DataFrame(
        [
            {"pid": 11, "caption": "历史视频标题"},
            {"pid": 21, "caption": "历史商品标题"},
            {"pid": 31, "caption": "命中商品标题"},
            {"pid": 32, "caption": "无关商品标题"},
        ]
    ).to_parquet(caption_path, index=False)

    with (rubric_dir / "product.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"criterion": "c1", "weight": 2, "explanation": "e1"},
                {"criterion": "c2", "weight": 1, "explanation": "e2"},
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    original_factory = benchmark_eval.get_offline_hf_judge_client
    benchmark_eval.get_offline_hf_judge_client = lambda **_: FakeJudge()
    try:
        summary = benchmark_eval._score_samples(
            task_name="product",
            task_df=pd.read_parquet(task_data_file),
            generation_samples=json.load(generation_file.open("r", encoding="utf-8"))["samples"],
            rubric_dir=str(rubric_dir),
            sidecar_index_path=str(sidecar_path),
            judge_model_path="fake-judge",
            history_summary_model_path="",
            cache_path=str(tmp_path / "judge_cache.sqlite"),
            history_limit=20,
            caption_max_chars=96,
            judge_max_new_tokens=128,
            device_map="cpu",
            torch_dtype="float32",
            attn_implementation="none",
            output_dir=tmp_path / "output",
            caption_files=[str(caption_path)],
            generation_file=str(generation_file),
            task_data_file=str(task_data_file),
            preference_summary_lookup={},
            adaptive_rule_lookup_by_sample_id={},
            adaptive_rule_lookup_by_uuid={},
            context_pids={11, 21, 31, 32},
            k=2,
            pass_ks=[1, 2],
            coverage_ks=[1, 2],
            pairwise_top_n=2,
            pairwise_raw_rank_anchor=0.1,
            pairwise_judge_mode="multi_rule",
        )
    finally:
        benchmark_eval.get_offline_hf_judge_client = original_factory

    assert summary["num_total_examples"] == 1
    assert summary["candidate_coverage"]["pass@2"] == 1.0
    assert summary["raw_ranking"]["pass@1"] == 0.0
    assert summary["raw_ranking"]["pass@2"] == 1.0
    assert summary["raw_ranking"]["pid_pass@1"] == 0.0
    assert summary["rubric_rerank"]["pass@1"] == 1.0
    assert summary["rubric_rerank"]["pid_pass@1"] == 1.0
    assert summary["delta"]["pass@1"] == 1.0
    judge_prompt_records = [
        json.loads(line)
        for line in (tmp_path / "output" / "judge_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(judge_prompt_records) == 2
    assert any(record["swap"] for record in judge_prompt_records)
    assert all("ground_truth_captions" not in record["judge_prompt"] for record in judge_prompt_records)
    assert judge_prompt_records[0]["typed_criteria"][0]["criterion_id"] == "query_summary_match"
    assert (tmp_path / "output" / "details.json").exists()
    assert (tmp_path / "output" / "rubric_snapshot.json").exists()
    assert (tmp_path / "output" / "judge_prompts.jsonl").exists()


def test_load_generation_samples_supports_shards(tmp_path):
    generation_file = tmp_path / "test_generated.json"
    with generation_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "fake-model",
                "samples": {str(i): {"prompt": f"p{i}", "ground_truth": "", "metadata": {}, "generations": []} for i in range(5)},
            },
            handle,
            ensure_ascii=False,
        )

    _, shard0 = benchmark_eval._load_generation_samples(str(generation_file), max_samples=0, shard_id=0, num_shards=2)
    _, shard1 = benchmark_eval._load_generation_samples(str(generation_file), max_samples=0, shard_id=1, num_shards=2)

    assert list(shard0) == ["0", "2", "4"]
    assert list(shard1) == ["1", "3"]


def test_sid_value_to_sid_supports_array_like_sid():
    sid = benchmark_eval._sid_value_to_sid([700, 5323, 6150])
    assert sid == "<|sid_begin|><s_a_700><s_b_5323><s_c_6150><|sid_end|>"


def test_get_task_row_falls_back_to_uuid_when_row_index_mismatches():
    task_df = pd.DataFrame(
        [
            {"metadata": json.dumps({"uuid": "u0"}, ensure_ascii=False), "value": "row0"},
            {"metadata": json.dumps({"uuid": "u1"}, ensure_ascii=False), "value": "row1"},
        ]
    )

    row = benchmark_eval._get_task_row(task_df, row_index=99, uuid="u1")

    assert row["value"] == "row1"


def test_build_preference_summary_lookup_extracts_summary(tmp_path):
    preference_file = tmp_path / "product_summary.parquet"
    pd.DataFrame(
        [
            {
                "metadata": json.dumps({"uuid": "u1"}, ensure_ascii=False),
                "messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "用户浏览过很多内容。用户当前的偏好涉及学习, 文具, 校园等主题。请推荐商品。",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    ).to_parquet(preference_file, index=False)

    lookup = benchmark_eval._build_preference_summary_lookup(str(preference_file))

    assert lookup == {"u1": "用户当前的偏好涉及学习, 文具, 校园等主题。"}


def test_offline_openonerec_benchmark_eval_writes_single_rule_audit_files(tmp_path):
    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"

    task_data_file = tmp_path / "video_test.parquet"
    generation_file = tmp_path / "test_generated.json"
    sidecar_path = tmp_path / "sidecar.parquet"
    caption_path = tmp_path / "pid2caption.parquet"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    pd.DataFrame(
        [
            {
                "hist_pid": [11, 12],
                "metadata": json.dumps({"answer": hit_sid, "uid": 1, "uuid": "u1", "answer_iid": [31]}, ensure_ascii=False),
                "messages": json.dumps(
                    [
                        {"role": "system", "content": [{"type": "text", "text": "你是推荐专家"}]},
                        {"role": "user", "content": [{"type": "text", "text": "后续视频："}]},
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    ).to_parquet(task_data_file, index=False)

    with generation_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "fake-model",
                "samples": {
                    "0": {
                        "prompt": "fake prompt",
                        "ground_truth": hit_sid,
                        "metadata": {"row_index": 0, "uuid": "u1", "answer_iid": [31]},
                        "generations": [miss_sid, hit_sid],
                    }
                },
            },
            handle,
            ensure_ascii=False,
        )

    pd.DataFrame(
        [
            {"sid": hit_sid, "pid": 31, "caption": "命中视频标题", "mapping_file": "video_ad_pid2sid.parquet"},
            {"sid": miss_sid, "pid": 32, "caption": "无关视频标题", "mapping_file": "video_ad_pid2sid.parquet"},
        ]
    ).to_parquet(sidecar_path, index=False)
    pd.DataFrame(
        [
            {"pid": 11, "caption": "校园互动视频"},
            {"pid": 12, "caption": "轻松人物剧情视频"},
            {"pid": 31, "caption": "命中视频标题"},
            {"pid": 32, "caption": "无关视频标题"},
        ]
    ).to_parquet(caption_path, index=False)

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
                }
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    original_factory = benchmark_eval.get_offline_hf_judge_client
    benchmark_eval.get_offline_hf_judge_client = lambda **_: FakeVideoSingleRuleJudge()
    try:
        summary = benchmark_eval._score_samples(
            task_name="video",
            task_df=pd.read_parquet(task_data_file),
            generation_samples=json.load(generation_file.open("r", encoding="utf-8"))["samples"],
            rubric_dir=str(rubric_dir),
            sidecar_index_path=str(sidecar_path),
            judge_model_path="fake-judge",
            history_summary_model_path="",
            cache_path=str(tmp_path / "judge_cache.sqlite"),
            history_limit=20,
            caption_max_chars=96,
            judge_max_new_tokens=128,
            device_map="cpu",
            torch_dtype="float32",
            attn_implementation="none",
            output_dir=tmp_path / "output",
            caption_files=[str(caption_path)],
            generation_file=str(generation_file),
            task_data_file=str(task_data_file),
            preference_summary_lookup={},
            adaptive_rule_lookup_by_sample_id={},
            adaptive_rule_lookup_by_uuid={},
            context_pids={11, 12, 31, 32},
            k=2,
            pass_ks=[1, 2],
            coverage_ks=[1, 2],
            pairwise_top_n=2,
            pairwise_raw_rank_anchor=0.0,
            pairwise_judge_mode="single_rule_audit",
        )
    finally:
        benchmark_eval.get_offline_hf_judge_client = original_factory

    assert summary["rule_completion_rate"] == 1.0
    assert summary["history_summary_nonempty_rate"] == 1.0
    assert (tmp_path / "output" / "history_summary.jsonl").exists()
    assert (tmp_path / "output" / "single_rule_prompts.jsonl").exists()
    assert (tmp_path / "output" / "single_rule_outputs.jsonl").exists()
    assert (tmp_path / "output" / "single_rule_audit.jsonl").exists()


def test_offline_openonerec_benchmark_eval_can_use_separate_history_summary_model(tmp_path):
    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"

    task_data_file = tmp_path / "video_test.parquet"
    generation_file = tmp_path / "test_generated.json"
    sidecar_path = tmp_path / "sidecar.parquet"
    caption_path = tmp_path / "pid2caption.parquet"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()

    pd.DataFrame(
        [
            {
                "hist_pid": [11, 12],
                "metadata": json.dumps({"answer": hit_sid, "uid": 1, "uuid": "u1", "answer_iid": [31]}, ensure_ascii=False),
                "messages": json.dumps(
                    [
                        {"role": "system", "content": [{"type": "text", "text": "你是推荐专家"}]},
                        {"role": "user", "content": [{"type": "text", "text": "推荐后续视频："}]},
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    ).to_parquet(task_data_file, index=False)

    with generation_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "fake-model",
                "samples": {
                    "0": {
                        "prompt": "fake prompt",
                        "ground_truth": hit_sid,
                        "metadata": {"row_index": 0, "uuid": "u1", "answer_iid": [31]},
                        "generations": [miss_sid, hit_sid],
                    }
                },
            },
            handle,
            ensure_ascii=False,
        )

    pd.DataFrame(
        [
            {"sid": hit_sid, "pid": 31, "caption": "命中视频标题", "mapping_file": "video_ad_pid2sid.parquet"},
            {"sid": miss_sid, "pid": 32, "caption": "普通视频标题", "mapping_file": "video_ad_pid2sid.parquet"},
        ]
    ).to_parquet(sidecar_path, index=False)
    pd.DataFrame(
        [
            {"pid": 11, "caption": "校园互动视频"},
            {"pid": 12, "caption": "轻松人物剧情视频"},
            {"pid": 31, "caption": "命中视频标题"},
            {"pid": 32, "caption": "普通视频标题"},
        ]
    ).to_parquet(caption_path, index=False)

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
                }
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    created_models: list[str] = []

    class SummaryOnlyJudge:
        def generate(self, prompt: str) -> str:
            assert "推荐历史摘要器" in prompt
            return json.dumps(
                {
                    "dominant_topics": ["独立摘要模型主题"],
                    "recent_patterns": ["近期偏好校园互动"],
                    "supporting_history_ids": [1],
                },
                ensure_ascii=False,
            )

    class FinalOnlyJudge:
        def generate(self, prompt: str) -> str:
            assert "推荐历史摘要器" not in prompt
            return json.dumps(
                {
                    "rule_id": "history_topic_match",
                    "vote": "B",
                    "history_basis": "历史更接近命中视频标题。",
                    "candidate_a_basis": "A 与历史更远。",
                    "candidate_b_basis": "B 与历史更近。",
                    "tie_analysis": "差异明显，因此不是 tie。",
                    "decision_rationale": "按该规则 B 更优。",
                },
                ensure_ascii=False,
            )

    def fake_factory(*, judge_model, **_kwargs):
        created_models.append(str(judge_model))
        if str(judge_model) == "fake-summary":
            return SummaryOnlyJudge()
        if str(judge_model) == "fake-judge":
            return FinalOnlyJudge()
        raise AssertionError(f"unexpected model path: {judge_model}")

    original_factory = benchmark_eval.get_offline_hf_judge_client
    benchmark_eval.get_offline_hf_judge_client = fake_factory
    try:
        summary = benchmark_eval._score_samples(
            task_name="video",
            task_df=pd.read_parquet(task_data_file),
            generation_samples=json.load(generation_file.open("r", encoding="utf-8"))["samples"],
            rubric_dir=str(rubric_dir),
            sidecar_index_path=str(sidecar_path),
            judge_model_path="fake-judge",
            history_summary_model_path="fake-summary",
            cache_path=str(tmp_path / "judge_cache.sqlite"),
            history_limit=20,
            caption_max_chars=96,
            judge_max_new_tokens=128,
            device_map="cpu",
            torch_dtype="float32",
            attn_implementation="none",
            output_dir=tmp_path / "output",
            caption_files=[str(caption_path)],
            generation_file=str(generation_file),
            task_data_file=str(task_data_file),
            preference_summary_lookup={},
            adaptive_rule_lookup_by_sample_id={},
            adaptive_rule_lookup_by_uuid={},
            context_pids={11, 12, 31, 32},
            k=2,
            pass_ks=[1, 2],
            coverage_ks=[1, 2],
            pairwise_top_n=2,
            pairwise_raw_rank_anchor=0.0,
            pairwise_judge_mode="single_rule_audit",
        )
    finally:
        benchmark_eval.get_offline_hf_judge_client = original_factory

    assert created_models == ["fake-judge", "fake-summary"]
    history_records = [
        json.loads(line)
        for line in (tmp_path / "output" / "history_summary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert history_records[0]["history_summary"]["dominant_topics"] == ["独立摘要模型主题"]
    assert summary["history_summary_model_path"] == "fake-summary"


def test_build_pid_caption_lookup_uses_column_pruning_and_filters():
    calls: list[dict[str, object]] = []
    original_read_parquet = benchmark_eval.pd.read_parquet
    original_parquet_columns = benchmark_eval._parquet_columns

    def fake_read_parquet(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return pd.DataFrame([{"pid": 11, "caption": "标题A"}, {"pid": 12, "caption": "标题B"}])

    benchmark_eval.pd.read_parquet = fake_read_parquet
    benchmark_eval._parquet_columns = lambda _: ["pid", "caption", "unused"]
    try:
        lookup = benchmark_eval._build_pid_caption_lookup(
            ["fake_caption.parquet"],
            caption_max_chars=96,
            allowed_pids={11, 12},
        )
    finally:
        benchmark_eval.pd.read_parquet = original_read_parquet
        benchmark_eval._parquet_columns = original_parquet_columns

    assert lookup == {11: "标题A", 12: "标题B"}
    assert len(calls) == 1
    assert calls[0]["columns"] == ["pid", "caption"]
    filters = calls[0]["filters"]
    assert isinstance(filters, list) and len(filters) == 1
    assert filters[0][0] == "pid"
    assert filters[0][1] == "in"
    assert set(filters[0][2]) == {11, 12}


def test_merge_openonerec_benchmark_rerank_merges_shards(tmp_path):
    shard0 = tmp_path / "shard_0"
    shard1 = tmp_path / "shard_1"
    shard0.mkdir()
    shard1.mkdir()

    hit_sid = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    miss_sid = "<|sid_begin|><s_a_2><s_b_2><s_c_2><|sid_end|>"
    record0 = {
        "sample_id": "0",
        "prompt": "p0",
        "ground_truth": hit_sid,
        "candidates": [
            {"output": miss_sid, "raw_rank": 1, "rubric_score": 0.1, "unresolved_sid_ratio": 0.0},
            {"output": hit_sid, "raw_rank": 2, "rubric_score": 0.9, "unresolved_sid_ratio": 0.0},
        ],
    }
    record1 = {
        "sample_id": "1",
        "prompt": "p1",
        "ground_truth": miss_sid,
        "candidates": [
            {"output": miss_sid, "raw_rank": 1, "rubric_score": 0.8, "unresolved_sid_ratio": 0.0},
        ],
    }

    for shard_dir, record in ((shard0, record0), (shard1, record1)):
        (shard_dir / "candidate_pool.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        (shard_dir / "judge_prompts.jsonl").write_text(json.dumps({"sample_id": record["sample_id"], "raw_rank": 1}, ensure_ascii=False) + "\n", encoding="utf-8")
        (shard_dir / "details.json").write_text(json.dumps([{"sample_id": record["sample_id"]}], ensure_ascii=False), encoding="utf-8")
        (shard_dir / "rubric_snapshot.json").write_text(json.dumps({"product": [{"criterion": "c1", "weight": 1}]}, ensure_ascii=False), encoding="utf-8")
        (shard_dir / "summary.json").write_text(json.dumps({"task_name": "product", "model_name": "fake"}, ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "merged"
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "merge_openonerec_benchmark_rerank.py",
            "--shard_dirs",
            str(shard0),
            str(shard1),
            "--output_dir",
            str(output_dir),
            "--k",
            "2",
            "--pass_ks",
            "1",
            "2",
            "--coverage_ks",
            "1",
            "2",
        ]
        merge_eval.main()
    finally:
        sys.argv = original_argv

    merged_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert merged_summary["num_total_examples"] == 2
    assert merged_summary["raw_ranking"]["pass@1"] == 0.5
    assert merged_summary["rubric_rerank"]["pass@1"] == 1.0
    assert (output_dir / "judge_prompts.jsonl").exists()
    assert (output_dir / "rubric_snapshot.json").exists()
