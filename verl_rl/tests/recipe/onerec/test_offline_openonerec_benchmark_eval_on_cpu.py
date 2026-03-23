from __future__ import annotations

import json
import sys

import pandas as pd

from recipe.onerec import merge_openonerec_benchmark_rerank as merge_eval
from recipe.onerec import offline_openonerec_benchmark_eval as benchmark_eval


class FakeJudge:
    def generate(self, prompt: str) -> str:
        satisfied = "命中商品标题" in prompt
        score = 0.9 if satisfied else 0.1
        return json.dumps(
            {
                "criterion_results": [
                    {"criterion": "c1", "satisfied": satisfied, "evidence": "ok"},
                    {"criterion": "c2", "satisfied": False, "evidence": "skip"},
                ],
                "rubric_score": score,
                "judge_reason": "ranked",
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
                        {"role": "user", "content": [{"type": "text", "text": "用户观看过视频并点击过商品，请推荐下一个商品"}]},
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
            context_pids={11, 21, 31, 32},
            k=2,
            pass_ks=[1, 2],
            coverage_ks=[1, 2],
        )
    finally:
        benchmark_eval.get_offline_hf_judge_client = original_factory

    assert summary["num_total_examples"] == 1
    assert summary["num_rerank_examples"] == 1
    assert summary["candidate_coverage"]["pass@2"] == 1.0
    assert summary["all_examples"]["raw_ranking"]["pass@1"] == 0.0
    assert summary["all_examples"]["raw_ranking"]["pass@2"] == 1.0
    assert summary["all_examples"]["rubric_rerank"]["pass@1"] == 1.0
    assert summary["rerankable_subset"]["delta"]["pass@1"] == 1.0
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
    assert merged_summary["num_rerank_examples"] == 2
    assert merged_summary["all_examples"]["raw_ranking"]["pass@1"] == 0.5
    assert merged_summary["all_examples"]["rubric_rerank"]["pass@1"] == 1.0
    assert (output_dir / "judge_prompts.jsonl").exists()
    assert (output_dir / "rubric_snapshot.json").exists()
