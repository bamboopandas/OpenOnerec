from __future__ import annotations

import json

from recipe.onerec import build_rubrics


def test_load_candidate_pool_supports_candidate_pool_jsonl(tmp_path):
    input_path = tmp_path / "candidate_pool.jsonl"
    record = {
        "uuid": "u1",
        "extra_info": {"schema_id": "product"},
        "candidates": [
            {"output": "cand-a"},
            {"output": "cand-b"},
        ],
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = build_rubrics._load_candidate_pool(str(input_path))

    assert loaded == {"u1": ["cand-a", "cand-b"]}


def test_default_examples_include_candidate_examples():
    records = [
        {
            "uuid": "u1",
            "prompt": "推荐下一个商品",
            "extra_info": {
                "schema_id": "product",
                "query_text": "推荐下一个商品",
                "history_product_captions": ["补水面霜", "保湿精华"],
                "ground_truth_captions": ["修护面霜"],
            },
            "candidates": [
                {
                    "raw_rank": 1,
                    "rubric_score": 0.8,
                    "output": "cand-a",
                    "predicted_items": [{"caption": "修护面霜", "pid": 1}],
                },
                {
                    "raw_rank": 2,
                    "rubric_score": 0.2,
                    "output": "cand-b",
                    "predicted_items": [{"caption": "彩妆套装", "pid": 2}],
                },
            ],
        }
    ]

    examples = build_rubrics._default_examples(records, sample_size=1)

    assert len(examples) == 1
    assert examples[0]["prompt_text"] == "推荐下一个商品"
    assert examples[0]["candidate_examples"][0]["captions"] == ["修护面霜"]
    assert examples[0]["candidate_examples"][1]["captions"] == ["彩妆套装"]


def test_sanitize_rubric_items_filters_unobservable_signals():
    sanitized = build_rubrics._sanitize_rubric_items(
        [
            {"criterion": "商品销量高", "weight": 2, "explanation": "销量高说明更受欢迎"},
            {"criterion": "推荐商品与用户历史护肤兴趣一致，且类别相近。", "weight": 2.7, "explanation": "只用可观察文本"},
            {"criterion": "评分高", "weight": 1, "explanation": "评分高"},
        ]
    )

    assert sanitized == [
        {
            "criterion": "推荐商品与用户历史护肤兴趣一致，且类别相近。",
            "weight": 3,
            "explanation": "只用可观察文本",
        }
    ]
