from __future__ import annotations

from recipe.onerec import adaptive_rule_induction as induction


def _candidate(output: str, raw_rank: int, pid: int, caption: str, rubric_score: float) -> dict:
    return {
        "output": output,
        "raw_rank": raw_rank,
        "predicted_pid": pid,
        "predicted_items": [{"pid": pid, "caption": caption}],
        "rubric_score": rubric_score,
        "unresolved_sid_ratio": 0.0,
    }


def test_build_support_pair_records_uses_target_and_hard_negatives():
    records = [
        {
            "sample_id": "s1",
            "uuid": "u1",
            "ground_truth": "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>",
            "extra_info": {
                "ground_truth_pids": [101],
                "ground_truth_captions": ["街舞教学视频"],
                "history_item_captions": ["街舞表演", "舞蹈练习"],
                "history_ad_captions": [],
                "query_text": "请为用户推荐喜欢的视频",
            },
            "candidates": [
                _candidate("wrong-1", 1, 201, "军事训练片段", 0.9),
                _candidate("wrong-2", 2, 202, "街舞表演合集", 0.1),
                _candidate("hit", 3, 101, "街舞教学视频", 0.2),
            ],
        }
    ]

    support_pairs = induction._build_support_pair_records(records, "video", candidate_top_k=3, max_support_pairs=0)

    assert len(support_pairs) == 2
    assert support_pairs[0]["target_caption"] == "街舞教学视频"
    assert support_pairs[0]["negative_caption"] == "军事训练片段"
    assert support_pairs[1]["negative_caption"] == "街舞表演合集"
    assert "rubric_score" not in support_pairs[0]


def test_retrieve_support_pairs_uses_history_and_candidate_signatures_not_query_only():
    test_record = {
        "sample_id": "test",
        "uuid": "ut",
        "extra_info": {
            "history_item_captions": ["街舞表演", "舞蹈练习"],
            "history_ad_captions": [],
            "query_text": "请为用户推荐喜欢的视频",
        },
        "candidates": [
            _candidate("a", 1, 1, "街舞舞台视频", 0.0),
            _candidate("b", 2, 2, "舞蹈教学短片", 0.0),
        ],
    }
    support_pairs = [
        {
            "support_pair_id": "p1",
            "sample_id": "dev1",
            "uuid": "u1",
            "task_name": "video",
            "query_text": "请为用户推荐喜欢的视频",
            "history_item_captions": ["街舞表演", "舞蹈练习"],
            "history_ad_captions": [],
            "history_signature": "街舞表演 | 舞蹈练习",
            "candidate_signature": "街舞舞台视频 | 舞蹈教学短片",
            "target_caption": "街舞教学视频",
            "negative_caption": "军事训练片段",
        },
        {
            "support_pair_id": "p2",
            "sample_id": "dev2",
            "uuid": "u2",
            "task_name": "video",
            "query_text": "请为用户推荐喜欢的视频",
            "history_item_captions": ["军事训练", "士兵生活"],
            "history_ad_captions": [],
            "history_signature": "军事训练 | 士兵生活",
            "candidate_signature": "军营纪实 | 军事演习",
            "target_caption": "军事演习视频",
            "negative_caption": "街舞片段",
        },
    ]

    retrieved = induction._retrieve_support_pairs("video", test_record, support_pairs, support_top_k=1)

    assert len(retrieved) == 1
    assert retrieved[0]["support_pair_id"] == "p1"


def test_validate_rules_rejects_question_style_and_illegal_evidence():
    rules, errors = induction._validate_rules(
        "video",
        [
            {
                "rule_id": "history_topic_match",
                "rule_text": "如何比较 A 和 B 是否更相关？",
                "evidence_source": ["history_item_captions", "candidate_caption"],
                "decision_rule": "比较 A 和 B 与历史主题的贴近程度。",
                "tie_condition": "若差异不明显则 tie。",
                "weight": 3,
            },
            {
                "rule_id": "query_intent_match",
                "rule_text": "优先选择更贴近当前任务意图的候选。",
                "evidence_source": ["销量"],
                "decision_rule": "比较 A 和 B 与当前任务意图的贴近程度，更贴近者胜。",
                "tie_condition": "若差异不明显则 tie。",
                "weight": 2,
            },
            {
                "rule_id": "duplicate_penalty",
                "rule_text": "避免把明显重复的候选排在前面。",
                "evidence_source": ["candidate_caption"],
                "decision_rule": "若 A 比 B 更重复，则选 B；若 B 比 A 更重复，则选 A。",
                "tie_condition": "若两者都无明显重复则 tie。",
                "weight": 1,
            },
        ],
    )

    assert len(rules) == 1
    assert rules[0]["rule_id"] == "duplicate_penalty"
    assert any("question-style rule_text" in error for error in errors)
    assert any("illegal evidence_source" in error for error in errors)
