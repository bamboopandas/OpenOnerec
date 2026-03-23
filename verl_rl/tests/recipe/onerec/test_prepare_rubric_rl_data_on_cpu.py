from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "data" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MODULE_PATH = SCRIPT_DIR / "prepare_rubric_rl_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_rubric_rl_data", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_enrich_dataframe_builds_expected_extra_info(tmp_path):
    mapping_path = tmp_path / "pid2sid.parquet"
    caption_path = tmp_path / "pid2caption.parquet"

    sid_value = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
    pd.DataFrame([{"pid": 11, "sid": sid_value}]).to_parquet(mapping_path, index=False)
    pd.DataFrame([{"pid": 11, "caption": "科幻动作短片"}]).to_parquet(caption_path, index=False)

    sidecar_df = MODULE.build_sidecar_index(
        mapping_files=[str(mapping_path)],
        caption_files=[str(caption_path)],
        caption_max_chars=96,
    )
    sidecar_lookup = MODULE.load_sidecar_lookup(sidecar_df)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "你是一个智能推荐助手"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"用户画像：喜欢科幻动作\n用户查询：机甲\n最近看过：{sid_value}\n请推荐相关内容。"}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sid_value}]},
    ]
    df = pd.DataFrame(
        [
            {
                "uuid": "sample-1",
                "source": "RecIF_InteractiveRec",
                "messages": json.dumps(messages, ensure_ascii=False),
                "metadata": json.dumps({"uid": 1, "keyword": "机甲"}, ensure_ascii=False),
            }
        ]
    )

    enriched = MODULE.enrich_dataframe(
        df=df,
        sidecar_lookup=sidecar_lookup,
        history_limit=20,
        caption_max_chars=96,
    )
    extra_info = json.loads(enriched.loc[0, "extra_info"])

    assert extra_info["task_name"] == "interactive"
    assert extra_info["schema_id"] == "interactive"
    assert extra_info["query_text"] == "机甲"
    assert extra_info["ground_truth_sids"] == [sid_value]
    assert extra_info["ground_truth_captions"] == ["科幻动作短片"]
    assert extra_info["history_item_captions"] == ["科幻动作短片"]
    assert extra_info["context_version"] == "v1"
