from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "data" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MODULE_PATH = SCRIPT_DIR / "prepare_onerec_think_beauty_rl_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_onerec_think_beauty_rl_data", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_split_frames_from_raw_inputs(tmp_path):
    sequential_path = tmp_path / "sequential_data_processed.txt"
    items_path = tmp_path / "Beauty.pretrain.json"

    sid1 = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    sid2 = "<|sid_begin|><s_a_1><s_b_1><s_c_2><|sid_end|>"
    sid3 = "<|sid_begin|><s_a_1><s_b_1><s_c_3><|sid_end|>"
    sid4 = "<|sid_begin|><s_a_1><s_b_1><s_c_4><|sid_end|>"

    sequential_path.write_text("u1 i1 i2 i3 i4\n", encoding="utf-8")
    items_path.write_text(
        json.dumps(
            {
                "i1": {"sid": sid1, "title": "Lip Balm", "categories": ["Beauty", "Lip Care"]},
                "i2": {"sid": sid2, "title": "Face Serum", "categories": ["Beauty", "Skin Care"]},
                "i3": {"sid": sid3, "title": "Night Cream", "categories": ["Beauty", "Skin Care"]},
                "i4": {"sid": sid4, "title": "Toner", "categories": ["Beauty", "Skin Care"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    split_frames, sidecar_df = MODULE.build_split_frames_from_raw_inputs(
        str(sequential_path),
        str(items_path),
        max_history_len=50,
        caption_max_chars=96,
    )

    assert list(split_frames.keys()) == ["train", "val", "test"]
    assert len(split_frames["train"]) == 1
    assert len(split_frames["val"]) == 1
    assert len(split_frames["test"]) == 1
    assert len(sidecar_df) == 4

    test_row = split_frames["test"].iloc[0]
    extra_info = json.loads(test_row["extra_info"])
    metadata = json.loads(test_row["metadata"])

    assert extra_info["schema_id"] == "product"
    assert extra_info["task_variant"] == "beauty"
    assert extra_info["ground_truth_sids"] == [sid4]
    assert extra_info["history_product_captions"] == [
        "Lip Balm | Beauty > Lip Care",
        "Face Serum | Beauty > Skin Care",
        "Night Cream | Beauty > Skin Care",
    ]
    assert metadata["answer_pid"] == ["i4"]


def test_build_split_frames_from_prediction_parquets(tmp_path):
    prediction_dir = tmp_path / "prediction_data"
    prediction_dir.mkdir()
    sid1 = "<|sid_begin|><s_a_1><s_b_1><s_c_1><|sid_end|>"
    sid2 = "<|sid_begin|><s_a_1><s_b_1><s_c_2><|sid_end|>"
    history_description = (
        "The user has purchased the following items: "
        f'{sid1}, its title is "Lip Balm", its categories are "Beauty > Lip Care"; '
        f'{sid2}, its title is "Face Serum", its categories are "Beauty > Skin Care";'
    )

    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "description": history_description,
                "groundtruth": sid2,
                "title": "Face Serum",
                "categories": "Beauty > Skin Care",
            }
        ]
    )
    for split_name in ("train", "val", "test"):
        frame.to_parquet(prediction_dir / f"training_prediction_sid_data_{split_name}.parquet", index=False)

    split_frames, sidecar_df = MODULE.build_split_frames_from_prediction_parquets(
        str(prediction_dir),
        caption_max_chars=96,
    )

    assert len(sidecar_df) == 2
    assert len(split_frames["train"]) == 1
    extra_info = json.loads(split_frames["train"].iloc[0]["extra_info"])
    assert extra_info["history_item_captions"][0] == "Lip Balm | Beauty > Lip Care"
