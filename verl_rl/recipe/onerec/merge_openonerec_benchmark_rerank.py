from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from recipe.onerec.offline_openonerec_benchmark_eval import summarize_candidate_pool_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded OpenOneRec offline rubric rerank outputs")
    parser.add_argument("--shard_dirs", nargs="+", required=True, help="Per-shard output directories")
    parser.add_argument("--output_dir", required=True, help="Merged output directory")
    parser.add_argument("--k", type=int, default=32, help="Top-k cutoff")
    parser.add_argument("--pass_ks", nargs="*", type=int, default=[1, 5, 10, 32], help="Pass@k values")
    parser.add_argument("--coverage_ks", nargs="*", type=int, default=[10, 20, 50, 100], help="Coverage pass@k values")
    return parser.parse_args()


def _sample_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    sample_id = str(record.get("sample_id", ""))
    try:
        return (int(sample_id), sample_id)
    except ValueError:
        return (10**12, sample_id)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _merge_caches(shard_dirs: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rubric_cache (
            cache_key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        )
        """
    )
    for shard_dir in shard_dirs:
        cache_path = shard_dir / "judge_cache.sqlite"
        if not cache_path.exists():
            continue
        shard_conn = sqlite3.connect(cache_path)
        try:
            rows = shard_conn.execute("SELECT cache_key, value_json FROM rubric_cache").fetchall()
            conn.executemany(
                """
                INSERT INTO rubric_cache(cache_key, value_json)
                VALUES(?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET value_json = excluded.value_json
                """,
                rows,
            )
            conn.commit()
        finally:
            shard_conn.close()
    conn.close()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _judge_prompt_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _sample_sort_key(record),
        int(record.get("left_raw_rank", record.get("raw_rank", 0))),
        int(record.get("right_raw_rank", 0)),
        int(bool(record.get("swap", False))),
        int(record.get("comparison_id", 0)),
    )


def main() -> None:
    args = parse_args()
    shard_dirs = [Path(item) for item in args.shard_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_pool_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    judge_prompts: list[dict[str, Any]] = []
    rubric_snapshot: dict[str, Any] = {}
    shard_summaries: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        candidate_pool_records.extend(_read_jsonl(shard_dir / "candidate_pool.jsonl"))
        details.extend(_read_json(shard_dir / "details.json", []))
        judge_prompts.extend(_read_jsonl(shard_dir / "judge_prompts.jsonl"))
        rubric_snapshot.update(_read_json(shard_dir / "rubric_snapshot.json", {}))
        summary = _read_json(shard_dir / "summary.json", {})
        if summary:
            shard_summaries.append(summary)

    candidate_pool_records.sort(key=_sample_sort_key)
    details.sort(key=_sample_sort_key)
    judge_prompts.sort(key=_judge_prompt_sort_key)

    summary = summarize_candidate_pool_records(
        candidate_pool_records,
        k=args.k,
        pass_ks=args.pass_ks,
        coverage_ks=args.coverage_ks,
    )
    if shard_summaries:
        first_summary = shard_summaries[0]
        for key in (
            "task_name",
            "generation_file",
            "task_data_file",
            "judge_model_path",
            "sidecar_index_path",
            "model_name",
        ):
            if key in first_summary:
                summary[key] = first_summary[key]
    summary["k"] = args.k
    summary["pass_ks"] = list(args.pass_ks)
    summary["coverage_ks"] = list(args.coverage_ks)
    summary["num_shards"] = len(shard_dirs)
    summary["shard_summaries"] = [str(shard_dir / "summary.json") for shard_dir in shard_dirs]

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "details.json", details)
    _write_json(output_dir / "rubric_snapshot.json", rubric_snapshot)
    with (output_dir / "candidate_pool.jsonl").open("w", encoding="utf-8") as handle:
        for record in candidate_pool_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "judge_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in judge_prompts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _merge_caches(shard_dirs, output_dir / "judge_cache.sqlite")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
