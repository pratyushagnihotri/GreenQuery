"""Require a complete, human-approved and content-locked Movie Q3 ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def validate(csv_path: Path, lock_path: Path, expected_rows: int = 99) -> dict:
    with open(csv_path, newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} ground-truth rows, found {len(rows)}")
    invalid = [
        row.get("reviewid", "<missing>")
        for row in rows
        if (row.get("expected_label") or "").strip() not in {"0", "1"}
    ]
    if invalid:
        raise ValueError(
            f"ground truth is incomplete; {len(invalid)} rows lack a 0/1 label"
        )
    if not lock_path.exists():
        raise ValueError(
            f"human approval lock is missing: {lock_path}. "
            "Create it after review with annotator, reviewed_at, and csv_sha256."
        )
    lock = json.loads(lock_path.read_text())
    required = ("annotator", "reviewed_at", "csv_sha256")
    missing = [key for key in required if not lock.get(key)]
    if missing:
        raise ValueError(f"ground-truth lock lacks: {', '.join(missing)}")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if lock["csv_sha256"] != digest:
        raise ValueError("ground-truth CSV changed after human approval")
    return {
        "rows": len(rows),
        "positive_rows": sum(row["expected_label"].strip() == "1" for row in rows),
        "csv_sha256": digest,
        "annotator": lock["annotator"],
        "reviewed_at": lock["reviewed_at"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("evaluation/ground_truth/movie_q3_actor_performance.csv"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("evaluation/ground_truth/movie_q3_actor_performance.lock.json"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.ground_truth, args.lock), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
