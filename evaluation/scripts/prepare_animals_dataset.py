"""Validate staged Animals media before starting a measured job."""

import csv
from pathlib import Path


ROOT = Path(__file__).parents[2]
FILES = ROOT / "reasondb/evaluation/benchmarks/files"


def main() -> None:
    missing = []
    for filename, path_column, expected_rows in (
        ("animals_500.csv", "imagepath", 500),
        ("animals_audio.csv", "audiopath", 166),
    ):
        with (FILES / filename).open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != expected_rows:
            raise SystemExit(f"{filename}: expected {expected_rows} rows, found {len(rows)}")
        missing.extend(ROOT / row[path_column] for row in rows if not (ROOT / row[path_column]).is_file())
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise SystemExit(f"missing {len(missing)} referenced media files; first: {preview}")
    print("validated Animals media: 500 images and 166 audio clips")


if __name__ == "__main__":
    main()
