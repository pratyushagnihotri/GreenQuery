"""Download and extract the existing Medical archive without executing a demo query."""

from __future__ import annotations

import zipfile
import csv
from pathlib import Path

import gdown
from PIL import Image


ROOT = Path(__file__).parents[2]
TARGET = ROOT / "SemBench/files/medical"
ARCHIVE = TARGET / "medical_data.zip"
RAW_DATA = TARGET / "data/raw_data"
URL = "https://drive.google.com/uc?id=1v4--C7PE_SQDNZj6hZ8wNn4OvswIyu2s"
FILES = ROOT / "reasondb/evaluation/benchmarks/files"


def _validate_images() -> None:
    missing, unreadable, checked = [], [], 0
    for csv_name in ("image_skin_data_1000.csv", "image_x_ray_data_1000.csv"):
        with (FILES / csv_name).open(newline="") as source:
            for row in csv.DictReader(source):
                path = ROOT / row["image_path"]
                if not path.is_file():
                    missing.append(str(path))
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    checked += 1
                except Exception as error:
                    unreadable.append(f"{path}: {error}")
    if missing or unreadable:
        details = (missing + unreadable)[:10]
        raise RuntimeError(
            f"Medical image validation failed: {len(missing)} missing, "
            f"{len(unreadable)} unreadable. Examples: {details}"
        )
    print(f"Validated {checked} Medical image assets")


def main() -> None:
    if not RAW_DATA.exists():
        TARGET.mkdir(parents=True, exist_ok=True)
        if not ARCHIVE.exists():
            gdown.download(URL, output=str(ARCHIVE), quiet=False)
        with zipfile.ZipFile(ARCHIVE) as source:
            root = TARGET.resolve()
            for member in source.infolist():
                destination = (TARGET / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            source.extractall(TARGET)
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Archive did not contain expected directory: {RAW_DATA}")
    _validate_images()
    print(f"Prepared Medical archive: {RAW_DATA}")


if __name__ == "__main__":
    main()
