"""Download and validate the fixed 65-image Artwork dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "reasondb/evaluation/benchmarks/files/paintings_sampled_no_duplicated.csv"
TARGET = ROOT / "SemBench/files/artwork"
IMAGES = TARGET / "images"
LOCAL_CSV = TARGET / "paintings_local.csv"


def bounded_wikimedia_url(url: str) -> str:
    filename = unquote(urlparse(url).path.rsplit("/", 1)[-1]).replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(filename)}?width=1024"


def download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "greenQueryLab Artwork benchmark/1.0"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
            time.sleep(31)
            return
        except HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            delay = int(error.headers.get("Retry-After", "60"))
            print(f"Wikimedia throttled the download; retrying in {delay}s")
            time.sleep(delay)


def validate_prepared_data() -> int:
    from PIL import Image

    if not LOCAL_CSV.is_file():
        raise FileNotFoundError(f"Missing prepared Artwork CSV: {LOCAL_CSV}")
    with SOURCE.open(newline="") as source, LOCAL_CSV.open(newline="") as prepared:
        source_rows = list(csv.DictReader(source))
        prepared_rows = list(csv.DictReader(prepared))
    if len(source_rows) != 65 or len(prepared_rows) != len(source_rows):
        raise RuntimeError("Artwork preparation must preserve exactly 65 source rows")
    for source_row, prepared_row in zip(source_rows, prepared_rows):
        if prepared_row["title"] != source_row["title"]:
            raise RuntimeError("Prepared Artwork row order does not match the source dataset")
        if prepared_row["inception_text"] != f"Creation date: {source_row['inception']}":
            raise RuntimeError("Prepared Artwork inception text does not match the source dataset")
        path = ROOT / prepared_row["image_url"]
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise RuntimeError(f"Unreadable Artwork image: {path}: {error}") from error
    return len(prepared_rows)


def prepare() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
        if len(rows) != 65:
            raise RuntimeError("Artwork source must contain exactly 65 rows")
        fieldnames = [*rows[0], "inception_text"]
    prepared_rows = []
    for index, row in enumerate(rows):
        path = IMAGES / f"{index:03d}.jpg"
        if not path.is_file():
            temporary = path.with_suffix(".part")
            download(bounded_wikimedia_url(row["image_url"]), temporary)
            temporary.replace(path)
        prepared_row = dict(row)
        prepared_row["image_url"] = str(path.relative_to(ROOT))
        prepared_row["inception_text"] = f"Creation date: {row['inception']}"
        prepared_rows.append(prepared_row)

    temporary_csv = LOCAL_CSV.with_suffix(".tmp")
    with temporary_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prepared_rows)
    temporary_csv.replace(LOCAL_CSV)
    print(f"Prepared and validated {validate_prepared_data()} Artwork images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(f"Validated {validate_prepared_data()} prepared Artwork images")
    else:
        prepare()


if __name__ == "__main__":
    main()
