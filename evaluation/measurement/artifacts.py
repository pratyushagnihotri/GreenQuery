"""Lossless split storage for large measurement JSON arrays."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


LAYOUT_VERSION = "split-jsonl-gzip-v1"
DEFAULT_EXTERNALIZE_BYTES = 64 * 1024
_FORMAT = "jsonl+gzip"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace existing artifact: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_gzip(raw: tempfile.SpooledTemporaryFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.sha256()
        try:
            with gzip.open(path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except (OSError, EOFError) as error:
            raise ValueError(f"Unreadable existing sidecar: {path}") from error
        if digest.hexdigest() != path.name.removesuffix(".jsonl.gz"):
            raise ValueError(f"Existing sidecar checksum mismatch: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as destination:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=destination, mtime=0
            ) as compressed:
                raw.seek(0)
                for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                    compressed.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_reference(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == _FORMAT
        and isinstance(value.get("$ref"), str)
        and isinstance(value.get("sha256"), str)
        and isinstance(value.get("count"), int)
    )


def _split_value(
    value: Any,
    *,
    document_dir: Path,
    sidecar_dir: Path,
    threshold: int,
    path: tuple[str | int, ...] = (),
) -> Any:
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _split_value(
                value[key],
                document_dir=document_dir,
                sidecar_dir=sidecar_dir,
                threshold=threshold,
                path=path + (key,),
            )
        return value
    if not isinstance(value, list):
        return value

    for index in range(len(value)):
        value[index] = _split_value(
            value[index],
            document_dir=document_dir,
            sidecar_dir=sidecar_dir,
            threshold=threshold,
            path=path + (index,),
        )

    digest = hashlib.sha256()
    size = 0
    with tempfile.SpooledTemporaryFile(max_size=threshold) as raw:
        for item in value:
            line = _json_bytes(item) + b"\n"
            raw.write(line)
            digest.update(line)
            size += len(line)
        if size < threshold or path == ("queries",):
            return value

        checksum = digest.hexdigest()
        sidecar = sidecar_dir / f"{checksum}.jsonl.gz"
        _write_gzip(raw, sidecar)
        return {
            "$ref": sidecar.relative_to(document_dir).as_posix(),
            "format": _FORMAT,
            "count": len(value),
            "sha256": checksum,
        }


def save_measurement(
    measurement: dict[str, Any],
    path: Path,
    *,
    threshold: int = DEFAULT_EXTERNALIZE_BYTES,
) -> dict[str, Any]:
    """Split large arrays in-place and atomically save the compact measurement."""
    if threshold < 1:
        raise ValueError("threshold must be positive")
    measurement["artifact_layout_version"] = LAYOUT_VERSION
    sidecar_dir = path.with_name(f"{path.stem}_data")
    compact = _split_value(
        measurement,
        document_dir=path.parent,
        sidecar_dir=sidecar_dir,
        threshold=threshold,
        path=(),
    )
    _atomic_write(
        path,
        (json.dumps(compact, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return compact


def iter_sidecar(reference: dict[str, Any], document_path: Path) -> Iterator[Any]:
    """Yield and validate a referenced JSONL array."""
    if not _is_reference(reference):
        raise ValueError("Invalid sidecar reference")
    sidecar = document_path.parent / reference["$ref"]
    digest = hashlib.sha256()
    count = 0
    try:
        with gzip.open(sidecar, "rb") as source:
            for line_number, line in enumerate(source, start=1):
                digest.update(line)
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed sidecar {sidecar} at line {line_number}"
                    ) from error
                count += 1
                yield item
    except (OSError, EOFError) as error:
        raise ValueError(f"Unreadable sidecar: {sidecar}") from error
    if count != reference["count"]:
        raise ValueError(
            f"Sidecar count mismatch for {sidecar}: expected {reference['count']}, got {count}"
        )
    observed = digest.hexdigest()
    if observed != reference["sha256"]:
        raise ValueError(
            f"Sidecar checksum mismatch for {sidecar}: expected {reference['sha256']}, got {observed}"
        )


def iter_array(value: Any, document_path: Path) -> Iterator[Any]:
    """Iterate either an inline list or a split-array reference."""
    if isinstance(value, list):
        yield from value
        return
    if _is_reference(value):
        yield from iter_sidecar(value, document_path)
        return
    raise TypeError("Expected an inline list or sidecar reference")


def array_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if _is_reference(value):
        return value["count"]
    raise TypeError("Expected an inline list or sidecar reference")


def _hydrate(value: Any, document_path: Path) -> Any:
    if _is_reference(value):
        return [
            _hydrate(item, document_path)
            for item in iter_sidecar(value, document_path)
        ]
    if isinstance(value, dict):
        return {key: _hydrate(item, document_path) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydrate(item, document_path) for item in value]
    return value


def load_measurement(path: Path, *, hydrate: bool = False) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        measurement = json.load(source)
    if hydrate and measurement.get("artifact_layout_version") == LAYOUT_VERSION:
        measurement = _hydrate(measurement, path)
        measurement.pop("artifact_layout_version", None)
    return measurement
