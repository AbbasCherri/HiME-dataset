"""Atomic writes, content hashing and deterministic parquet output.

Every derived artefact in this package is written through `atomic_write`
(or `write_parquet_deterministic`, which uses it), so an interrupted run
never leaves a half-written file that a later run would trust.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Fixed zstd level, so two builds on different pyarrow/zstd patch versions
# still round-trip to the same compressed bytes for identical input.
PARQUET_ZSTD_LEVEL = 9


def atomic_write(path: Path | str, data: bytes) -> None:
    """Write `data` to `path` so a crash mid-write never corrupts it.

    Writes to a temp file in the same directory, flushes and fsyncs it, then
    `os.replace`s it onto the destination. A rename on the same filesystem is
    atomic, so an exception at any point before the replace leaves the
    original file (or no file) exactly as it was, with the temp file removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_deterministic(
    df: pd.DataFrame,
    path: Path | str,
    *,
    schema: pa.Schema,
    sort_by: list[str],
) -> None:
    """Write `df` as a byte-stable parquet file.

    Fixed column order (the schema's), explicit types, no pandas index, rows
    sorted by `sort_by`, zstd at a fixed level, and dictionary encoding is
    switched off for float columns, so a value's own bytes -- not a
    dictionary index -- determine the file's bytes.
    """
    ordered = df.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    ordered = ordered[list(schema.names)]
    table = pa.Table.from_pandas(ordered, schema=schema, preserve_index=False)
    # `use_dictionary` takes column names to enable, not a per-column bool list.
    use_dictionary = [field.name for field in schema if not pa.types.is_floating(field.type)]
    sink = BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=PARQUET_ZSTD_LEVEL,
        use_dictionary=use_dictionary,
        write_statistics=True,
    )
    atomic_write(path, sink.getvalue())
