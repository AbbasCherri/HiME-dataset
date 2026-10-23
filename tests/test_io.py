import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from notodata import io as nio


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("building_id", pa.int64()),
            pa.field("value", pa.float64()),
        ]
    )


def test_atomic_write_creates_the_file_with_exact_bytes(tmp_path):
    target = tmp_path / "sub" / "out.bin"
    nio.atomic_write(target, b"hello world")
    assert target.read_bytes() == b"hello world"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_write_leaves_no_partial_file_when_interrupted(tmp_path, monkeypatch):
    target = tmp_path / "out.bin"
    target.write_bytes(b"original-content")

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(nio.os, "fsync", boom)

    with pytest.raises(RuntimeError):
        nio.atomic_write(target, b"new-content-that-must-never-land")

    assert target.read_bytes() == b"original-content"
    assert list(tmp_path.iterdir()) == [target]


def test_sha256_bytes():
    assert nio.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"some content")
    assert nio.sha256_file(path) == hashlib.sha256(b"some content").hexdigest()


def test_parquet_write_is_byte_identical_across_runs(tmp_path):
    df = pd.DataFrame({"building_id": [3, 1, 2], "value": [30.0, 10.0, 20.0]})
    schema = _schema()
    out1, out2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    nio.write_parquet_deterministic(df, out1, schema=schema, sort_by=["building_id"])
    nio.write_parquet_deterministic(df, out2, schema=schema, sort_by=["building_id"])
    assert out1.read_bytes() == out2.read_bytes()


def test_parquet_rows_sorted_and_no_pandas_index(tmp_path):
    df = pd.DataFrame({"building_id": [3, 1, 2], "value": [30.0, 10.0, 20.0]}, index=[9, 8, 7])
    out = tmp_path / "a.parquet"
    nio.write_parquet_deterministic(df, out, schema=_schema(), sort_by=["building_id"])
    table = pq.read_table(out)
    assert table.column("building_id").to_pylist() == [1, 2, 3]
    assert set(table.schema.names) == {"building_id", "value"}


def test_parquet_uses_zstd_and_no_dictionary_for_float_columns(tmp_path):
    df = pd.DataFrame({"building_id": [1, 2], "value": [1.5, 2.5]})
    out = tmp_path / "a.parquet"
    nio.write_parquet_deterministic(df, out, schema=_schema(), sort_by=["building_id"])
    meta = pq.ParquetFile(out).metadata
    row_group = meta.row_group(0)
    value_col = row_group.column(1)
    assert value_col.compression == "ZSTD"
    assert "RLE_DICTIONARY" not in value_col.encodings
