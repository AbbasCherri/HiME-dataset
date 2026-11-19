"""D8 manifest: write, verify, flip one byte; chip tree hash; tile fingerprint."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from notodata import chips, manifest, paths
from notodata.io import write_parquet_deterministic
from test_table_schema import make_table

LAYER = "post_stub"


def _make_chip_tree(layer, n=4):
    """Real PNGs plus an index.parquet, as `chips.build_chips` would leave them."""
    import pandas as pd

    from notodata.cache import _CHIP_INDEX_SCHEMA

    rows = []
    rng = np.random.default_rng(0)
    for b in range(n):
        px = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        p = paths.chip_path(layer, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(px, "RGB").save(p, format="PNG")
        rel = p.relative_to(paths.notodata_home()).as_posix()
        rows.append((b, "ok", rel, chips.pixel_sha256(px)))
    rows.append((n, "missing", "", ""))
    index = pd.DataFrame(rows, columns=["building_id", "status", "path", "sha256"])
    write_parquet_deterministic(
        index, paths.chip_index_path(layer), schema=_CHIP_INDEX_SCHEMA, sort_by=["building_id"]
    )
    return index


@pytest.fixture
def release(tmp_home, tmp_path):
    """A small derived tree: table, chips, a split file, a report CSV and the decision."""
    from notodata import table

    table.write_table(make_table(), pre_available=True)
    _make_chip_tree(LAYER)
    paths.splits_dir().mkdir(parents=True)
    paths.split_path(1000).write_bytes(b"split-bytes")
    reports = tmp_path / "repo" / "reports"
    reports.mkdir(parents=True)
    (reports / "folds.csv").write_text("fold,n\n0,1\n")
    (reports / "imagery_decision.json").write_text("{}\n")
    return reports.parent


def test_manifest_roundtrip(release, tmp_path):
    entries = manifest.collect_entries([LAYER], root=release)
    names = set(entries)
    assert "derived/buildings.parquet" in names
    assert "derived/splits/seed_1000.parquet" in names
    assert f"chips/{LAYER}/index.parquet" in names
    assert f"chips/{LAYER}/@tree" in names
    assert "reports/folds.csv" in names
    assert "reports/imagery_decision.json" in names
    out = tmp_path / "MANIFEST.sha256"
    manifest.write_manifest(out, entries)
    first = out.read_bytes()
    assert manifest.verify_manifest(out, [LAYER], root=release) == []
    manifest.write_manifest(out, manifest.collect_entries([LAYER], root=release))
    assert out.read_bytes() == first  # sha256sum format, sorted, stable
    line = first.decode().splitlines()[0]
    digest, name = line.split("  ", 1)
    assert len(digest) == 64
    assert name in names

    target = paths.split_path(1000)
    raw = bytearray(target.read_bytes())
    raw[0] ^= 0x01
    target.write_bytes(bytes(raw))
    problems = manifest.verify_manifest(out, [LAYER], root=release)
    assert len(problems) == 1
    assert "derived/splits/seed_1000.parquet" in problems[0]


def test_verify_reports_missing_and_unlisted(release, tmp_path):
    out = tmp_path / "MANIFEST.sha256"
    manifest.write_manifest(out, manifest.collect_entries([LAYER], root=release))
    (release / "reports" / "folds.csv").unlink()
    (release / "reports" / "extra.csv").write_text("a\n")
    problems = "\n".join(manifest.verify_manifest(out, [LAYER], root=release))
    assert "reports/folds.csv" in problems
    assert "missing" in problems
    assert "reports/extra.csv" in problems
    assert "not in manifest" in problems


def test_chip_tree_hash_covers_every_chip(tmp_home):
    index = _make_chip_tree(LAYER)
    ok = index[index["status"] == "ok"].sort_values("path")
    expected = hashlib.sha256(
        "".join(f"{p} {h}\n" for p, h in zip(ok["path"], ok["sha256"], strict=True)).encode()
    ).hexdigest()
    assert manifest.chip_tree_hash(LAYER) == expected
    assert manifest.chip_tree_hash(LAYER, from_pixels=True) == expected

    # repaint one PNG: the index-based hash cannot see it, the pixel-based one does
    p = paths.chip_path(LAYER, 2)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(p, format="PNG")
    assert manifest.chip_tree_hash(LAYER) == expected
    assert manifest.chip_tree_hash(LAYER, from_pixels=True) != expected


def test_chip_tree_hash_changes_with_a_pixel_hash(tmp_home):
    import pandas as pd

    from notodata.cache import _CHIP_INDEX_SCHEMA

    index = _make_chip_tree(LAYER)
    before = manifest.chip_tree_hash(LAYER)
    index.loc[0, "sha256"] = "0" * 64
    write_parquet_deterministic(
        pd.DataFrame(index),
        paths.chip_index_path(LAYER),
        schema=_CHIP_INDEX_SCHEMA,
        sort_by=["building_id"],
    )
    assert manifest.chip_tree_hash(LAYER) != before


def _tile(layer, z, x, y, data, ext="jpg"):
    p = paths.tile_path(layer, z, x, y, ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_tile_fingerprint(tmp_home):
    _tile(LAYER, 18, 5, 7, b"aaa")
    _tile(LAYER, 18, 5, 6, b"bbb")
    (paths.tiles_dir(LAYER) / "layer.meta.json").write_text("{}")
    _tile(LAYER, 18, 9, 9, b"", ext="missing")
    lines = sorted(
        [
            f"18/5/6 {hashlib.sha256(b'bbb').hexdigest()}",
            f"18/5/7 {hashlib.sha256(b'aaa').hexdigest()}",
        ]
    )
    expected = hashlib.sha256("".join(f"{ln}\n" for ln in lines).encode()).hexdigest()
    assert manifest.tile_fingerprint(LAYER) == (expected, 2)

    _tile(LAYER, 18, 5, 7, b"aab")
    assert manifest.tile_fingerprint(LAYER)[0] != expected
    assert manifest.tile_fingerprint(LAYER)[1] == 2


def _fake_embeddings(deterministic):
    import json

    folder = paths.embeddings_dir("enc")
    folder.mkdir(parents=True)
    (folder / f"{LAYER}.npy").write_bytes(b"vectors")
    (folder / f"{LAYER}.ids.npy").write_bytes(b"ids")
    (folder / f"{LAYER}.npy.meta.json").write_text(json.dumps({"deterministic": deterministic}))


def test_a_gpu_embedding_build_is_left_out_of_the_manifest_with_its_ids(release):
    _fake_embeddings(deterministic=False)  # the rebuild cannot make GPU files: verified by cosine
    names = set(manifest.collect_entries([LAYER], root=release))
    assert not {n for n in names if n.startswith("embeddings/")}


def test_a_deterministic_embedding_build_lists_vectors_and_ids(release):
    _fake_embeddings(deterministic=True)
    names = set(manifest.collect_entries([LAYER], root=release))
    assert {f"embeddings/enc/{LAYER}.npy", f"embeddings/enc/{LAYER}.ids.npy"} <= names
