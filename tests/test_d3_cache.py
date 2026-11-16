import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from _fsscan import repo_root
from notodata import cache, chips, config, paths

IDS = [3, 1, 2]


def _meta(**over):
    base = {
        "layer_ids": ["a"],
        "zoom": 18,
        "chip_size": 128,
        "encoder_id": None,
        "building_ids": IDS,
    }
    base.update(over)
    return cache.expected_meta(**base)


def _compute_factory(calls):
    def compute(path):
        calls.append(path)
        path.write_bytes(b"data")

    return compute


def test_meta_has_the_brief_keys():
    meta = _meta()
    assert set(meta) == {
        "chip_pipeline_version",
        "layer_ids",
        "zoom",
        "chip_size",
        "encoder_id",
        "row_count",
        "building_ids_sha256",
    }
    assert meta["chip_pipeline_version"] == config.PIPELINE_VERSION
    assert meta["row_count"] == 3
    # order of ids does not matter
    assert meta["building_ids_sha256"] == _meta(building_ids=[1, 2, 3])["building_ids_sha256"]


def test_valid_sidecar_skips_compute(tmp_path):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    assert cache.ensure(path, _meta(), comp) is True
    assert cache.ensure(path, _meta(), comp) is False
    assert len(calls) == 1
    assert json.loads(cache.sidecar_path(path).read_text()) == _meta()


def test_missing_sidecar_recomputes(tmp_path):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    cache.ensure(path, _meta(), comp)
    cache.sidecar_path(path).unlink()
    assert cache.ensure(path, _meta(), comp) is True
    assert len(calls) == 2


def test_unreadable_sidecar_or_missing_data_recomputes(tmp_path):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    cache.ensure(path, _meta(), comp)
    cache.sidecar_path(path).write_text("{not json")
    assert cache.ensure(path, _meta(), comp) is True
    path.unlink()
    assert cache.ensure(path, _meta(), comp) is True
    assert len(calls) == 3


def test_layer_change_recomputes(tmp_path, caplog):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    cache.ensure(path, _meta(layer_ids=["a"]), comp)
    with caplog.at_level(logging.INFO, logger="notodata.cache"):
        assert cache.ensure(path, _meta(layer_ids=["b"]), comp) is True
    assert "layer_ids" in caplog.text
    assert "zoom" not in caplog.text
    assert len(calls) == 2


def test_building_ids_sha256_forces_recompute_at_same_row_count(tmp_path, caplog):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    cache.ensure(path, _meta(building_ids=[1, 2, 3]), comp)
    with caplog.at_level(logging.INFO, logger="notodata.cache"):
        assert cache.ensure(path, _meta(building_ids=[1, 2, 4]), comp) is True
    assert "building_ids_sha256" in caplog.text
    assert "row_count" not in caplog.text


def test_pipeline_version_in_sidecar_forces_recompute(tmp_path, monkeypatch):
    path, calls = tmp_path / "x.bin", []
    comp = _compute_factory(calls)
    cache.ensure(path, _meta(), comp)
    monkeypatch.setattr(config, "PIPELINE_VERSION", config.PIPELINE_VERSION + 1)
    assert cache.ensure(path, _meta(), comp) is True


def test_crash_between_data_and_sidecar_leaves_no_valid_sidecar(tmp_path):
    path = tmp_path / "x.bin"
    cache.ensure(path, _meta(layer_ids=["old"]), _compute_factory([]))

    def dying(p):
        p.write_bytes(b"new data")
        raise KeyboardInterrupt  # killed after the data, before the sidecar

    with pytest.raises(KeyboardInterrupt):
        cache.ensure(path, _meta(layer_ids=["new"]), dying)
    assert path.exists()
    assert not cache.sidecar_path(path).exists()
    # the stale sidecar of the old data must not vouch for the new data
    calls = []
    assert cache.ensure(path, _meta(layer_ids=["new"]), _compute_factory(calls)) is True
    assert len(calls) == 1


def test_pipeline_version_guard():
    """Editing chips.py without bumping PIPELINE_VERSION fails here.

    After a bump, run `python scripts/update_pipeline_hash.py`.
    """
    assert cache.pipeline_guard_problem() is None


def _guard_fixture(tmp_path):
    lock = tmp_path / "pipeline_version.json"
    srcs = (tmp_path / "chips.py", tmp_path / "tiles.py")
    for src in srcs:
        src.write_text("a = 1\n")
    cache.update_pipeline_lock(lock_path=lock, source_paths=srcs)
    return lock, srcs


@pytest.mark.parametrize("which", [0, 1])
def test_guard_trips_on_either_source(tmp_path, which):
    lock, srcs = _guard_fixture(tmp_path)
    assert cache.pipeline_guard_problem(lock_path=lock, source_paths=srcs) is None
    srcs[which].write_text("a = 2\n")
    assert "PIPELINE_VERSION" in cache.pipeline_guard_problem(lock_path=lock, source_paths=srcs)


def test_guarded_sources_include_tiles():
    assert {p.name for p in cache.GUARDED_SOURCES} == {"chips.py", "tiles.py"}


def test_update_refuses_without_version_bump_then_succeeds_after(tmp_path, monkeypatch):
    lock, srcs = _guard_fixture(tmp_path)
    srcs[1].write_text("a = 2\n")
    with pytest.raises(cache.PipelineVersionError):
        cache.update_pipeline_lock(lock_path=lock, source_paths=srcs)
    assert cache.pipeline_guard_problem(lock_path=lock, source_paths=srcs) is not None
    monkeypatch.setattr(config, "PIPELINE_VERSION", config.PIPELINE_VERSION + 1)
    cache.update_pipeline_lock(lock_path=lock, source_paths=srcs)
    assert cache.pipeline_guard_problem(lock_path=lock, source_paths=srcs) is None


def test_update_script_writes_given_lock_and_leaves_tracked_lock(tmp_path):
    root = repo_root()
    tracked = cache.LOCK_PATH.read_bytes()
    lock = tmp_path / "lock.json"
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "update_pipeline_hash.py"), str(lock)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.returncode == 0
    assert json.loads(lock.read_text()) == json.loads(tracked)
    assert cache.LOCK_PATH.read_bytes() == tracked


def test_chip_index_written_with_sidecar(tmp_home):
    idx = pd.DataFrame(
        {
            "building_id": [2, 1],
            "status": ["missing", "ok"],
            "path": ["", "chips/L/0/1.png"],
            "sha256": ["", "ab"],
        }
    )
    blds = pd.DataFrame({"building_id": [1, 2]})
    calls = []

    def build():
        calls.append(1)
        return idx

    assert cache.ensure_chip_index("L", blds, build) is True
    assert cache.ensure_chip_index("L", blds, build) is False
    assert len(calls) == 1
    path = paths.chip_index_path("L")
    got = pd.read_parquet(path)
    assert got["building_id"].tolist() == [1, 2]
    meta = json.loads(cache.sidecar_path(path).read_text())
    assert meta["layer_ids"] == ["L"]
    assert meta["row_count"] == 2
    assert meta["zoom"] == config.CONFIG.tile_chip.zoom
    assert meta["chip_size"] == config.CONFIG.tile_chip.chip_size_px
    assert meta["encoder_id"] is None
    # different buildings, same count -> rebuilt
    assert cache.ensure_chip_index("L", pd.DataFrame({"building_id": [1, 5]}), build) is True


def test_chips_module_is_what_is_hashed():
    assert Path(chips.__file__) == cache.CHIPS_SOURCE
