"""CLI wiring: every phase through `main`, on synthetic data, with injected network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from conftest import StubFetcher, synthetic_buildings
from notodata import audit, cli, layers, paths, source
from notodata.config import CONFIG
from test_source import make_grid_xml

POST = layers.by_role("post")[0].layer_id
PRE = layers.by_role("pre")[0].layer_id
BASE = CONFIG.baseline_imagery.layer_id
LAYER_ARGS = ["--layers", f"{BASE},{POST},{PRE}"]


@pytest.fixture
def env(tmp_home, tmp_path):
    raw = tmp_home / "raw"
    raw.mkdir()
    df = synthetic_buildings(n=400, seed=1)
    df.loc[df["damage_val"].isna(), "damage_val"] = df["label"]
    gdf = gpd.GeoDataFrame(
        {
            "damage_val": df["damage_val"].to_numpy(),
            "damage_2": df["label"].to_numpy(),
            "municipality": ["m"] * len(df),
        },
        geometry=[Point(x, y).buffer(1e-5) for x, y in zip(df["lon"], df["lat"], strict=True)],
        crs="EPSG:4326",
    )
    gpkg = raw / CONFIG.buildings.filename
    gdf.to_file(gpkg, layer="v2.5", driver="GPKG", engine="pyogrio")
    root = tmp_path / "repo"
    root.mkdir()
    ctx = cli.Context(
        get=lambda url: make_grid_xml(),
        fetcher_factory=lambda layer: StubFetcher(),
        gpkg_md5=hashlib.md5(gpkg.read_bytes()).hexdigest(),
        root=root,
    )
    return ctx


def run(ctx, *argv) -> int:
    return cli.main(list(argv), ctx)


def _fill_marks(mark_for) -> None:
    key = pd.read_csv(paths.audit_dir() / audit.KEY_NAME)
    marks = pd.DataFrame(
        {
            "file": key["file"],
            "mark": [mark_for(r) for r in key.itertuples()],
        }
    )
    marks.to_csv(paths.audit_dir() / audit.MARKS_NAME, index=False)


def _upto_audit_export(ctx):
    for stage in (["source"], ["labels"], ["geometry"], ["chips", *LAYER_ARGS], ["coverage"]):
        assert run(ctx, *stage) == 0
    assert run(ctx, "audit-export") == 0


def _ingest(ctx) -> int:
    return run(ctx, "audit-ingest", "--rater", "tester", "--date", "2026-10-23")


def _after_decision(ctx):
    assert run(ctx, "imagery") == 0
    assert run(ctx, "spatial") == 0
    assert run(ctx, "folds") == 0
    assert run(ctx, "splits", "--seeds", "1000", "1001") == 0
    assert run(ctx, "clients", "--N", "5", "--seeds", "1000") == 0
    assert run(ctx, "table") == 0
    assert run(ctx, "manifest") == 0


def _logs():
    lines = []
    for p in sorted(paths.logs_dir().glob("run_*.jsonl")):
        lines += [json.loads(line) for line in p.read_text().splitlines()]
    return lines


def test_unknown_stage_exits_nonzero(env):
    with pytest.raises(SystemExit) as exc:
        run(env, "no-such-stage")
    assert exc.value.code != 0


def test_source_stage_builds_the_table_and_skips_when_valid(env):
    assert run(env, "source") == 0
    table = pd.read_parquet(source.source_table_path())
    assert len(table) == 400
    assert table["MMI_shape"].isna().all()
    assert (env.root / "reports" / "source_checks.csv").exists()
    assert run(env, "source") == 0
    runs = [r for r in _logs() if r["stage"] == "source"]
    assert [r["recomputed"] for r in runs] == [True, False]


def test_source_stage_recomputes_when_the_grid_changes(env):
    run(env, "source")
    (paths.raw_dir() / CONFIG.shakemap.grid_file).write_bytes(make_grid_xml() + b"\n")
    run(env, "source")
    assert [r["recomputed"] for r in _logs() if r["stage"] == "source"] == [True, True]


def test_run_log_carries_the_required_fields(env):
    run(env, "source")
    record = _logs()[-1]
    for field in (
        "stage",
        "git_commit",
        "pipeline_version",
        "config_hash",
        "inputs",
        "outputs",
        "recomputed",
        "cache_key",
        "utc",
    ):
        assert field in record, field
    assert isinstance(record["pipeline_version"], int)
    assert len(record["config_hash"]) == 64
    assert list(paths.logs_dir().glob("run_*.jsonl"))


def test_labels_geometry_and_class_count_reports(env):
    run(env, "source")
    assert run(env, "labels") == 0
    assert run(env, "geometry") == 0
    raw = pd.read_csv(env.root / "reports" / "class_counts_raw.csv")
    assert {"code", "count", "old_count", "difference"} <= set(raw.columns)
    assert (env.root / "reports" / "class_counts_mapped.csv").exists()
    ref = json.loads((env.root / "reports" / "geometry_reference.json").read_text())
    assert len(ref["lat0"].split(".")[1]) == 10


def test_fetch_dry_run_counts_tiles_without_fetching(env, capsys):
    run(env, "source")
    assert run(env, "fetch", "--dry-run", *LAYER_ARGS) == 0
    out = capsys.readouterr().out
    assert POST in out
    assert not paths.tiles_dir(POST).exists()


def test_every_stage_end_to_end(env):
    _upto_audit_export(env)
    audit_dir = paths.audit_dir()
    assert len(list((audit_dir / "blind").glob("*.png"))) == 120
    key_before = (audit_dir / audit.KEY_NAME).read_bytes()
    assert run(env, "audit-export") == 0  # does not redraw a salt over finished marks
    assert (audit_dir / audit.KEY_NAME).read_bytes() == key_before
    _fill_marks(lambda r: "no_visible_damage")
    assert _ingest(env) == 0
    decision = json.loads((env.root / "reports" / "imagery_decision.json").read_text())
    assert decision["rule_row"] == 3
    assert decision["post_layer"] == POST
    assert decision["pre_layer"] == PRE
    _after_decision(env)
    table = pd.read_parquet(paths.buildings_table_path())
    assert len(table) == decision["n_kept"]
    assert table["MMI_shape"].isna().all()
    manifest_text = (env.root / "MANIFEST.sha256").read_text()
    assert "buildings.parquet" in manifest_text
    # second pass: everything skips
    before = len(_logs())
    for stage in (["imagery"], ["spatial"], ["folds"], ["table"], ["source"], ["labels"]):
        run(env, *stage)
    assert [r["recomputed"] for r in _logs()[before:]] == [False] * 6
    # report fills AUTO blocks without touching hand text
    doc = env.root / "DATASET.md"
    doc.write_text("hand\n<!-- AUTO:moran --> x <!-- /AUTO -->\nmore hand\n")
    assert run(env, "report") == 0
    text = doc.read_text()
    assert text.startswith("hand\n")
    assert text.endswith("more hand\n")
    assert "bin_hi_km" in text


def test_a_rejected_pre_layer_exits_with_the_reaudit_code(env):
    _upto_audit_export(env)
    key = pd.read_csv(paths.audit_dir() / audit.KEY_NAME)
    pre_collapsed = key[(key["layer_id"] == PRE) & (key["label_raw"] == 1)]["file"]
    _fill_marks(lambda r: "visible_damage" if r.file in set(pre_collapsed) else "cannot_tell")
    assert _ingest(env) == cli.EXIT_NEEDS_REAUDIT
    assert not (env.root / "reports" / "imagery_decision.json").exists()


def test_verify_rebuilds_and_matches_then_catches_tampering(env):
    _upto_audit_export(env)
    _fill_marks(lambda r: "no_visible_damage")
    _ingest(env)
    _after_decision(env)
    assert run(env, "verify", "--reuse-tiles") == 0
    result = json.loads((env.root / "reports" / "verify_result.json").read_text())
    assert result["status"] == "pass"
    assert result["mode"] == "reuse-tiles"
    manifest = env.root / "MANIFEST.sha256"
    lines = manifest.read_text().splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    manifest.write_text("\n".join(lines) + "\n")
    assert run(env, "verify", "--reuse-tiles") == 1
    assert json.loads((env.root / "reports" / "verify_result.json").read_text())["status"] == "fail"


def test_build_stops_cleanly_at_the_audit(env, capsys):
    assert run(env, "build", *LAYER_ARGS) == 0
    assert "audit" in capsys.readouterr().out.lower()
    assert (env.root / "reports" / "coverage_by_layer.csv").exists()
    assert not Path(env.root / "reports" / "imagery_decision.json").exists()


def test_client_counts_come_from_config():
    from notodata.config import CLIENT_COUNTS

    assert CLIENT_COUNTS == (50, 100, 200)


def test_build_propagates_a_failing_sub_stage(env):
    (paths.raw_dir() / CONFIG.buildings.filename).unlink()
    assert run(env, "build", *LAYER_ARGS) == cli.EXIT_MISSING_INPUT


def test_a_wrong_md5_is_a_clean_exit_2(env, capsys):
    bad = cli.Context(**{**env.__dict__, "gpkg_md5": "0" * 32})
    assert run(bad, "source") == cli.EXIT_MISSING_INPUT
    assert "MD5" in capsys.readouterr().err


def test_geometry_reference_is_the_table_origin(env):
    from notodata import geometry

    _upto_audit_export(env)
    _fill_marks(lambda r: "no_visible_damage")
    _ingest(env)
    _after_decision(env)
    ref = json.loads((env.root / "reports" / "geometry_reference.json").read_text())
    table = pd.read_parquet(paths.buildings_table_path())
    x, y = geometry.project(table["lat"], table["lon"], float(ref["lat0"]), float(ref["lon0"]))
    assert abs(x - table["x_m"]).max() < 1e-3  # the reference is printed to 10 decimals
    assert abs(y - table["y_m"]).max() < 1e-3
