"""`notodata <stage>`: one idempotent command per phase.

Thread-limiting environment variables are set in `main`, before numpy (or
anything that transitively imports it) is imported in this process, so
BLAS-backed libraries never see more than one thread -- required for
bit-stable KMeans fits and distance sums across machines with a different
core count. For that reason this module imports only the standard library at
the top; every stage imports what it needs when it runs.

Each stage reads only its declared inputs. A stage that writes a data file
keeps a sidecar (`cache.ensure`) holding the hashes of those inputs and skips
when it still matches; report-only stages rewrite their file only if the
bytes change. Every stage appends one JSON line to
`<NOTODATA_HOME>/logs/run_<utc>.jsonl`. Exit codes: 0 ok, 1 verify failed,
2 an input is missing, 3 the A1 pre layer was rejected (re-audit needed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

EXIT_OK = 0
EXIT_VERIFY_FAILED = 1
EXIT_MISSING_INPUT = 2
EXIT_NEEDS_REAUDIT = 3

MANIFEST_NAME = "MANIFEST.sha256"
log = logging.getLogger("notodata.cli")


class CliError(RuntimeError):
    """An input a stage needs is missing; the message says what to do."""


def set_single_threaded_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Force OMP/MKL/OPENBLAS to one thread each; return what was set.

    Must run before numpy is imported in this process, because these
    libraries read the variable once, at load time.
    """
    target = os.environ if env is None else env
    for name in _THREAD_ENV_VARS:
        target[name] = "1"
    return {name: target[name] for name in _THREAD_ENV_VARS}


# -- run log and context ---------------------------------------------------


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_COMMIT", "unknown")


def _config_hash() -> str:
    from notodata.config import CONFIG

    text = json.dumps(asdict(CONFIG), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class RunLog:
    """Appends one JSON line per stage to `logs/run_<utc-timestamp>.jsonl`."""

    def __init__(self) -> None:
        self._name = f"run_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%S%fZ')}.jsonl"
        self._static: dict | None = None

    def record(self, root: Path, stage: str, **fields) -> None:
        from notodata import paths
        from notodata.config import PIPELINE_VERSION

        if self._static is None:
            self._static = {
                "git_commit": _git_commit(root),
                "pipeline_version": PIPELINE_VERSION,
                "config_hash": _config_hash(),
            }
        line = {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "stage": stage,
            **self._static,
            **fields,
        }
        out = paths.logs_dir() / self._name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")


@dataclass
class Context:
    """What a stage may not decide for itself; tests inject fakes here, never the network."""

    get: Callable[[str], bytes] | None = None
    """Download callable for `grid.xml` (default: `source.http_get`)."""
    fetcher_factory: Callable | None = None
    """`layer -> TileFetcher` (default: `HttpTileFetcher`)."""
    gpkg_md5: str | None = None
    """Expected MD5 of the gpkg (default: the pinned value in config)."""
    root: Path = field(default_factory=Path)
    """Repo root: holds `reports/`, `DATASET.md` and `MANIFEST.sha256`."""
    log: RunLog = field(default_factory=RunLog)

    def record(self, stage: str, **fields) -> None:
        self.log.record(self.root, stage, **fields)

    def reports(self) -> Path:
        from notodata import paths

        return paths.reports_dir(self.root)


def _sha(path: Path) -> str:
    from notodata.io import sha256_file

    return sha256_file(path)


def _hashes(files: list[Path]) -> dict[str, str]:
    return {p.name: _sha(p) for p in files}


def _cached(ctx: Context, stage: str, out: Path, inputs: list[Path], key: dict, compute) -> bool:
    """Run `compute(out)` unless `out` has a sidecar matching these inputs and key."""
    from notodata import cache
    from notodata.config import PIPELINE_VERSION

    meta = {"stage": stage, "pipeline_version": PIPELINE_VERSION, "inputs": _hashes(inputs), **key}
    ran = cache.ensure(out, meta, compute)
    ctx.record(stage, inputs=meta["inputs"], outputs={out.name: _sha(out)}, recomputed=ran,
               cache_key=meta)  # fmt: skip
    return ran


def _write_changed(path: Path, data: bytes) -> bool:
    from notodata.io import atomic_write

    if path.exists() and path.read_bytes() == data:
        return False
    atomic_write(path, data)
    return True


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode()


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def _write_reports(ctx: Context, stage: str, files: dict[Path, bytes], inputs: list[Path]) -> bool:
    changed = [_write_changed(p, data) for p, data in files.items()]
    ctx.record(stage, inputs=_hashes(inputs), outputs=_hashes(list(files)),
               recomputed=any(changed), cache_key={})  # fmt: skip
    return any(changed)


# -- loaders ---------------------------------------------------------------


def _labels_path() -> Path:
    from notodata import paths

    return paths.derived_dir() / "labels.parquet"


def _folds_path() -> Path:
    from notodata import paths

    return paths.derived_dir() / "folds.parquet"


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise CliError(f"{path} is missing; {hint}")
    return path


def _buildings(need_labels: bool = True):
    """Source rows joined to the mapped labels, sorted by building id."""
    import pandas as pd

    from notodata import source

    df = pd.read_parquet(_need(source.source_table_path(), "run `notodata source`"))
    if need_labels:
        lab = pd.read_parquet(_need(_labels_path(), "run `notodata labels`"))
        df = df.merge(lab, on="building_id", validate="1:1")
    return df.sort_values("building_id", kind="mergesort").reset_index(drop=True)


def _decision(ctx: Context) -> dict:
    path = _need(ctx.reports() / "imagery_decision.json", "run `notodata audit-ingest`")
    return json.loads(path.read_text())


def _read_index(layer: str):
    import pandas as pd

    from notodata import paths

    return pd.read_parquet(_need(paths.chip_index_path(layer), f"run `notodata chips` ({layer})"))


def _ok_ids(layer: str) -> set[int]:
    index = _read_index(layer)
    return {int(b) for b in index.loc[index["status"] == "ok", "building_id"]}


def _decision_layers(decision: dict) -> list[str]:
    layers = [decision["post_layer"]]
    if decision["pre_layer"] is not None and decision["pre_available"]:
        layers.append(decision["pre_layer"])
    return layers


def _chip_layers(decision: dict) -> list[str]:
    """Every layer the decision names: `_working` checks it against the pre layer's index."""
    layers = [decision["post_layer"]]
    if decision["pre_layer"] is not None:
        layers.append(decision["pre_layer"])
    return layers


def _core_inputs(ctx: Context) -> list[Path]:
    from notodata import paths, source

    decision = ctx.reports() / "imagery_decision.json"
    layers = _decision_layers(_decision(ctx))
    return [
        source.source_table_path(),
        _labels_path(),
        decision,
        *[paths.chip_index_path(layer) for layer in layers],
    ]


def _working(ctx: Context):
    """(kept buildings with labels, flags, coverage result, decision), sorted by id."""
    from notodata import coverage, imagery

    decision = _decision(ctx)
    buildings = _buildings()
    pre_layer = decision["pre_layer"]
    res = coverage.apply_coverage_rules(
        buildings,
        _ok_ids(decision["post_layer"]),
        None if pre_layer is None else _ok_ids(pre_layer),
    )
    if len(res.kept) != decision["n_kept"] or res.pre_available != decision["pre_available"]:
        raise CliError("imagery_decision.json no longer matches the chip indexes; rerun the stages")
    kept = res.kept.sort_values("building_id", kind="mergesort").reset_index(drop=True)
    pre_index = _read_index(pre_layer) if pre_layer is not None else None
    flags = imagery.flags_from_indexes(
        kept["building_id"], _read_index(decision["post_layer"]), pre_index, res.pre_available
    )
    return kept, flags, res, decision


def _origin() -> tuple[float, float]:
    """The one projection origin: the mean over ALL source rows (spec), kept or not."""
    from notodata import geometry

    df = _buildings(need_labels=False)
    return geometry.reference_point(df["lat"], df["lon"])


def _frame(ctx: Context):
    """The working frame stages 12+ use: kept rows, labels, flags and projected metres."""
    from notodata import geometry

    kept, flags, res, decision = _working(ctx)
    lat0, lon0 = _origin()
    df = kept.merge(flags, on="building_id", validate="1:1")
    df["x_m"], df["y_m"] = geometry.project(df["lat"], df["lon"], lat0, lon0)
    return df, res, decision


# -- stages: source, labels, geometry -------------------------------------


def stage_source(ctx: Context, args) -> int:
    from notodata import paths, source
    from notodata.config import CONFIG

    gpkg = source.gpkg_path()
    _need(
        gpkg,
        f"download {CONFIG.buildings.filename} from Zenodo {CONFIG.buildings.concept_doi} "
        f"into {paths.raw_dir()}",
    )
    md5 = ctx.gpkg_md5 or CONFIG.buildings.md5
    grid = source.fetch_grid(get=ctx.get or source.http_get)
    gpkg_sha = source.record_source_file(gpkg)["sha256"]

    def compute(out: Path) -> None:
        table, checks = source.build_source_table(
            source.read_buildings(gpkg, md5), source.parse_grid(grid.read_bytes())
        )
        source.write_source_table(table, out)
        _write_changed(ctx.reports() / "source_checks.csv", _csv_bytes(_checks_frame(checks)))

    _cached(
        ctx,
        "source",
        source.source_table_path(),
        [grid],
        {"gpkg_md5": md5, "gpkg_sha256": gpkg_sha},
        compute,
    )
    return EXIT_OK


def _checks_frame(checks: dict):
    import pandas as pd

    rows = [(k, "", v) for k, v in checks.items() if not isinstance(v, dict)]
    for k in ("n_nonpositive", "units"):
        rows += [(k, col, v) for col, v in checks[k].items()]
    return pd.DataFrame(rows, columns=["check", "column", "value"])


def _class_counts(df):
    import pandas as pd

    from notodata.config import CONFIG

    old = CONFIG.old_class_counts
    codes = {0: old.code_0, 1: old.code_1, 9: old.code_9, 99: old.code_99}
    inside = df[df["bbox_outside"] == 0]
    rows = [
        (c, int((df["label_raw"] == c).sum()), int((inside["label_raw"] == c).sum()), n)
        for c, n in codes.items()
    ]
    rows.append(("total", len(df), len(inside), old.total))
    out = pd.DataFrame(rows, columns=["code", "count", "in_bbox", "old_count"])
    out["difference"] = out["count"] - out["old_count"]
    return out


def _mapped_counts(df):
    grouped = df.assign(y_collapse=df["y_collapse"].fillna(-1)).groupby(
        ["label_raw", "y_collapse", "observable"], as_index=False
    )
    out = grouped.size().rename(columns={"size": "count"})
    out["y_collapse"] = out["y_collapse"].map(lambda v: "NaN" if v < 0 else str(int(v)))
    return out


def stage_labels(ctx: Context, args) -> int:
    from notodata import labels, source

    src = _need(source.source_table_path(), "run `notodata source`")

    def compute(out: Path) -> None:
        df = _buildings(need_labels=False)
        mapped = labels.map_labels(df)
        mapped.insert(0, "building_id", df["building_id"].to_numpy())
        mapped["observable"] = mapped["observable"].astype("int8")
        out.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_parquet(out, index=False)
        both = df.merge(mapped, on="building_id", validate="1:1")
        reports = ctx.reports()
        _write_changed(reports / "class_counts_raw.csv", _csv_bytes(_class_counts(both)))
        _write_changed(reports / "class_counts_mapped.csv", _csv_bytes(_mapped_counts(both)))

    _cached(ctx, "labels", _labels_path(), [src], {}, compute)
    return EXIT_OK


def stage_geometry(ctx: Context, args) -> int:
    from notodata import source

    src = _need(source.source_table_path(), "run `notodata source`")
    lat0, lon0 = _origin()
    ref = {
        "basis": "all source rows, before any coverage restriction",
        "lat0": f"{lat0:.10f}",
        "lon0": f"{lon0:.10f}",
        "n_rows": len(_buildings(need_labels=False)),
    }
    _write_reports(ctx, "geometry", {ctx.reports() / "geometry_reference.json": _json_bytes(ref)},
                   [src])  # fmt: skip
    return EXIT_OK


# -- stages: imagery fetch, chips, coverage -------------------------------


def _layer_ids(args) -> list[str]:
    from notodata.layers import LAYERS, get_layer

    if not args.layers:
        return [layer.layer_id for layer in LAYERS]
    ids = args.layers.split(",")
    for layer_id in ids:
        get_layer(layer_id)  # KeyError names an unknown layer
    return ids


def _fetcher(ctx: Context, layer_id: str):
    from notodata.layers import get_layer
    from notodata.tiles import HttpTileFetcher

    layer = get_layer(layer_id)
    if ctx.fetcher_factory is not None:
        return ctx.fetcher_factory(layer)
    return HttpTileFetcher(layer.layer_id, layer.url_template, layer.ext)


def unique_tiles(buildings) -> list[tuple[int, int]]:
    """Every tile any chip window touches, sorted."""
    from notodata import tiles
    from notodata.config import CONFIG

    gx, gy = tiles.global_pixel(buildings["lat"], buildings["lon"], CONFIG.tile_chip.zoom)
    x0, y0 = tiles.window_origin(gx, gy)
    found: set[tuple[int, int]] = set()
    for a, b in zip(x0, y0, strict=True):
        found.update(tiles.covering_tiles(a, b))
    return sorted(found)


def _fetch_layer(ctx: Context, layer_id: str, wanted: list[tuple[int, int]]) -> None:
    from tqdm import tqdm

    from notodata.config import CONFIG

    fetcher = _fetcher(ctx, layer_id)
    ids = [(layer_id, CONFIG.tile_chip.zoom, x, y) for x, y in wanted]
    # Forced on (disable=False): `run_all.sh` pipes through tee, which would hide a default bar.
    # The bar counts ids handed to the workers, which runs a little ahead of finished tiles.
    progress = tqdm(ids, desc=layer_id, unit="tile", mininterval=10, disable=False)
    if hasattr(fetcher, "prefetch_many"):
        fetcher.prefetch_many(progress)
    else:
        for tile_id in progress:
            fetcher.get(*tile_id)
    ctx.record("fetch", layer_id=layer_id, tiles=len(wanted), recomputed=True,
               inputs={}, outputs={}, cache_key={})  # fmt: skip


def stage_fetch(ctx: Context, args) -> int:
    from notodata.config import CONFIG

    wanted = unique_tiles(_buildings(need_labels=False))
    for layer_id in _layer_ids(args):
        print(f"{layer_id}: {len(wanted)} unique tiles at zoom {CONFIG.tile_chip.zoom}")
        if not args.dry_run:
            _fetch_layer(ctx, layer_id, wanted)
    return EXIT_OK


def stage_chips(ctx: Context, args) -> int:
    from notodata import cache, chips

    buildings = _buildings()
    for layer_id in _layer_ids(args):
        fetcher = _fetcher(ctx, layer_id)
        ran = cache.ensure_chip_index(
            layer_id,
            buildings,
            lambda f=fetcher, lid=layer_id: chips.build_chips(f, lid, buildings),
        )
        ctx.record("chips", layer_id=layer_id, recomputed=ran, inputs={}, outputs={}, cache_key={})
    return EXIT_OK


def _present_layers() -> list[str]:
    from notodata import paths
    from notodata.layers import LAYERS

    return [x.layer_id for x in LAYERS if paths.chip_index_path(x.layer_id).exists()]


def stage_coverage(ctx: Context, args) -> int:
    from notodata import coverage
    from notodata.layers import by_role

    buildings = _buildings()
    present = _present_layers()
    if not present:
        raise CliError("no chip indexes found; run `notodata chips`")
    table = coverage.coverage_by_layer({lid: _read_index(lid) for lid in present}, buildings)
    overall = table[table["group"] == "overall"].set_index("layer_id")["ok"]
    n_ok = {lid: int(overall[lid]) for lid in present}
    posts = {x.layer_id: n_ok[x.layer_id] for x in by_role("post") if x.layer_id in n_ok}
    pres = {x.layer_id: n_ok[x.layer_id] for x in by_role("pre") if x.layer_id in n_ok}
    candidates = {
        "n_ok": n_ok,
        "best_post": coverage.best_post_layer(posts) if posts else None,
        "ranked_pre": coverage.rank_pre_layers(pres) if pres else [],
    }
    files = {
        ctx.reports() / "coverage_by_layer.csv": _csv_bytes(table),
        ctx.reports() / "coverage_candidates.json": _json_bytes(candidates),
    }
    _write_reports(ctx, "coverage", files, [paths_index(lid) for lid in present])
    return EXIT_OK


def paths_index(layer_id: str) -> Path:
    from notodata import paths

    return paths.chip_index_path(layer_id)


# -- stages: audit ---------------------------------------------------------


def _candidates(ctx: Context) -> dict:
    path = _need(ctx.reports() / "coverage_candidates.json", "run `notodata coverage`")
    cand = json.loads(path.read_text())
    if cand["best_post"] is None or not cand["ranked_pre"]:
        raise CliError("the audit needs at least one chipped post layer and one pre layer")
    return cand


def stage_audit_export(ctx: Context, args) -> int:
    from notodata import audit, paths
    from notodata.config import CONFIG

    key = paths.audit_dir() / audit.KEY_NAME
    if key.exists() and not args.force:
        print(f"{key} exists; keeping it (a new draw would orphan the marks). Use --force.")
        return EXIT_OK
    cand = _candidates(ctx)
    post, pre, seam = cand["best_post"], cand["ranked_pre"][0], CONFIG.baseline_imagery.layer_id
    result = audit.export(
        _buildings(),
        post_layer=post,
        pre_layer=pre,
        post_ok=_ok_ids(post),
        seam_ok=_ok_ids(seam),
        pre_ok=_ok_ids(pre),
    )
    layers = {"seam": seam, "post": post, "pre": pre}
    (paths.audit_dir() / "audit_layers.json").write_bytes(_json_bytes(layers))
    print(f"blinded chips in {result.blind_dir}; mark {result.marks_path}")
    ctx.record("audit-export", layers=layers, recomputed=True, inputs={}, outputs={}, cache_key={})
    return EXIT_OK


def _decide(ctx: Context, counts: dict, cand: dict, layers: dict) -> dict:
    from notodata import coverage

    coverage.check_pre_layer(layers["pre"], counts["n_pre"], cand["ranked_pre"])
    row = coverage.rule_row(counts["n_seam"], counts["n_post"])
    post, pre = coverage.choose_layers(row, cand["best_post"], layers["pre"])
    buildings = _buildings()
    res = coverage.apply_coverage_rules(
        buildings, _ok_ids(post), None if pre is None else _ok_ids(pre)
    )
    coverage.write_dropped_by_class(res.dropped_by_class, ctx.root)
    return coverage.build_decision(
        **counts,
        best_post=cand["best_post"],
        a1_pre=layers["pre"],
        ranked_pre=cand["ranked_pre"],
        n_buildings=len(buildings),
        n_kept=len(res.kept),
        pre_available=res.pre_available,
        post_coverage=res.post_coverage,
        pre_coverage=res.pre_coverage,
    )


def stage_audit_ingest(ctx: Context, args) -> int:
    from notodata import audit, coverage, paths

    layers = json.loads(
        _need(paths.audit_dir() / "audit_layers.json", "run audit-export").read_text()
    )
    marks = Path(args.marks) if args.marks else paths.audit_dir() / audit.MARKS_NAME
    result = audit.ingest(
        marks,
        paths.audit_dir() / audit.KEY_NAME,
        rater=args.rater,
        date=args.date,
        seam_layer=layers["seam"],
        post_layer=layers["post"],
        pre_layer=layers["pre"],
        root=ctx.root,
    )
    decision = _decide(ctx, result.counts, _candidates(ctx), layers)
    out = coverage.write_decision(decision, ctx.root)
    inputs, outputs = {marks.name: _sha(marks)}, {out.name: _sha(out)}
    ctx.record("audit-ingest", counts=result.counts, recomputed=True, inputs=inputs,
               outputs=outputs, cache_key={})  # fmt: skip
    return EXIT_OK


# -- stages: imagery .. clients -------------------------------------------


def stage_imagery(ctx: Context, args) -> int:
    from notodata import imagery

    df, res, decision = _frame(ctx)
    status = {
        lid: _read_index(lid).set_index("building_id")["status"].reindex(df["building_id"])
        .fillna("missing").value_counts()
        for lid in _decision_layers(decision)
    }  # fmt: skip
    cov = imagery.coverage_by_pattern(df, status)
    files = {
        ctx.reports() / "coverage_by_pattern.csv": _csv_bytes(cov),
        ctx.reports() / "dropped_by_post_coverage.csv": _csv_bytes(res.dropped_by_class),
    }
    _write_reports(ctx, "imagery", files, _core_inputs(ctx))
    return EXIT_OK


def stage_spatial(ctx: Context, args) -> int:
    from notodata import spatial

    df, _, _ = _frame(ctx)
    lab = df[df["y_collapse"].notna()]
    table, block_km = spatial.run(lab["building_id"], lab[["x_m", "y_m"]], lab["y_collapse"])
    files = {
        ctx.reports() / "moran.csv": _csv_bytes(table),
        ctx.reports() / "block_size.json": _json_bytes({"L_km": block_km}),
    }
    _write_reports(ctx, "spatial", files, _core_inputs(ctx))
    return EXIT_OK


def _block_size(ctx: Context) -> float:
    path = _need(ctx.reports() / "block_size.json", "run `notodata spatial`")
    return float(json.loads(path.read_text())["L_km"])


def stage_folds(ctx: Context, args) -> int:
    from notodata import folds

    block_json = ctx.reports() / "block_size.json"
    inputs = [*_core_inputs(ctx), _need(block_json, "run `notodata spatial`")]

    def compute(out: Path) -> None:
        df, _, _ = _frame(ctx)
        cols = folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], _block_size(ctx))
        cols.insert(0, "building_id", df["building_id"].to_numpy())
        out.parent.mkdir(parents=True, exist_ok=True)
        cols.to_parquet(out, index=False)
        both = df.merge(cols, on="building_id", validate="1:1")
        _write_changed(ctx.reports() / "folds.csv", _csv_bytes(folds.fold_table(both)))

    _cached(ctx, "folds", _folds_path(), inputs, {}, compute)
    return EXIT_OK


def _fold_inputs(ctx: Context) -> list[Path]:
    return [*_core_inputs(ctx), _need(_folds_path(), "run `notodata folds`")]


def _with_folds(ctx: Context):
    import pandas as pd

    df, res, decision = _frame(ctx)
    cols = pd.read_parquet(_folds_path())
    return df.merge(cols, on="building_id", validate="1:1"), res, decision


def stage_splits(ctx: Context, args) -> int:
    import pandas as pd

    from notodata import paths, splits

    inputs = _fold_inputs(ctx)
    df, _, _ = _with_folds(ctx)
    for seed in args.seeds:

        def compute(out: Path, seed=seed) -> None:
            role = splits.assign_roles(df["building_id"], df["fold"], df[["x_m", "y_m"]], seed)
            splits.write_split(role, seed)

        _cached(ctx, "splits", paths.split_path(seed), inputs, {"seed": seed}, compute)
    found = {
        int(re.fullmatch(r"seed_(\d+)", p.stem).group(1)): pd.read_parquet(p)
        for p in sorted(paths.splits_dir().glob("seed_*.parquet"))
    }
    out = ctx.reports() / "splits_summary.csv"
    _write_changed(out, _csv_bytes(splits.summary(found, df)))
    return EXIT_OK


def stage_clients(ctx: Context, args) -> int:
    import pandas as pd

    from notodata import clients, paths

    inputs = _fold_inputs(ctx)
    df, _, _ = _with_folds(ctx)
    for n in args.N:
        for seed in args.seeds:
            split_file = _need(paths.split_path(seed), f"run `notodata splits --seeds {seed}`")

            def compute(out: Path, n=n, seed=seed, split_file=split_file) -> None:
                assign, meta = clients.build_clients(df, pd.read_parquet(split_file), n, seed)
                clients.write_clients(assign, meta, n, seed)

            _cached(ctx, "clients", paths.client_path(n, seed), [*inputs, split_file],
                    {"N": n, "seed": seed}, compute)  # fmt: skip
    rows = []
    for p in sorted(paths.clients_dir().glob("N*_seed_*.parquet")):
        m = re.fullmatch(r"N(\d+)_seed_(\d+)", p.stem)
        if m:
            rows.append(clients.summary_row(pd.read_parquet(p), df, int(m[1]), int(m[2])))
    if rows:
        _write_changed(ctx.reports() / "clients_summary.csv", _csv_bytes(pd.DataFrame(rows)))
    return EXIT_OK


def stage_embed(ctx: Context, args) -> int:
    from notodata import embeddings

    decision = _decision(ctx)
    encoder = embeddings.get_encoder(args.encoder)
    for layer_id in _decision_layers(decision):
        ran = embeddings.ensure_embeddings(encoder, layer_id, _read_index(layer_id))
        ctx.record("embed", layer_id=layer_id, encoder_id=args.encoder, recomputed=ran,
                   inputs={}, outputs={}, cache_key={})  # fmt: skip
    return EXIT_OK


# -- stages: table, manifest, verify, report ------------------------------


def stage_table(ctx: Context, args) -> int:
    import pandas as pd

    from notodata import paths, table

    inputs = _fold_inputs(ctx)

    def compute(out: Path) -> None:
        kept, flags, res, decision = _working(ctx)
        lab = pd.read_parquet(_labels_path()).set_index("building_id").loc[kept["building_id"]]
        fold_cols = pd.read_parquet(_folds_path()).set_index("building_id").loc[kept["building_id"]]
        pre = decision["pre_layer"]
        built = table.assemble_table(
            kept,
            lab.reset_index(drop=True),
            flags,
            fold_cols.reset_index(drop=True),
            _read_index(decision["post_layer"]),
            _read_index(pre) if pre is not None and res.pre_available else None,
            origin=_origin(),
        )
        table.write_table(built, res.pre_available)

    _cached(ctx, "table", paths.buildings_table_path(), inputs, {}, compute)
    return EXIT_OK


def stage_manifest(ctx: Context, args) -> int:
    from notodata import manifest

    entries = manifest.collect_entries(_decision_layers(_decision(ctx)), root=ctx.root)
    path = ctx.root / MANIFEST_NAME
    manifest.write_manifest(path, entries)
    ctx.record("manifest", entries=len(entries), recomputed=True, inputs={},
               outputs={path.name: _sha(path)}, cache_key={})  # fmt: skip
    return EXIT_OK


def stage_report(ctx: Context, args) -> int:
    from notodata import report

    changed = report.update_dataset_md(ctx.root / "DATASET.md", root=ctx.root)
    ctx.record("report", recomputed=changed, inputs={}, outputs={}, cache_key={})
    return EXIT_OK


_SPLIT_NAME = re.compile(r"derived/splits/seed_(\d+)\.parquet")
_CLIENT_NAME = re.compile(r"derived/clients/N(\d+)_seed_(\d+)\.parquet")


def _rebuild_plan(entries: dict[str, str], layers: list[str]) -> list[list[str]]:
    seeds = sorted({m[1] for n in entries if (m := _SPLIT_NAME.fullmatch(n))}, key=int)
    pairs = [m for n in entries if (m := _CLIENT_NAME.fullmatch(n))]
    plan = [["source"], ["labels"], ["geometry"], ["chips", "--layers", ",".join(layers)]]
    plan += [["imagery"], ["spatial"], ["folds"]]
    if seeds:
        plan.append(["splits", "--seeds", *seeds])
    for n in sorted({m[1] for m in pairs}, key=int):
        cl_seeds = sorted({m[2] for m in pairs if m[1] == n}, key=int)
        plan.append(["clients", "--N", n, "--seeds", *cl_seeds])
    plan.append(["table"])
    return plan


def _stage_verify_inputs(ctx: Context, tmp: Path, reuse_tiles: bool) -> tuple[Path, Path]:
    from notodata import paths

    home, root = tmp / "home", tmp / "root"
    home.mkdir()
    shutil.copytree(paths.raw_dir(), home / "raw")
    if reuse_tiles and (paths.notodata_home() / "tiles").is_dir():
        shutil.copytree(paths.notodata_home() / "tiles", home / "tiles")
    shutil.copytree(ctx.reports(), root / "reports")
    return home, root


class _Swapped:
    """Point `NOTODATA_HOME` at another directory for the length of a `with` block."""

    def __init__(self, home: Path):
        self.home, self.old = home, os.environ.get("NOTODATA_HOME")

    def __enter__(self):
        os.environ["NOTODATA_HOME"] = str(self.home)

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("NOTODATA_HOME", None)
        else:
            os.environ["NOTODATA_HOME"] = self.old


def _tile_hint(layers: list[str], original: Path, rebuilt: Path) -> str:
    from notodata import manifest

    changed = []
    for layer in layers:
        with _Swapped(original):
            before = manifest.tile_fingerprint(layer)
        with _Swapped(rebuilt):
            after = manifest.tile_fingerprint(layer)
        if before != after:
            changed.append(layer)
    return (
        f"tile fingerprints differ for {changed}: GSI imagery changed"
        if changed
        else ("tile fingerprints match: the code changed, not the imagery")
    )


def _rebuild(ctx: Context, entries: dict[str, str], layers: list[str], clean: bool,
             chip_layers: list[str] | None = None):  # fmt: skip
    """Rebuild into a throwaway home; return (problems, tile hint)."""
    from notodata import manifest, paths

    original = paths.notodata_home()
    with tempfile.TemporaryDirectory(dir=os.environ.get("VERIFY_HOME")) as tmp_name:
        home, root = _stage_verify_inputs(ctx, Path(tmp_name), not clean)
        sub = Context(get=ctx.get, fetcher_factory=ctx.fetcher_factory,
                      gpkg_md5=ctx.gpkg_md5, root=root)  # fmt: skip
        with _Swapped(home):
            for argv in _rebuild_plan(entries, chip_layers or layers):
                _run_or_stop(argv, sub)
            problems = manifest.verify_manifest(ctx.root / MANIFEST_NAME, layers, root=root)
        hint = _tile_hint(layers, original, home) if problems and clean else ""
    return problems, hint


def stage_verify(ctx: Context, args) -> int:
    from notodata import manifest

    path = _need(ctx.root / MANIFEST_NAME, "run `notodata manifest`")
    entries = manifest.read_manifest(path)
    decision = _decision(ctx)
    problems, hint = _rebuild(ctx, entries, _decision_layers(decision), args.clean,
                              _chip_layers(decision))  # fmt: skip
    failed = {p.split(":")[0] for p in problems}
    result = {
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "mode": "clean" if args.clean else "reuse-tiles",
        "status": "fail" if problems else "pass",
        "files": {name: ("fail" if name in failed else "pass") for name in sorted(entries)},
    }
    _write_changed(ctx.reports() / "verify_result.json", _json_bytes(result))
    for line in [*problems, hint]:
        if line:
            print(line)
    return EXIT_VERIFY_FAILED if problems else EXIT_OK


# -- build ------------------------------------------------------------------


class StageFailed(Exception):
    """A sub-stage returned a nonzero exit code; `main` returns that code."""

    def __init__(self, code: int):
        self.code = code
        super().__init__(f"sub-stage exited with {code}")


def _run_or_stop(argv: list[str], ctx: Context) -> None:
    code = main(argv, ctx)
    if code:
        raise StageFailed(code)


def stage_build(ctx: Context, args) -> int:
    """Every stage except the two audit stages and `embed`; stops at the audit if undecided."""
    layers = ["--layers", args.layers] if args.layers else []
    for argv in (["source"], ["labels"], ["geometry"], ["fetch", *layers], ["chips", *layers],
                 ["coverage"]):  # fmt: skip
        _run_or_stop(argv, ctx)
    if not (ctx.reports() / "imagery_decision.json").exists():
        print(
            "audit pending: run `audit-export`, mark audit/marks.csv by hand, then "
            "`audit-ingest --rater <name> --date <date>`; then run `build` again."
        )
        return EXIT_OK
    for argv in (["imagery"], ["spatial"], ["folds"], ["splits"], ["clients"], ["table"],
                 ["manifest"], ["report"]):  # fmt: skip
        _run_or_stop(argv, ctx)
    return EXIT_OK


# -- parser and entry point -------------------------------------------------


_LAYERS_HELP = "comma-separated layer ids (default: every registered layer)"

_STAGES = (
    ("source", stage_source, "gpkg + ShakeMap grid -> source table"),
    ("labels", stage_labels, "map damage codes, write class counts"),
    ("geometry", stage_geometry, "projection reference"),
    ("fetch", stage_fetch, "download tiles (or count them with --dry-run)"),
    ("chips", stage_chips, "cut chips and write chip indexes"),
    ("coverage", stage_coverage, "chip coverage per layer and audit candidates"),
    ("audit-export", stage_audit_export, "write the blinded audit set"),
    ("audit-ingest", stage_audit_ingest, "ingest marks and write the imagery decision"),
    ("imagery", stage_imagery, "image flags and coverage by pattern"),
    ("spatial", stage_spatial, "Moran's I and block size L"),
    ("folds", stage_folds, "spatial blocks and folds"),
    ("splits", stage_splits, "per-seed roles"),
    ("clients", stage_clients, "per-(N, seed) clients"),
    ("embed", stage_embed, "chip embeddings"),
    ("table", stage_table, "the building table"),
    ("manifest", stage_manifest, "write MANIFEST.sha256"),
    ("verify", stage_verify, "rebuild and compare with the manifest"),
    ("report", stage_report, "rewrite the AUTO blocks of DATASET.md"),
    ("build", stage_build, "every stage except audit and embed"),
)


def _stage_options() -> dict[str, list[tuple[tuple[str, ...], dict]]]:
    from notodata.config import CLIENT_COUNTS, CONFIG

    layers = (("--layers",), {"default": "", "help": _LAYERS_HELP})
    seeds = (("--seeds",), {"type": int, "nargs": "+", "default": list(CONFIG.test_seeds)})
    return {
        "fetch": [layers, (("--dry-run",), {"action": "store_true"})],
        "chips": [layers],
        "build": [layers],
        "audit-export": [(("--force",), {"action": "store_true", "help": "draw a new sample"})],
        "audit-ingest": [
            (("--rater",), {"required": True}),
            (("--date",), {"required": True}),
            (("--marks",), {"default": ""}),
        ],
        "splits": [seeds],
        "clients": [seeds, (("--N",), {"type": int, "nargs": "+", "default": list(CLIENT_COUNTS)})],
        "embed": [(("--encoder",), {"required": True})],
        "verify": [
            (("--reuse-tiles",), {"action": "store_true", "help": "reuse the tile cache"}),
            (("--clean",), {"action": "store_true", "help": "re-fetch every tile"}),
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notodata")
    sub = p.add_subparsers(dest="stage", metavar="stage")
    options = _stage_options()
    for name, func, text in _STAGES:
        sp = sub.add_parser(name, help=text)
        sp.set_defaults(func=func)
        for flags, kwargs in options.get(name, []):
            sp.add_argument(*flags, **kwargs)
    return p


def main(argv: list[str] | None = None, ctx: Context | None = None) -> int:
    values = set_single_threaded_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name, value in values.items():
        log.info("%s=%s", name, value)

    import numpy as np  # noqa: F401  (imported only after threads are pinned)

    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.stage is None:
        return EXIT_OK
    from notodata.coverage import NeedsReaudit
    from notodata.source import SourceError

    try:
        return args.func(ctx or Context(), args)
    except NeedsReaudit as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_NEEDS_REAUDIT
    except StageFailed as exc:
        return exc.code
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
