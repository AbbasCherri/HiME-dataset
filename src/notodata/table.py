"""The building table (D8, Phase 11): every seed-independent per-building fact.

One row per building, sorted by `building_id`, with a fixed schema so the
file's bytes depend only on its content. Chip paths are relative to
`NOTODATA_HOME`, so the hash does not depend on where the repo lives. The
table holds no seed-dependent column; roles and clients live in per-seed files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from notodata import geometry, paths
from notodata.config import CONFIG, LABEL_CODES
from notodata.imagery import PATTERNS, image_pattern
from notodata.io import write_parquet_deterministic

log = logging.getLogger(__name__)

_SCHEMA_FIELDS = [
    ("building_id", pa.int64()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("x_m", pa.float64()),
    ("y_m", pa.float64()),
    *[(c, pa.float64()) for c in CONFIG.seismic_columns],
    ("label_raw", pa.int16()),
    ("label_source", pa.string()),
    ("label_conflict", pa.int8()),
    ("y_collapse", pa.float64()),
    ("y_collapse_sens", pa.float64()),
    ("observable", pa.float64()),
    ("has_post", pa.int8()),
    ("has_pre", pa.int8()),
    ("G", pa.string()),
    ("post_status", pa.string()),
    ("pre_status", pa.string()),
    ("post_chip", pa.string()),
    ("pre_chip", pa.string()),
    ("post_chip_sha256", pa.string()),
    ("pre_chip_sha256", pa.string()),
    ("block_ix", pa.int32()),
    ("block_iy", pa.int32()),
    ("block_id", pa.string()),
    ("fold", pa.int8()),
]
SCHEMA = pa.schema(_SCHEMA_FIELDS)
COLUMNS = list(SCHEMA.names)

PRE_UNAVAILABLE = "unavailable"
"""`pre_status` of every row when no pre layer was chosen."""


class TableError(ValueError):
    """The building table breaks one or more invariants (all are listed)."""


def _status_lookup(ids: pd.Series, index: pd.DataFrame | None, fallback: str) -> np.ndarray:
    if index is None:
        return np.full(len(ids), fallback, dtype=object)
    by_id = index.set_index("building_id")["status"]
    missing = ids[~ids.isin(by_id.index)]
    if len(missing):
        raise ValueError(
            f"chip index has no row for {len(missing)} buildings, e.g. {missing.iloc[0]}"
        )
    return by_id.loc[ids].to_numpy(dtype=object)


def _origin(buildings: pd.DataFrame, origin: tuple[float, float] | None) -> tuple[float, float]:
    return origin or geometry.reference_point(buildings["lat"], buildings["lon"])


def assemble_table(
    buildings: pd.DataFrame,
    label_cols: pd.DataFrame,
    flags: pd.DataFrame,
    fold_cols: pd.DataFrame,
    post_index: pd.DataFrame,
    pre_index: pd.DataFrame | None,
    origin: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Join the stage outputs into the table; every input is aligned by row position.

    `buildings` carries ids, lat, lon and the seismic columns; `label_cols` is
    `labels.map_labels`, `flags` is `imagery.flags_from_indexes`, `fold_cols` is
    `folds.assign_folds`. The projection origin is `origin` if given
    (the build passes the mean over ALL source rows), else the mean over `buildings`.
    """
    n = len(buildings)
    for name, part in (("label_cols", label_cols), ("flags", flags), ("fold_cols", fold_cols)):
        if len(part) != n:
            raise ValueError(f"{name} has {len(part)} rows, buildings has {n}")
    ids = buildings["building_id"].reset_index(drop=True).astype("int64")
    if not (flags["building_id"].reset_index(drop=True) == ids).all():
        raise ValueError("flags rows are not in the same order as buildings")
    lat0, lon0 = _origin(buildings, origin)
    x, y = geometry.project(buildings["lat"], buildings["lon"], lat0, lon0)
    lab = label_cols.reset_index(drop=True)
    flg = flags.reset_index(drop=True)
    fld = fold_cols.reset_index(drop=True)
    out = pd.DataFrame({"building_id": ids})
    out["lat"] = buildings["lat"].to_numpy(dtype="float64")
    out["lon"] = buildings["lon"].to_numpy(dtype="float64")
    out["x_m"], out["y_m"] = x, y
    for c in CONFIG.seismic_columns:
        out[c] = buildings[c].to_numpy(dtype="float64")
    for c in ("label_raw", "label_source", "label_conflict", "y_collapse", "y_collapse_sens"):
        out[c] = lab[c].to_numpy()
    out["observable"] = lab["observable"].to_numpy(dtype="float64")
    for c in ("has_post", "has_pre", "G", "post_chip", "pre_chip"):
        out[c] = flg[c].to_numpy()
    out["post_chip_sha256"] = flg["post_chip_sha256"].to_numpy()
    out["pre_chip_sha256"] = flg["pre_chip_sha256"].to_numpy()
    out["post_status"] = _status_lookup(ids, post_index, "")
    out["pre_status"] = _status_lookup(ids, pre_index, PRE_UNAVAILABLE)
    for c in ("block_ix", "block_iy", "block_id", "fold"):
        out[c] = fld[c].to_numpy()
    out = out[COLUMNS].astype(
        {f.name: f.type.to_pandas_dtype() for f in SCHEMA if not pa.types.is_string(f.type)}
    )
    return out.sort_values("building_id", kind="mergesort").reset_index(drop=True)


def _in_set(series: pd.Series, allowed: set) -> pd.Series:
    return series.isin(allowed) | series.isna()


def _side_problems(df: pd.DataFrame, side: str) -> list[str]:
    has, chip, sha = df[f"has_{side}"], df[f"{side}_chip"], df[f"{side}_chip_sha256"]
    checks = [
        (~has.isin([0, 1]), f"has_{side} is not 0 or 1"),
        ((has == 1) != (chip != ""), f"has_{side} disagrees with {side}_chip being non-empty"),
        ((chip != "") != (sha != ""), f"{side}_chip and {side}_chip_sha256 are not empty together"),
        (
            (has == 1) != (df[f"{side}_status"] == "ok"),
            f"has_{side} disagrees with {side}_status == 'ok'",
        ),
        (
            chip.str.startswith("/") | chip.str.contains("\\", regex=False),
            f"{side}_chip holds an absolute or non-posix path",
        ),
    ]
    return [msg for mask, msg in checks if mask.any()]


def _g_problems(df: pd.DataFrame, pre_available: bool) -> list[str]:
    expected_g = [image_pattern(a, b) for a, b in zip(df["has_post"], df["has_pre"], strict=True)]
    checks = [
        (
            (~df["G"].isin(PATTERNS)) | (df["G"].to_numpy() != np.array(expected_g)),
            "G is inconsistent with has_post and has_pre",
        ),
        (
            pd.Series(not pre_available, index=df.index) & (df["has_pre"] != 0),
            "has_pre is 1 although pre_available is false",
        ),
    ]
    return [msg for mask, msg in checks if mask.any()]


def _flag_problems(df: pd.DataFrame, pre_available: bool) -> list[str]:
    return _side_problems(df, "post") + _side_problems(df, "pre") + _g_problems(df, pre_available)


def _label_problems(df: pd.DataFrame) -> list[str]:
    problems = []
    if not df["label_raw"].isin(LABEL_CODES).all():
        problems.append(f"label_raw outside {sorted(LABEL_CODES)}")
    sources = {CONFIG.label_column, CONFIG.label_fallback_column}
    if not df["label_source"].isin(sources).all():
        problems.append(f"label_source outside {sorted(sources)}")
    if not df["label_conflict"].isin([0, 1]).all():
        problems.append("label_conflict is not 0 or 1")
    for col in ("y_collapse", "y_collapse_sens", "observable"):
        if not _in_set(df[col], {0.0, 1.0}).all():
            problems.append(f"{col} is not 0, 1 or NaN")
    if (df["observable"].fillna(0).eq(1) != df["y_collapse"].notna()).any():
        problems.append("observable disagrees with y_collapse being labelled")
    return problems


def validate_table(df: pd.DataFrame, pre_available: bool) -> None:
    """Assert every table invariant; raise one `TableError` listing all failures."""
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append(f"columns differ from the schema: {list(df.columns)}")
        raise TableError("; ".join(problems))
    if not df["building_id"].is_unique:
        problems.append("building_id is not unique")
    geo = ["lat", "lon", "x_m", "y_m"]
    if not np.isfinite(df[geo].to_numpy(dtype="float64")).all():
        problems.append("NaN or infinite value in lat, lon, x_m or y_m")
    if not df["fold"].between(0, CONFIG.folds.n_folds - 1).all():
        problems.append(f"fold outside 0..{CONFIG.folds.n_folds - 1}")
    ids = df["block_ix"].astype(str) + "_" + df["block_iy"].astype(str)
    if (df["block_id"] != ids).any():
        problems.append("block_id is not '<block_ix>_<block_iy>'")
    problems += _label_problems(df)
    problems += _flag_problems(df, pre_available)
    if problems:
        raise TableError("; ".join(problems))


def write_table(df: pd.DataFrame, pre_available: bool) -> Path:
    """Validate, then write `derived/buildings.parquet` byte-stably."""
    validate_table(df, pre_available)
    out = paths.buildings_table_path()
    write_parquet_deterministic(df, out, schema=SCHEMA, sort_by=["building_id"])
    log.info("wrote %s (%d rows)", out, len(df))
    return out


def read_table() -> pd.DataFrame:
    return pd.read_parquet(paths.buildings_table_path())
