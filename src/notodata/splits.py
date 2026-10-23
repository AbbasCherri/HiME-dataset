"""Per-seed roles with the 500 m buffer, the raw feature table and the scaler (brief D5.6).

Roles come from the fold column alone: test fold `s mod 5`, validation fold
`(s + 1) mod 5`, the other three folds are the training pool, and a pool
building within the buffer of any held-out building is dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.preprocessing import StandardScaler

from notodata import paths
from notodata.config import CONFIG
from notodata.geometry import min_distance
from notodata.io import write_parquet_deterministic

ROLES = ("test", "val", "train_pool", "buffer_dropped")
_HELD_OUT = ("test", "val")

_SCHEMA = pa.schema(
    [
        ("building_id", pa.int64()),
        ("role", pa.string()),
        ("nearest_heldout_m", pa.float64()),
    ]
)


def heldout_folds(seed: int) -> tuple[int, int]:
    """(test fold, validation fold) for a seed."""
    n = CONFIG.folds.n_folds
    return seed % n, (seed + 1) % n


def assign_roles(building_id, fold, xy, seed: int) -> pd.DataFrame:
    """Role per building for one seed.

    `nearest_heldout_m` is set for every training-pool candidate (kept or
    dropped) and is NaN for validation and test buildings.
    """
    ids = np.asarray(building_id)
    fold = np.asarray(fold)
    xy = np.asarray(xy, dtype=np.float64)
    test_f, val_f = heldout_folds(seed)
    role = np.full(len(ids), "train_pool", dtype=object)
    role[fold == test_f] = "test"
    role[fold == val_f] = "val"
    held = np.isin(role, _HELD_OUT)
    nearest = np.full(len(ids), np.nan)
    pool = ~held
    if held.any() and pool.any():
        nearest[pool] = min_distance(xy[pool], xy[held])
        role[pool & (nearest < CONFIG.folds.buffer_m)] = "buffer_dropped"
    return pd.DataFrame({"building_id": ids, "role": role, "nearest_heldout_m": nearest})


def write_split(split: pd.DataFrame, seed: int) -> Path:
    path = paths.split_path(seed)
    write_parquet_deterministic(split, path, schema=_SCHEMA, sort_by=["building_id"])
    return path


def summary(splits_by_seed: dict[int, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Counts per seed and role, split by class (the `reports/splits_summary.csv` table)."""
    classes = df.set_index("building_id")["y_collapse"]
    rows = []
    for seed in sorted(splits_by_seed):
        sp = splits_by_seed[seed]
        y = classes.reindex(sp["building_id"]).to_numpy()
        frame = pd.DataFrame(
            {
                "role": sp["role"].to_numpy(),
                "collapsed": y == 1,
                "survived": y == 0,
                "unlabelled": np.isnan(y),
            }
        )
        agg = frame.groupby("role").agg(
            buildings=("role", "size"),
            collapsed=("collapsed", "sum"),
            survived=("survived", "sum"),
            unlabelled=("unlabelled", "sum"),
        )
        agg = agg.reindex(list(ROLES), fill_value=0).rename_axis("role").reset_index()
        agg.insert(0, "seed", seed)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def feature_table(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """`building_id` plus exactly the raw seismic columns; anything else raises."""
    allowed = list(CONFIG.seismic_columns)
    wanted = allowed if columns is None else list(columns)
    extra = [c for c in wanted if c not in allowed]
    if extra:
        raise ValueError(f"not a model feature: {extra}; allowed are {allowed}")
    return df[["building_id", *wanted]].copy()


def fit_scaler(df: pd.DataFrame, split: pd.DataFrame, rows: str = "train", building_ids=None):
    """StandardScaler fitted on training rows only.

    Default rows are the split's post-buffer training pool. `building_ids`
    (for example a client's buildings) overrides that, but every id must be
    a `train_pool` building of `split`; a val, test or dropped id raises.
    """
    if rows != "train":
        raise ValueError("scaler rows must be 'train'")
    role = split.set_index("building_id")["role"]
    if building_ids is None:
        ids = role.index[role == "train_pool"]
    else:
        ids = pd.Index(np.asarray(building_ids))
        bad = ids[role.reindex(ids).to_numpy() != "train_pool"]
        if len(bad):
            raise ValueError(f"{len(bad)} rows are not training-pool buildings, e.g. {bad[0]}")
    feats = feature_table(df).set_index("building_id").loc[ids]
    return StandardScaler().fit(feats.to_numpy(dtype=np.float64))
