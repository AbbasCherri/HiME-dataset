"""Spatial blocks and the greedy fold assignment (brief D5.5).

Every building gets a block from the projected metres and the block size L;
whole blocks are then dealt to folds, so two buildings in one block always
share a fold. Blocks are ordered by integer tuple, never by the display string.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from notodata.config import CONFIG


def block_indices(x_m, y_m, l_km: float) -> tuple[np.ndarray, np.ndarray]:
    """`floor(x / L)` and `floor(y / L)` as int32 (negative indices allowed)."""
    size_m = l_km * 1000.0
    ix = np.floor(np.asarray(x_m, dtype=np.float64) / size_m).astype(np.int32)
    iy = np.floor(np.asarray(y_m, dtype=np.float64) / size_m).astype(np.int32)
    return ix, iy


def block_counts(ix, iy, y_collapse) -> pd.DataFrame:
    """Per block: collapsed (y_collapse == 1) and total buildings, NaN labels included in total."""
    frame = pd.DataFrame(
        {
            "block_ix": np.asarray(ix),
            "block_iy": np.asarray(iy),
            "collapsed": (np.asarray(y_collapse, dtype=np.float64) == 1.0).astype(np.int64),
        }
    )
    grouped = frame.groupby(["block_ix", "block_iy"], sort=True)["collapsed"]
    out = grouped.agg(collapsed="sum", total="size").reset_index()
    return out.astype({"collapsed": np.int64, "total": np.int64})


def _pick_fold(collapsed: list[int], buildings: list[int]) -> int:
    """Fewest collapsed so far, then fewest buildings, then lowest fold index."""
    return min(range(len(collapsed)), key=lambda f: (collapsed[f], buildings[f], f))


def greedy_assign(blocks: pd.DataFrame, n_folds: int | None = None) -> pd.DataFrame:
    """Add a `fold` column; rows come back in assignment order.

    Order: collapsed descending, total descending, then `(block_ix, block_iy)`
    ascending as integers. Running totals update after each block.
    """
    n_folds = CONFIG.folds.n_folds if n_folds is None else n_folds
    ordered = blocks.sort_values(
        ["collapsed", "total", "block_ix", "block_iy"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    collapsed, buildings = [0] * n_folds, [0] * n_folds
    assigned = []
    for row in ordered.itertuples(index=False):
        fold = _pick_fold(collapsed, buildings)
        collapsed[fold] += int(row.collapsed)
        buildings[fold] += int(row.total)
        assigned.append(fold)
    ordered["fold"] = np.asarray(assigned, dtype=np.int8)
    return ordered


def assign_folds(x_m, y_m, y_collapse, l_km: float) -> pd.DataFrame:
    """Per building (input order): block_ix, block_iy, block_id, fold."""
    ix, iy = block_indices(x_m, y_m, l_km)
    blocks = greedy_assign(block_counts(ix, iy, y_collapse))
    keys = pd.MultiIndex.from_arrays([ix, iy])
    lookup = blocks.set_index(["block_ix", "block_iy"])["fold"]
    fold = lookup.reindex(keys).to_numpy().astype(np.int8)
    return pd.DataFrame(
        {
            "block_ix": ix,
            "block_iy": iy,
            "block_id": pd.Series(ix).astype(str).str.cat(pd.Series(iy).astype(str), sep="_"),
            "fold": fold,
        }
    )


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """The `reports/folds.csv` table: one row per fold, plus a count per image pattern G.

    Rows with a NaN `G` are counted under `G_missing`, so the G columns always
    sum to `buildings`.
    """
    y = df["y_collapse"]
    parts = pd.DataFrame(
        {
            "fold": df["fold"].to_numpy(),
            "block_id": df["block_id"].to_numpy(),
            "collapsed": (y == 1).to_numpy(),
            "survived": (y == 0).to_numpy(),
            "unlabelled": y.isna().to_numpy(),
        }
    )
    grouped = parts.groupby("fold")
    out = grouped.agg(
        blocks=("block_id", "nunique"),
        buildings=("block_id", "size"),
        collapsed=("collapsed", "sum"),
        survived=("survived", "sum"),
        unlabelled=("unlabelled", "sum"),
    )
    labelled = out["collapsed"] + out["survived"]
    out["prevalence"] = (out["collapsed"] / labelled).where(labelled > 0)
    if "G" in df:
        by_g = pd.crosstab(df["fold"].to_numpy(), df["G"].fillna("missing").to_numpy()).add_prefix(
            "G_"
        )
        out = out.join(by_g)
    return out.reset_index()
