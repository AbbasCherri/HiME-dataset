import numpy as np
import pandas as pd

from conftest import synthetic_buildings
from notodata import folds
from notodata.geometry import project, reference_point


def _blocks(rows):
    return pd.DataFrame(rows, columns=["block_ix", "block_iy", "collapsed", "total"])


def test_fold_greedy():
    # Three folds. Sorted order is collapsed desc, total desc, then the integer
    # tuple (2, 0) before (10, 0) (the strings "10_0" < "2_0" would flip it).
    blocks = _blocks(
        [
            (4, 4, 0, 3),  # G
            (10, 0, 5, 5),  # D
            (0, 0, 10, 20),  # A
            (3, 3, 1, 9),  # F
            (2, 0, 5, 5),  # E
            (0, 1, 8, 10),  # C
            (1, 0, 8, 30),  # B
        ]
    )
    out = folds.greedy_assign(blocks, n_folds=3)
    got = {(r.block_ix, r.block_iy): r.fold for r in out.itertuples()}
    assert got == {
        (0, 0): 0,  # empty folds: lowest index
        (1, 0): 1,  # B
        (0, 1): 2,  # C
        (2, 0): 2,  # E: f1 and f2 tie on 8 collapsed; f2 has fewer buildings
        (10, 0): 1,  # D: f1 now holds the fewest collapsed
        (3, 3): 0,  # F: f0 holds the fewest collapsed
        (4, 4): 0,  # G
    }
    assert list(out["block_ix"])[:3] == [0, 1, 0]  # returned in assignment order


def test_fold_greedy_lowest_index_on_full_tie():
    blocks = _blocks([(0, 0, 2, 4), (5, 5, 2, 4), (9, 9, 2, 4)])
    out = folds.greedy_assign(blocks, n_folds=2)
    got = {(r.block_ix, r.block_iy): r.fold for r in out.itertuples()}
    assert got == {(0, 0): 0, (5, 5): 1, (9, 9): 0}


def test_block_indices_floor_and_negative():
    ix, iy = folds.block_indices(
        np.array([0.0, 1999.9, 2000.0, -0.1]), np.array([5000.0, 0.0, -2000.0, 3999.0]), 2.0
    )
    assert list(ix) == [0, 0, 1, -1]
    assert list(iy) == [2, 0, -1, 1]
    assert ix.dtype == np.int32


def _frame(n=3000):
    df = synthetic_buildings(n)
    lat0, lon0 = reference_point(df["lat"], df["lon"])
    df["x_m"], df["y_m"] = project(df["lat"], df["lon"], lat0, lon0)
    y = df["damage_val"].map({0.0: 0.0, 1.0: 1.0, 9.0: np.nan, 99.0: np.nan})
    df["y_collapse"] = y
    return df


def test_assign_folds_blocks_are_atomic_and_deterministic():
    df = _frame()
    a = folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], 2.0)
    b = folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], 2.0)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["fold"].unique()) <= set(range(5))
    assert a["fold"].dtype == np.int8
    assert (a.groupby("block_id")["fold"].agg(lambda f: f.is_unique or f.eq(f.iloc[0]).all())).all()
    assert (a["block_id"] == a["block_ix"].astype(str) + "_" + a["block_iy"].astype(str)).all()


def test_fold_table_counts():
    df = _frame()
    df = pd.concat([df, folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], 2.0)], axis=1)
    df["G"] = np.where(df["building_id"] % 2 == 0, "IP", "none")
    t = folds.fold_table(df)
    assert list(t["fold"]) == list(range(5))
    assert t["buildings"].sum() == len(df)
    assert t["collapsed"].sum() == (df["y_collapse"] == 1).sum()
    assert t["survived"].sum() == (df["y_collapse"] == 0).sum()
    assert t["unlabelled"].sum() == df["y_collapse"].isna().sum()
    assert (t["collapsed"] + t["survived"] + t["unlabelled"] == t["buildings"]).all()
    assert (t["G_IP"] + t["G_none"] == t["buildings"]).all()
