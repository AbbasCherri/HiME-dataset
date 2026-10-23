import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_buildings
from notodata import folds, splits
from notodata.config import CONFIG, EARTH_RADIUS_M
from notodata.geometry import min_distance, project, reference_point

SEEDS = CONFIG.test_seeds


def _frame(n=3000):
    df = synthetic_buildings(n)
    lat0, lon0 = reference_point(df["lat"], df["lon"])
    df["x_m"], df["y_m"] = project(df["lat"], df["lon"], lat0, lon0)
    df["y_collapse"] = df["damage_val"].map({0.0: 0.0, 1.0: 1.0, 9.0: np.nan, 99.0: np.nan})
    df = pd.concat([df, folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], 2.0)], axis=1)
    return df


@pytest.fixture(scope="module")
def df():
    return _frame()


def _split(df, seed):
    return splits.assign_roles(df["building_id"], df["fold"], df[["x_m", "y_m"]], seed)


def test_fold_for_seed():
    assert splits.heldout_folds(1003) == (3, 4)
    assert splits.heldout_folds(2004) == (4, 0)


def test_roles_disjoint(df):
    for s in SEEDS:
        sp = _split(df, s)
        assert sp["building_id"].is_unique
        assert set(sp["building_id"]) == set(df["building_id"])
        assert set(sp["role"]) <= set(splits.ROLES)
        merged = sp.merge(df[["building_id", "fold"]])
        test_f, val_f = splits.heldout_folds(s)
        assert (merged.loc[merged.role == "test", "fold"] == test_f).all()
        assert (merged.loc[merged.role == "val", "fold"] == val_f).all()
        pool = merged[merged.role.isin(["train_pool", "buffer_dropped"])]
        assert not pool["fold"].isin([test_f, val_f]).any()


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _assert_geodesic_buffer(full):
    """Independent of the projection: the 200 closest kept buildings, brute-force haversine."""
    train = full[full.role == "train_pool"].nsmallest(200, "nearest_heldout_m")
    held = full[full.role.isin(["val", "test"])]
    d = _haversine_m(
        train["lat"].to_numpy()[:, None],
        train["lon"].to_numpy()[:, None],
        held["lat"].to_numpy()[None, :],
        held["lon"].to_numpy()[None, :],
    ).min(axis=1)
    # equirectangular error is under 2 m on 500 m (spec risk table)
    assert (d >= CONFIG.folds.buffer_m - 2.0).all()


def test_buffer_500m(df):
    for s in SEEDS:
        sp = _split(df, s).merge(df[["building_id", "x_m", "y_m", "lat", "lon"]])
        held = sp[sp.role.isin(["val", "test"])][["x_m", "y_m"]].to_numpy()
        train = sp[sp.role == "train_pool"]
        assert len(train) > 0
        d = min_distance(train[["x_m", "y_m"]].to_numpy(), held)
        assert (d >= CONFIG.folds.buffer_m).all()
        assert np.allclose(d, train["nearest_heldout_m"].to_numpy())
        dropped = sp[sp.role == "buffer_dropped"]
        dd = min_distance(dropped[["x_m", "y_m"]].to_numpy(), held)
        assert (dd < CONFIG.folds.buffer_m).all()
        _assert_geodesic_buffer(sp)
        assert sp.loc[sp.role.isin(["val", "test"]), "nearest_heldout_m"].isna().all()


def test_buffer_drops_something(df):
    assert any((_split(df, s)["role"] == "buffer_dropped").any() for s in SEEDS)


def test_each_fold_tested_once():
    assert sorted(splits.heldout_folds(s)[0] for s in SEEDS) == list(range(5))


def test_same_residue_seeds_share_roles(df):
    pd.testing.assert_frame_equal(_split(df, 1000), _split(df, 2000))


def test_no_latlon_features(df):
    ft = splits.feature_table(df)
    assert list(ft.columns) == ["building_id", *CONFIG.seismic_columns]
    for bad in ("lat", "lon", "x_m", "y_m", "block_ix", "fold"):
        with pytest.raises(ValueError, match="not a model feature"):
            splits.feature_table(df, columns=[bad])
    with pytest.raises(ValueError, match="not a model feature"):
        splits.feature_table(df, columns=[*CONFIG.seismic_columns, "lat"])


def test_scaler_train_only(df):
    sp = _split(df, 1000)
    scaler = splits.fit_scaler(df, sp)
    train_ids = sp.loc[sp.role == "train_pool", "building_id"]
    rows = df[df["building_id"].isin(train_ids)][list(CONFIG.seismic_columns)]
    np.testing.assert_allclose(scaler.mean_, rows.mean().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(scaler.scale_, rows.std(ddof=0).to_numpy(), atol=1e-9)
    with pytest.raises(ValueError, match="must be 'train'"):
        splits.fit_scaler(df, sp, rows="val")
    # explicit ids (e.g. a client's buildings) must all be training rows
    splits.fit_scaler(df, sp, building_ids=train_ids.iloc[:50])
    held_id = sp.loc[sp.role == "test", "building_id"].iloc[0]
    with pytest.raises(ValueError, match="not training-pool"):
        splits.fit_scaler(df, sp, building_ids=[*train_ids.iloc[:50], held_id])
    with pytest.raises(ValueError, match="duplicates"):
        splits.fit_scaler(df, sp, building_ids=[train_ids.iloc[0], train_ids.iloc[0]])
    dropped_id = sp.loc[sp.role == "buffer_dropped", "building_id"].iloc[0]
    with pytest.raises(ValueError, match="not training-pool"):
        splits.fit_scaler(df, sp, building_ids=[dropped_id])


def test_write_split_roundtrip(tmp_home, df):
    sp = _split(df, 1001)
    path = splits.write_split(sp, 1001)
    assert path.name == "seed_1001.parquet"
    back = pd.read_parquet(path)
    assert list(back.columns) == ["building_id", "role", "nearest_heldout_m"]
    assert len(back) == len(sp)
    assert splits.write_split(sp, 1001).read_bytes() == path.read_bytes()


def test_summary_counts(df):
    sp = {s: _split(df, s) for s in (1000, 1001)}
    t = splits.summary(sp, df)
    assert set(t["seed"]) == {1000, 1001}
    one = t[t.seed == 1000]
    assert one["buildings"].sum() == len(df)
    assert set(one["role"]) == set(splits.ROLES)
