import math

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_buildings
from notodata import clients, folds, splits
from notodata.config import CONFIG
from notodata.geometry import project, reference_point

N = 40
SEED = 1002


def _frame(n=3000):
    df = synthetic_buildings(n)
    lat0, lon0 = reference_point(df["lat"], df["lon"])
    df["x_m"], df["y_m"] = project(df["lat"], df["lon"], lat0, lon0)
    df["y_collapse"] = df["damage_val"].map({0.0: 0.0, 1.0: 1.0, 9.0: np.nan, 99.0: np.nan})
    df = pd.concat([df, folds.assign_folds(df["x_m"], df["y_m"], df["y_collapse"], 2.0)], axis=1)
    df["has_post"] = (df["building_id"] % 3 != 0).astype("int8")
    df["G"] = np.where(df["building_id"] % 2 == 0, "IP", np.where(df["has_post"] == 1, "I", "none"))
    return df


@pytest.fixture(scope="module")
def df():
    return _frame()


@pytest.fixture(scope="module")
def split(df):
    return splits.assign_roles(df["building_id"], df["fold"], df[["x_m", "y_m"]], SEED)


@pytest.fixture(scope="module")
def built(df, split):
    return clients.build_clients(df, split, N, SEED)


def test_medoid_is_member(built):
    assign, meta = built
    members = assign.groupby("client_id")["building_id"].apply(set)
    for row in meta.itertuples():
        assert row.medoid_building_id in members[row.client_id]


def test_medoid_is_deterministic_minimiser():
    # Small cluster (no sampling): the medoid is the exact summed-distance minimiser.
    ids = np.array([5, 3, 9, 1])
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    assert clients.medoid(ids, xy, np.random.default_rng(0)) == 3
    # tie: two points equidistant -> lowest id
    assert (
        clients.medoid(
            np.array([8, 2]), np.array([[0.0, 0.0], [1.0, 0.0]]), np.random.default_rng(0)
        )
        == 2
    )


def _sample(ids, seed_or_rng):
    """The `medoid_sample_max` ids a fresh/given rng draws from the sorted ids."""
    rng = np.random.default_rng(seed_or_rng) if isinstance(seed_or_rng, int) else seed_or_rng
    ids = np.sort(ids)
    return set(ids[rng.choice(len(ids), CONFIG.clients.medoid_sample_max, replace=False)])


def _big_cluster(seed, n=CONFIG.clients.medoid_sample_max + 100):
    g = np.random.default_rng(seed)
    ids = g.permutation(np.arange(n) * 3 + 7)
    return ids, g.normal(size=(n, 2))


def _brute_medoid(ids, xy, sample):
    keep = np.array([i in sample for i in ids])
    sid, sxy = ids[keep], xy[keep]
    d = np.linalg.norm(sxy[:, None] - sxy[None], axis=2).sum(axis=1)
    return int(sid[np.lexsort((sid, d))[0]])


def test_medoid_sampling_caps_cluster():
    ids, xy = _big_cluster(1)
    m = clients.medoid(ids, xy, np.random.default_rng(0))
    assert m == clients.medoid(ids, xy, np.random.default_rng(0))
    sample = _sample(ids, 0)
    assert len(sample) == CONFIG.clients.medoid_sample_max
    assert m in sample
    assert m == _brute_medoid(ids, xy, sample)
    perm = np.random.default_rng(5).permutation(len(ids))
    assert clients.medoid(ids[perm], xy[perm], np.random.default_rng(0)) == m


def test_medoids_one_rng_walks_labels_ascending():
    n = CONFIG.clients.medoid_sample_max + 100
    ids0, xy0 = _big_cluster(1, n)
    ids1, xy1 = _big_cluster(2, n)
    pool = pd.DataFrame(
        {
            "building_id": np.concatenate([ids0, ids1 + 1_000_000]),
            "x_m": np.concatenate([xy0[:, 0], xy1[:, 0]]),
            "y_m": np.concatenate([xy0[:, 1], xy1[:, 1]]),
        }
    )
    labels = np.repeat([0, 1], n)
    got = clients._medoids(pool, labels, 2, 7)
    rng = np.random.default_rng(7)
    s0 = _sample(ids0, rng)
    s1 = _sample(ids1 + 1_000_000, rng)
    assert got[0] == _brute_medoid(ids0, xy0, s0)
    assert got[1] == _brute_medoid(ids1 + 1_000_000, xy1, s1)
    # cluster 1's sample depends on cluster 0's draw: it is not a fresh-rng sample
    assert s1 != _sample(ids1 + 1_000_000, 7)


def test_clients_hold_no_heldout(built, split):
    assign, _ = built
    roles = split.set_index("building_id")["role"]
    assert set(assign["building_id"]) <= set(roles.index[roles == "train_pool"])
    assert set(assign["building_id"]) == set(roles.index[roles == "train_pool"])


def test_calib_train_disjoint(built):
    assign, _ = built
    calib = assign[assign.role == "calib"]
    train = assign[assign.role == "train"]
    assert set(calib["building_id"]).isdisjoint(train["building_id"])
    assert assign["building_id"].is_unique
    for _, g in calib.groupby("client_id"):
        assert g["building_id"].is_unique


def test_min_client_size(built, df, split):
    assign, meta = built
    kept = assign[assign.client_id >= 0]
    sizes = kept.groupby("client_id").size()
    assert (sizes >= CONFIG.clients.min_client_size).all()
    assert (meta.set_index("client_id")["n_buildings"] == sizes).all()
    dropped = assign[assign.role == "client_dropped"]
    assert len(dropped) > 0
    assert len(sizes) > 0
    assert (dropped["client_id"] == -1).all()


def test_dropped_recorded(built, df, split):
    assign, meta = built
    s = clients.summary_row(assign, df, N, SEED)
    dropped = assign[assign.role == "client_dropped"]
    kept = assign[assign.client_id >= 0]
    assert s["clients_dropped"] > 0
    assert s["clients_kept"] > 0
    assert s["buildings_dropped"] > 0
    assert s["clients_kept"] + s["clients_dropped"] == N
    assert len(meta) == s["clients_kept"]
    assert s["buildings_dropped"] == len(dropped)
    assert (dropped["client_id"] == -1).all()
    assert set(kept["role"]) <= {"train", "calib", "unlabelled"}
    assert -1 not in set(meta["client_id"])
    pool = (split["role"] == "train_pool").sum()
    assert s["buildings_dropped"] + len(kept) == pool
    assert s["dropped_collapsed"] + s["dropped_survived"] + s["dropped_unlabelled"] == len(dropped)
    assert s["size_min"] == meta["n_buildings"].min()
    assert s["size_max"] == meta["n_buildings"].max()
    assert s["size_median"] == meta["n_buildings"].median()


def test_calib_share(built, df):
    assign, _ = built
    y = df.set_index("building_id")["y_collapse"]
    lab = assign[assign.role.isin(["train", "calib"])].copy()
    lab["y"] = y.reindex(lab["building_id"]).to_numpy()
    for (_, _), g in lab.groupby(["client_id", "y"]):
        n = len(g)
        k = math.floor(CONFIG.clients.calibration_share * n + 0.5)
        assert (g.role == "calib").sum() == k
    unl = assign[assign.role == "unlabelled"]
    assert y.reindex(unl["building_id"]).isna().all()
    assert not y.reindex(lab["building_id"]).isna().any()


def test_clients_deterministic(df, split, tmp_path, monkeypatch):
    monkeypatch.setenv("NOTODATA_HOME", str(tmp_path / "a"))
    a, ma = clients.build_clients(df, split, N, SEED)
    pa_, pm = clients.write_clients(a, ma, N, SEED)
    monkeypatch.setenv("NOTODATA_HOME", str(tmp_path / "b"))
    b, mb = clients.build_clients(
        df.sample(frac=1.0, random_state=1), split.sample(frac=1.0, random_state=2), N, SEED
    )
    qa, qm = clients.write_clients(b, mb, N, SEED)
    assert pa_.read_bytes() == qa.read_bytes()
    assert pm.read_bytes() == qm.read_bytes()


def test_renumbered_by_medoid_id(built):
    _, meta = built
    assert list(meta["client_id"]) == list(range(len(meta)))
    assert meta["medoid_building_id"].is_monotonic_increasing
    assert meta["medoid_building_id"].is_unique


def test_meta_coordinates_match(built, df):
    _, meta = built
    d = df.set_index("building_id").loc[meta["medoid_building_id"]]
    assert np.array_equal(meta["x_m"].to_numpy(), d["x_m"].to_numpy())
    assert np.array_equal(meta["lat"].to_numpy(), d["lat"].to_numpy())


def test_expert_counts(built, df):
    assign, meta = built
    d = df.set_index("building_id")
    a = assign[assign.role.isin(["train", "calib"])].copy()
    a["y"] = d.loc[a["building_id"], "y_collapse"].to_numpy()
    a["post"] = d.loc[a["building_id"], "has_post"].to_numpy()
    a["G"] = d.loc[a["building_id"], "G"].to_numpy()
    masks = {"S": a["y"].notna(), "I": a["post"] == 1, "D": a["G"] == "IP"}
    for expert, m in masks.items():
        for role in ("train", "calib"):
            sel = a[m & (a.role == role)]
            got_c = meta.set_index("client_id")[f"{role}_collapsed_{expert}"]
            got_s = meta.set_index("client_id")[f"{role}_survived_{expert}"]
            exp_c = sel[sel.y == 1].groupby("client_id").size().reindex(got_c.index, fill_value=0)
            exp_s = sel[sel.y == 0].groupby("client_id").size().reindex(got_s.index, fill_value=0)
            assert (got_c == exp_c).all()
            assert (got_s == exp_s).all()


def test_summary_row(built, df):
    assign, meta = built
    s = clients.summary_row(assign, df, N, SEED)
    assert s["clients_kept"] == len(meta)
    assert s["N"] == N
    assert s["seed"] == SEED
    zero = int((meta["calib_collapsed_S"] == 0).sum())
    assert s["clients_zero_collapsed_calib"] == zero
