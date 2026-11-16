"""Per-seed KMeans clients with medoids, a size filter and calibration hold-outs (brief D7).

Clients are built from the training pool of one seed only, so a client never
holds a validation or test building. Client ids are renumbered by ascending
medoid `building_id`, which makes them independent of KMeans label order.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from notodata import paths
from notodata.config import CONFIG
from notodata.io import write_parquet_deterministic

ROLES = ("train", "calib", "unlabelled", "client_dropped")
DROPPED_ID = -1
"""`client_id` of a building whose cluster fell under the minimum client size."""

_EXPERTS = ("S", "I", "D")
"""E_S: all labelled rows; E_I: `has_post = 1`; E_Delta (`D`): `G = IP`."""

_ASSIGN_SCHEMA = pa.schema(
    [
        ("building_id", pa.int64()),
        ("client_id", pa.int64()),
        ("role", pa.string()),
    ]
)
_COUNT_COLUMNS = [
    f"{part}_{cls}_{e}"
    for e in _EXPERTS
    for part in ("train", "calib")
    for cls in ("collapsed", "survived")
]
_META_SCHEMA = pa.schema(
    [
        ("client_id", pa.int64()),
        ("medoid_building_id", pa.int64()),
        ("x_m", pa.float64()),
        ("y_m", pa.float64()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("n_buildings", pa.int64()),
        *[(c, pa.int64()) for c in _COUNT_COLUMNS],
    ]
)


def medoid(ids: np.ndarray, xy: np.ndarray, rng: np.random.Generator) -> int:
    """Medoid `building_id` of one cluster.

    A cluster over `medoid_sample_max` buildings is first reduced to a sample
    drawn without replacement from its sorted ids. The medoid is the sampled
    building with the smallest summed distance to the sample; ties go to the
    lowest id.
    """
    order = np.argsort(ids, kind="mergesort")
    ids, xy = np.asarray(ids)[order], np.asarray(xy, dtype=np.float64)[order]
    cap = CONFIG.clients.medoid_sample_max
    if len(ids) > cap:
        keep = np.sort(rng.choice(len(ids), size=cap, replace=False))
        ids, xy = ids[keep], xy[keep]
    totals = cdist(xy, xy).sum(axis=1)
    return int(ids[np.argmin(totals)])


def _cluster(pool: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    km = KMeans(
        n_clusters=n,
        n_init=CONFIG.clients.kmeans_n_init,
        random_state=CONFIG.clients.kmeans_random_state_base + seed,
    )
    return km.fit_predict(pool[["x_m", "y_m"]].to_numpy(dtype=np.float64))


def _medoids(pool: pd.DataFrame, labels: np.ndarray, n: int, seed: int) -> dict[int, int]:
    """KMeans label -> medoid id, walking labels in ascending order with one rng."""
    rng = np.random.default_rng(seed)
    ids = pool["building_id"].to_numpy()
    xy = pool[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    out = {}
    for label in range(n):
        m = labels == label
        if m.any():
            out[label] = medoid(ids[m], xy[m], rng)
    return out


def _calibration_roles(ids: np.ndarray, y: np.ndarray, seed: int, client_id: int) -> np.ndarray:
    """`train`, `calib` or `unlabelled` per building of one client."""
    role = np.full(len(ids), "unlabelled", dtype=object)
    rng = np.random.default_rng([seed, client_id])
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(y == cls)
        k = math.floor(CONFIG.clients.calibration_share * len(idx) + 0.5)
        role[idx] = "train"
        role[rng.permutation(idx)[:k]] = "calib"
    return role


def _expert_counts(rows: pd.DataFrame) -> dict[str, int]:
    """Collapsed and survived counts per expert for the train and calib rows of one client."""
    masks = {
        "S": rows["y"].notna(),
        "I": rows["has_post"] == 1,
        "D": rows["G"] == "IP",
    }
    counts = {}
    for e, mask in masks.items():
        for part in ("train", "calib"):
            sel = rows[mask & (rows["role"] == part)]["y"]
            counts[f"{part}_collapsed_{e}"] = int((sel == 1).sum())
            counts[f"{part}_survived_{e}"] = int((sel == 0).sum())
    return counts


def _meta_rows(assign: pd.DataFrame, df: pd.DataFrame, medoids: dict[int, int]) -> pd.DataFrame:
    feats = df.set_index("building_id")
    rows = assign[assign["client_id"] != DROPPED_ID].copy()
    rows["y"] = feats.loc[rows["building_id"], "y_collapse"].to_numpy()
    rows["has_post"] = feats.loc[rows["building_id"], "has_post"].to_numpy()
    rows["G"] = feats.loc[rows["building_id"], "G"].to_numpy()
    out = []
    for client_id, g in rows.groupby("client_id", sort=True):
        mid = medoids[client_id]
        m = feats.loc[mid]
        out.append(
            {
                "client_id": int(client_id),
                "medoid_building_id": int(mid),
                "x_m": float(m["x_m"]),
                "y_m": float(m["y_m"]),
                "lat": float(m["lat"]),
                "lon": float(m["lon"]),
                "n_buildings": len(g),
                **_expert_counts(g),
            }
        )
    return pd.DataFrame(out, columns=[f.name for f in _META_SCHEMA]).astype(
        {f.name: f.type.to_pandas_dtype() for f in _META_SCHEMA}
    )


def build_clients(
    df: pd.DataFrame, split: pd.DataFrame, n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clients for one (N, seed): the assignment table and the per-client meta table.

    `df` needs `building_id`, `x_m`, `y_m`, `lat`, `lon`, `y_collapse`,
    `has_post` and `G`; `split` is the seed's role table.
    """
    pool_ids = split.loc[split["role"] == "train_pool", "building_id"]
    pool = df[df["building_id"].isin(pool_ids)].sort_values("building_id", kind="mergesort")
    pool = pool.reset_index(drop=True)
    labels = _cluster(pool, n, seed)
    medoids = _medoids(pool, labels, n, seed)
    sizes = np.bincount(labels, minlength=n)
    kept = sorted(
        (m, lab) for lab, m in medoids.items() if sizes[lab] >= CONFIG.clients.min_client_size
    )
    new_id = {lab: i for i, (_, lab) in enumerate(kept)}
    client = np.array([new_id.get(lab, DROPPED_ID) for lab in labels], dtype=np.int64)
    role = np.full(len(pool), "client_dropped", dtype=object)
    y = pool["y_collapse"].to_numpy(dtype=np.float64)
    for cid in range(len(kept)):
        idx = np.flatnonzero(client == cid)
        role[idx] = _calibration_roles(pool["building_id"].to_numpy()[idx], y[idx], seed, cid)
    assign = pd.DataFrame(
        {"building_id": pool["building_id"].to_numpy(), "client_id": client, "role": role}
    )
    meta = _meta_rows(assign, df, {new_id[lab]: m for m, lab in kept})
    return assign, meta


def write_clients(assign: pd.DataFrame, meta: pd.DataFrame, n: int, seed: int) -> tuple[Path, Path]:
    ap, mp = paths.client_path(n, seed), paths.client_meta_path(n, seed)
    write_parquet_deterministic(assign, ap, schema=_ASSIGN_SCHEMA, sort_by=["building_id"])
    write_parquet_deterministic(meta, mp, schema=_META_SCHEMA, sort_by=["client_id"])
    return ap, mp


def summary_row(assign: pd.DataFrame, df: pd.DataFrame, n: int, seed: int) -> dict:
    """One `reports/clients_summary.csv` row for a built (N, seed)."""
    y = df.set_index("building_id")["y_collapse"].reindex(assign["building_id"]).to_numpy()
    kept = assign["client_id"] != DROPPED_ID
    dropped = ~kept
    sizes = assign[kept].groupby("client_id").size()
    calib = (assign["role"] == "calib").to_numpy()
    calib_collapsed = pd.Series(calib & (y == 1)).groupby(assign["client_id"].to_numpy()).sum()
    calib_collapsed = calib_collapsed[calib_collapsed.index != DROPPED_ID]
    return {
        "N": n,
        "seed": seed,
        "clients_kept": len(sizes),
        "clients_dropped": n - len(sizes),
        "buildings_dropped": int(dropped.sum()),
        "dropped_collapsed": int((dropped.to_numpy() & (y == 1)).sum()),
        "dropped_survived": int((dropped.to_numpy() & (y == 0)).sum()),
        "dropped_unlabelled": int((dropped.to_numpy() & np.isnan(y)).sum()),
        "size_median": float(sizes.median()) if len(sizes) else float("nan"),
        "size_min": int(sizes.min()) if len(sizes) else 0,
        "size_max": int(sizes.max()) if len(sizes) else 0,
        "clients_zero_collapsed_calib": int((calib_collapsed == 0).sum()),
    }
