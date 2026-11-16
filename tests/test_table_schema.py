"""D8 building table: schema, invariants, relative paths, byte stability."""

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_buildings
from notodata import folds, geometry, imagery, labels, paths, table
from notodata.config import CONFIG

POST, PRE = "post_stub", "pre_stub"


def _index(ids, ok_every, layer):
    rows = []
    for b in ids:
        if b % ok_every == 0:
            rows.append((int(b), "ok", f"chips/{layer}/{b // 1000}/{b}.png", f"{b:064x}"))
        else:
            rows.append((int(b), "missing", "", ""))
    return pd.DataFrame(rows, columns=["building_id", "status", "path", "sha256"])


def make_table(n=300, pre_available=True, seed=0):
    df = synthetic_buildings(n, seed=seed)
    lab = labels.map_labels(df)
    lat0, lon0 = geometry.reference_point(df["lat"], df["lon"])
    x, y = geometry.project(df["lat"], df["lon"], lat0, lon0)
    fold_df = folds.assign_folds(x, y, lab["y_collapse"], l_km=2.0)
    ids = df["building_id"]
    post, pre = _index(ids, 2, POST), _index(ids, 3, PRE)
    flags = imagery.flags_from_indexes(ids, post, pre, pre_available)
    return table.assemble_table(df, lab, flags, fold_df, post, pre if pre_available else None)


def test_assembled_table_has_schema_and_validates():
    t = make_table()
    assert list(t.columns) == table.COLUMNS
    table.validate_table(t, pre_available=True)
    assert t["building_id"].is_monotonic_increasing
    assert set(t["G"]) <= {"IP", "I", "none"}


def test_no_pre_layer_gives_no_pre_flags():
    t = make_table(pre_available=False)
    table.validate_table(t, pre_available=False)
    assert (t["has_pre"] == 0).all()
    assert (t["pre_chip"] == "").all()


def _corrupt_dup(t):
    t.loc[1, "building_id"] = t.loc[0, "building_id"]


def _corrupt_has_post(t):
    i = t.index[t["has_post"] == 1][0]
    t.loc[i, "post_chip"] = ""


def _corrupt_g(t):
    t.loc[0, "G"] = "IP" if t.loc[0, "G"] != "IP" else "none"


def _corrupt_fold(t):
    t.loc[0, "fold"] = CONFIG.folds.n_folds


def _corrupt_geometry(t):
    t.loc[0, "x_m"] = np.nan


def _corrupt_label(t):
    t.loc[0, "label_raw"] = 7


def _corrupt_absolute(t):
    i = t.index[t["has_post"] == 1][0]
    t.loc[i, "post_chip"] = "/home/me/" + t.loc[i, "post_chip"]


def _corrupt_block_id(t):
    t.loc[0, "block_id"] = "nope"


def _first_post(t):
    return t.index[t["has_post"] == 1][0]


def _corrupt_label_source(t):
    t.loc[0, "label_source"] = "guess"


def _corrupt_label_conflict(t):
    t.loc[0, "label_conflict"] = 2


def _corrupt_y_collapse(t):
    t.loc[t.index[t["y_collapse"].notna()][0], "y_collapse"] = 0.5


def _corrupt_observable(t):
    i = t.index[t["y_collapse"].notna()][0]
    t.loc[i, "observable"] = 0.0


def _corrupt_has_post_range(t):
    t.loc[_first_post(t), "has_post"] = 2


def _corrupt_post_status(t):
    t.loc[_first_post(t), "post_status"] = "missing"


CORRUPTIONS = [
    (_corrupt_dup, "building_id is not unique"),
    (_corrupt_has_post, "has_post disagrees with post_chip"),
    (_corrupt_g, "G is inconsistent"),
    (_corrupt_fold, "fold outside"),
    (_corrupt_geometry, "NaN or infinite"),
    (_corrupt_label, "label_raw outside"),
    (_corrupt_absolute, "post_chip holds an absolute"),
    (_corrupt_block_id, "block_id is not"),
    (_corrupt_label_source, "label_source outside"),
    (_corrupt_label_conflict, "label_conflict is not"),
    (_corrupt_y_collapse, "y_collapse is not 0, 1 or NaN"),
    (_corrupt_observable, "observable disagrees"),
    (_corrupt_has_post_range, "has_post is not 0 or 1"),
    (_corrupt_post_status, "has_post disagrees with post_status"),
]


@pytest.mark.parametrize(("corrupt", "message"), CORRUPTIONS)
def test_table_invariants(corrupt, message):
    t = make_table()
    table.validate_table(t, pre_available=True)
    corrupt(t)
    with pytest.raises(table.TableError, match=message):
        table.validate_table(t, pre_available=True)


def test_pre_flag_with_pre_unavailable_fails():
    t = make_table(pre_available=True)
    assert (t["has_pre"] == 1).any()
    with pytest.raises(table.TableError, match="pre_available"):
        table.validate_table(t, pre_available=False)


def test_relative_paths(tmp_home):
    t = make_table()
    for col in ("post_chip", "pre_chip"):
        paths_ = t[col][t[col] != ""]
        assert len(paths_)
        assert not paths_.str.startswith("/").any()
        assert not paths_.str.contains(str(tmp_home), regex=False).any()
    table.write_table(t, pre_available=True)
    raw = paths.buildings_table_path().read_bytes()
    assert str(tmp_home).encode() not in raw


def test_table_bytes_stable(tmp_home):
    t = make_table()
    out = paths.buildings_table_path()
    table.write_table(t, pre_available=True)
    first = out.read_bytes()
    shuffled = t.sample(frac=1.0, random_state=1).reset_index(drop=True)
    table.write_table(shuffled, pre_available=True)
    assert out.read_bytes() == first
    again = table.read_table()
    pd.testing.assert_frame_equal(again, t)


def test_write_refuses_invalid_table(tmp_home):
    t = make_table()
    _corrupt_fold(t)
    with pytest.raises(table.TableError):
        table.write_table(t, pre_available=True)
    assert not paths.buildings_table_path().exists()
