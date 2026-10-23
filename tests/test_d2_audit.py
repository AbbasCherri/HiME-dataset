import itertools
import subprocess

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from _fsscan import repo_root
from notodata import audit, paths
from notodata.config import CONFIG

BASE = CONFIG.baseline_imagery.layer_id
POST = "20240102noto_suzu_0102do"
PRE = "nendophoto2022"
LAYERS = (BASE, POST, PRE)


def _buildings(n=300):
    raw = np.array([1, 0, 99])[np.arange(n) % 3]
    return pd.DataFrame({"building_id": np.arange(n) * 7 + 300_000_000_003, "label_raw": raw})


def _all_ids(b):
    return set(b["building_id"])


def _write_chips(b, layers=LAYERS):
    for layer in layers:
        for bid in b["building_id"]:
            p = paths.chip_path(layer, int(bid))
            p.parent.mkdir(parents=True, exist_ok=True)
            px = np.full((128, 128, 3), (bid + len(layer)) % 256, dtype=np.uint8)
            Image.fromarray(px).save(p)


def test_audit_sample_deterministic():
    b = _buildings()
    ids = _all_ids(b)
    # drop some buildings from one layer: they must never be drawn
    seam_ok = ids - set(b["building_id"][:30])
    a = audit.draw_sample(b, ids, seam_ok, ids)
    c = audit.draw_sample(b, ids, seam_ok, ids)
    assert a == c
    coll, surv = a
    assert len(coll) == CONFIG.audit.n_collapsed
    assert len(surv) == CONFIG.audit.n_survived
    raw = b.set_index("building_id")["label_raw"]
    assert (raw[list(coll)] == 1).all()
    assert (raw[list(surv)] == 0).all()
    assert not set(coll) & set(b["building_id"][:30])
    # exactly the plan's recipe
    rng = np.random.default_rng(CONFIG.audit.seed)
    cand = b[b["building_id"].isin(seam_ok)].sort_values("building_id")
    cl = cand.loc[cand.label_raw == 1, "building_id"].to_numpy()
    sl = cand.loc[cand.label_raw == 0, "building_id"].to_numpy()
    assert list(coll) == list(rng.choice(cl, 20, replace=False))
    assert list(surv) == list(rng.choice(sl, 20, replace=False))


def test_sample_too_few_candidates():
    b = _buildings(30)
    with pytest.raises(ValueError, match="candidates"):
        audit.draw_sample(b, _all_ids(b), _all_ids(b), _all_ids(b))


def test_audit_blinding(tmp_home):
    b = _buildings()
    _write_chips(b)
    ids = _all_ids(b)
    res = audit.export(
        b, post_layer=POST, pre_layer=PRE, post_ok=ids, seam_ok=ids, pre_ok=ids, salt=b"\x07" * 32
    )
    blind = paths.audit_dir() / "blind"
    files = sorted(p.name for p in blind.iterdir())
    assert len(files) == 120
    key = pd.read_csv(res.key_path, dtype={"file": str, "salt": str})
    assert list(key.columns) == ["file", "building_id", "layer_id", "label_raw", "salt"]
    assert set(key["file"]) == set(files)
    assert len(key) == 120
    for name in files:
        stem = name.rsplit(".", 1)[0]
        assert len(stem) == 16
        assert int(stem, 16) >= 0
        assert not any(layer in name for layer in LAYERS)
    # long ids so a substring hit cannot be chance; fixed salt keeps this deterministic
    for name, bid in zip(key["file"], key["building_id"], strict=True):
        assert str(bid) not in name
        assert "collapsed" not in name
        assert "survived" not in name
    by_name = key.sort_values("file")["label_raw"].astype(int).tolist()
    assert sum(x != y for x, y in itertools.pairwise(by_name)) > 10
    assert key.groupby("building_id").size().eq(3).all()
    assert key["building_id"].nunique() == 40
    # shuffled: layers interleave rather than forming blocks
    seq = key["layer_id"].tolist()
    switches = sum(x != y for x, y in itertools.pairwise(seq))
    assert switches > 40
    assert len(set(key["salt"])) == 1
    assert len(key["salt"].iloc[0]) == 64
    marks = pd.read_csv(res.marks_path, dtype=str, keep_default_na=False)
    assert list(marks.columns) == ["file", "mark"]
    assert (marks["mark"] == "").all()
    assert set(marks["file"]) == set(files)
    html = res.contact_sheet_path.read_text()
    assert all(f in html for f in files)
    assert "pixelated" in html
    for layer in LAYERS:
        assert layer not in html
    # the chip is the layer's chip, byte for byte
    row = key.iloc[0]
    got = np.array(Image.open(blind / row["file"]))
    want = np.array(Image.open(paths.chip_path(row["layer_id"], int(row["building_id"]))))
    assert (got == want).all()


def test_export_reproducible_given_salt(tmp_home):
    b = _buildings()
    _write_chips(b)
    ids = _all_ids(b)
    kw = {"post_layer": POST, "pre_layer": PRE, "post_ok": ids, "seam_ok": ids, "pre_ok": ids}
    k1 = pd.read_csv(audit.export(b, salt=b"\x01" * 32, **kw).key_path)
    k2 = pd.read_csv(audit.export(b, salt=b"\x01" * 32, **kw).key_path)
    assert k1.equals(k2)


def _key(n=6):
    return pd.DataFrame({"file": [f"f{i:02d}.png" for i in range(n)]})


def _marks(key, mark="visible_damage"):
    return pd.DataFrame({"file": key["file"], "mark": mark})


def test_marks_validation(tmp_path):
    key = _key()
    ok = audit.validate_marks(_marks(key), key)
    assert len(ok) == 6
    with pytest.raises(ValueError, match="missing"):
        audit.validate_marks(_marks(key).iloc[:-1], key)
    dup = pd.concat([_marks(key), _marks(key).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        audit.validate_marks(dup, key)
    bad = _marks(key)
    bad.loc[2, "mark"] = "damaged"
    with pytest.raises(ValueError, match="allowed"):
        audit.validate_marks(bad, key)
    blank = _marks(key)
    blank.loc[0, "mark"] = ""
    with pytest.raises(ValueError, match="allowed"):
        audit.validate_marks(blank, key)
    extra = pd.concat([_marks(key), pd.DataFrame({"file": ["zz.png"], "mark": ["cannot_tell"]})])
    with pytest.raises(ValueError, match="unknown"):
        audit.validate_marks(extra, key)
    for m in audit.ALLOWED_MARKS:
        audit.validate_marks(_marks(key, m), key)


def test_read_marks_keeps_blank_strings(tmp_path):
    key = _key(2)
    p = tmp_path / "marks.csv"
    pd.DataFrame({"file": key["file"], "mark": ["", "cannot_tell"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="allowed"):
        audit.validate_marks(audit.read_marks(p), key)


def test_unblind_counts():
    key = pd.DataFrame(
        {
            "file": [f"{i}.png" for i in range(6)],
            "building_id": [1, 1, 1, 2, 2, 2],
            "layer_id": [BASE, POST, PRE] * 2,
            "label_raw": [1, 1, 1, 0, 0, 0],
        }
    )
    marks = pd.DataFrame(
        {
            "file": key["file"],
            "mark": [
                "visible_damage",
                "visible_damage",
                "no_visible_damage",
                "visible_damage",
                "cannot_tell",
                "no_visible_damage",
            ],
        }
    )
    table = audit.unblind(marks, key)
    assert list(table.columns) == ["layer_id", "label_raw", "mark", "n"]
    assert table["n"].sum() == 6
    n = audit.damage_counts(table, seam_layer=BASE, post_layer=POST, pre_layer=PRE)
    assert n == {"n_seam": 1, "n_post": 1, "n_pre": 0}


def test_ingest_writes_report_and_decision_inputs(tmp_home, tmp_path):
    b = _buildings()
    _write_chips(b)
    ids = _all_ids(b)
    res = audit.export(b, post_layer=POST, pre_layer=PRE, post_ok=ids, seam_ok=ids, pre_ok=ids)
    key = pd.read_csv(res.key_path)
    marks = pd.DataFrame({"file": key["file"], "mark": "no_visible_damage"})
    marks.loc[(key.layer_id == POST) & (key.label_raw == 1), "mark"] = "visible_damage"
    marks.to_csv(res.marks_path, index=False)
    out = audit.ingest(
        res.marks_path, res.key_path, rater="AC", date="2026-10-30", seam_layer=BASE,
        post_layer=POST, pre_layer=PRE, root=tmp_path,
    )  # fmt: skip
    assert out.counts == {"n_seam": 0, "n_post": 20, "n_pre": 0}
    rep = pd.read_csv(tmp_path / "reports" / "audit.csv")
    assert rep["n"].sum() == 120
    assert (tmp_path / "reports" / "audit" / "rater.json").read_text().count("AC") == 1


def test_key_stays_uncommitted_until_unblinding(tmp_home):
    lines = {line.strip() for line in (repo_root() / ".gitignore").read_text().splitlines()}
    assert "data/audit/key.csv" in lines
    assert "reports/audit/key.csv" in lines
    # the exported key lives under the ignored data/ tree
    assert paths.audit_dir().name == "audit"
    assert "data/" in lines
    assert audit.KEY_NAME == "key.csv"


def _ignored(rel: str) -> bool:
    """`git check-ignore` when git can read the repo, else the .gitignore lines that matter."""
    r = subprocess.run(
        ["git", "check-ignore", "-q", rel], cwd=repo_root(), check=False, capture_output=True
    )
    if r.returncode in (0, 1):
        return r.returncode == 0
    lines = [x.strip() for x in (repo_root() / ".gitignore").read_text().splitlines()]
    return any(x == rel or (x.endswith("/") and rel.startswith(x)) for x in lines)


def test_exported_key_is_git_ignored(tmp_home, monkeypatch):
    # the path a real run (NOTODATA_HOME=./data) writes the key to
    monkeypatch.setenv("NOTODATA_HOME", "data")
    rel = (paths.audit_dir() / audit.KEY_NAME).as_posix()
    assert _ignored(rel), f"{rel} is not git-ignored"


def test_ingest_never_writes_reports_key(tmp_home, tmp_path):
    b = _buildings()
    _write_chips(b)
    ids = _all_ids(b)
    res = audit.export(b, post_layer=POST, pre_layer=PRE, post_ok=ids, seam_ok=ids, pre_ok=ids)
    key = pd.read_csv(res.key_path)
    pd.DataFrame({"file": key["file"], "mark": "cannot_tell"}).to_csv(res.marks_path, index=False)
    kw = {"seam_layer": BASE, "post_layer": POST, "pre_layer": PRE, "root": tmp_path}
    audit.ingest(res.marks_path, res.key_path, rater="AC", date="2026-10-30", **kw)
    assert not (tmp_path / "reports" / "audit" / "key.csv").exists()


@pytest.mark.parametrize(
    ("rater", "date"), [("", "2026-10-30"), ("  ", "2026-10-30"), ("AC", ""), ("AC", " \t")]
)
def test_ingest_requires_rater_and_date(tmp_path, rater, date):
    kw = {"seam_layer": BASE, "post_layer": POST, "pre_layer": PRE, "root": tmp_path}
    with pytest.raises(ValueError, match="rater and date"):
        audit.ingest(tmp_path / "m.csv", tmp_path / "k.csv", rater=rater, date=date, **kw)
    assert not (tmp_path / "reports").exists()
