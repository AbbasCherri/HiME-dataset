import json

import pandas as pd
import pytest

from notodata import coverage
from notodata.config import CONFIG

BASE = CONFIG.baseline_imagery.layer_id
SUZU2 = "20240102noto_suzu_0102do"
HIGASHI2 = "20240102noto_wazimahigashi_0102do"
SUZU5 = "20240102_noto_suzu_0105do"


def _index(ids, ok, status="missing"):
    return pd.DataFrame(
        {
            "building_id": ids,
            "status": ["ok" if i in ok else status for i in ids],
            "path": "",
            "sha256": "",
        }
    )


def _buildings(n=10):
    return pd.DataFrame({"building_id": range(n), "label_raw": [0] * (n - 3) + [1, 1, 99]})


def test_coverage_counts():
    b = _buildings()
    idx = _index(list(range(10)), {0, 1, 8}, "blank")
    idx.loc[2, "status"] = "partial"
    t = coverage.coverage_by_layer({"L": idx}, b)
    row = t[(t.layer_id == "L") & (t.group == "overall")].iloc[0]
    assert (row.n, row.ok, row.blank, row.partial, row.missing) == (10, 3, 6, 1, 0)
    assert row.coverage == pytest.approx(0.3)
    r1 = t[(t.group == "label_raw") & (t.value == "1")].iloc[0]
    assert (r1.n, r1.ok, r1.coverage) == (2, 1, 0.5)


def test_error_status_refused():
    idx = _index([0, 1], {0}, "error")
    with pytest.raises(ValueError, match="error"):
        coverage.coverage_by_layer(
            {"L": idx}, pd.DataFrame({"building_id": [0, 1], "label_raw": 0})
        )


def test_best_post_ties_earliest_date_then_layer_id():
    assert coverage.best_post_layer({SUZU5: 9, SUZU2: 9, "20240102noto_nanao_0117do": 9}) == SUZU2
    assert coverage.best_post_layer({HIGASHI2: 9, SUZU2: 9}) == SUZU2
    assert coverage.best_post_layer({SUZU2: 8, SUZU5: 9}) == SUZU5


def test_best_pre_ties_most_recent_year():
    assert (
        coverage.rank_pre_layers({"nendophoto2019": 5, "nendophoto2021": 5, "nendophoto2020": 4})[0]
        == "nendophoto2021"
    )
    assert (
        coverage.rank_pre_layers({"nendophoto2019": 6, "nendophoto2021": 5})[0] == "nendophoto2019"
    )


@pytest.mark.parametrize("n_seam", [9, 10, 11])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_decision_rule(n_seam, delta):
    n_post = n_seam + delta
    row = coverage.rule_row(n_seam, n_post)
    if n_seam < 10:
        assert row == 3
    elif n_post > n_seam:
        assert row == 1
    else:
        assert row == 2


def test_chosen_layers_per_row():
    kw = {"best_post": SUZU2, "a1_pre": "nendophoto2022"}
    assert coverage.choose_layers(1, **kw) == (SUZU2, "nendophoto2022")
    assert coverage.choose_layers(2, **kw) == (BASE, "nendophoto2022")
    assert coverage.choose_layers(3, **kw) == (SUZU2, "nendophoto2022")
    none = {"best_post": SUZU2, "a1_pre": None}
    assert coverage.choose_layers(1, **none) == (SUZU2, None)
    assert coverage.choose_layers(2, **none) == (BASE, None)
    assert coverage.choose_layers(3, **none) == (SUZU2, BASE)


def test_coverage_thresholds():
    assert coverage.meets_threshold(800, 1000)
    assert not coverage.meets_threshold(799, 1000)
    b = pd.DataFrame({"building_id": range(1000), "label_raw": [0] * 900 + [1] * 100})
    # post covers 799/1000: restrict, dropped by class
    post_ok = set(range(799))
    res = coverage.apply_coverage_rules(b, post_ok, set(range(799)))
    assert len(res.kept) == 799
    assert res.dropped_by_class.set_index("label_raw")["dropped"].to_dict() == {0: 101, 1: 100}
    assert res.pre_available
    # exactly 80% passes, nothing dropped
    res = coverage.apply_coverage_rules(b, set(range(800)), set(range(800)))
    assert len(res.kept) == 1000
    assert res.dropped_by_class.empty
    res = coverage.apply_coverage_rules(b, set(range(1000)), set(range(1000)))
    assert res.dropped_by_class["dropped"].sum() == 0


def test_pre_coverage_on_remaining_buildings():
    b = pd.DataFrame({"building_id": range(100), "label_raw": 0})
    post_ok = set(range(50))
    # post 50% -> keep 50; pre covers 40 of those = 80% -> available
    res = coverage.apply_coverage_rules(b, post_ok, set(range(40)) | set(range(60, 100)))
    assert len(res.kept) == 50
    assert res.pre_available
    res = coverage.apply_coverage_rules(b, post_ok, set(range(39)) | set(range(60, 100)))
    assert not res.pre_available
    res = coverage.apply_coverage_rules(b, post_ok, None)
    assert not res.pre_available


def test_a1_rejection_needs_reaudit():
    ranked = ["nendophoto2022", "nendophoto2021"]
    with pytest.raises(coverage.NeedsReaudit) as e:
        coverage.check_pre_layer(ranked[0], 3, ranked)
    assert e.value.next_layer == "nendophoto2021"
    coverage.check_pre_layer(ranked[0], 2, ranked)
    with pytest.raises(coverage.NeedsReaudit) as e:
        coverage.check_pre_layer(ranked[1], 5, ranked)
    assert e.value.next_layer is None


def test_decision_json_has_counts_and_rule_row(tmp_path):
    d = coverage.build_decision(
        n_seam=11,
        n_post=12,
        n_pre=1,
        best_post=SUZU2,
        a1_pre="nendophoto2022",
        n_buildings=1000,
        n_kept=900,
        pre_available=True,
        post_coverage=0.9,
        pre_coverage=0.95,
    )
    assert d["rule_row"] == 1
    assert d["post_layer"] == SUZU2
    assert d["post_capture_dates"] == ["2024-01-02"]
    assert d["pre_layer"] == "nendophoto2022"
    for k in ("n_seam", "n_post", "n_pre", "n_buildings", "n_kept", "pre_available"):
        assert k in d
    p = coverage.write_decision(d, root=tmp_path)
    assert json.loads(p.read_text())["rule_row"] == 1
    assert p.name == "imagery_decision.json"
