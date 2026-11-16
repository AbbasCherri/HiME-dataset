import numpy as np
import pandas as pd
import pytest

from notodata import labels
from notodata.config import CONFIG


def test_mapping_each_branch():
    df = pd.DataFrame({CONFIG.label_column: [0, 1, 9, 99]})
    out = labels.map_labels(df)
    assert out["observable"].tolist() == [True, True, False, False]
    assert out["y_collapse"].iloc[:2].tolist() == [0.0, 1.0]
    assert out["y_collapse"].iloc[2:].isna().all()
    assert out["y_collapse_sens"].iloc[:2].tolist() == [0.0, 1.0]
    assert out["y_collapse_sens"].iloc[2:].isna().all()


def test_code99_sens_follows_config_meaning(monkeypatch):
    df = pd.DataFrame({CONFIG.label_column: [99]})
    monkeypatch.setattr(labels, "CODE99_MEANING", "no_longer_present")
    assert labels.map_labels(df)["y_collapse_sens"].tolist() == [1.0]
    monkeypatch.setattr(labels, "CODE99_MEANING", "unknown")
    assert np.isnan(labels.map_labels(df)["y_collapse_sens"].iloc[0])


def test_label_fallback():
    df = pd.DataFrame({CONFIG.label_fallback_column: [0, 1, 9]})
    out = labels.map_labels(df)
    assert out["observable"].tolist() == [True, True, False]
    mixed = pd.DataFrame({CONFIG.label_column: [0, np.nan], CONFIG.label_fallback_column: [0, 1]})
    assert labels.map_labels(mixed)["y_collapse"].tolist() == [0.0, 1.0]


def test_label_conflict_rule():
    n = 1000
    primary = [0] * n
    fallback = [0] * n
    for i in range(10):  # exactly 1%: tolerated
        fallback[i] = 1
    ok = pd.DataFrame({CONFIG.label_column: primary, CONFIG.label_fallback_column: fallback})
    labels.map_labels(ok)
    fallback[10] = 1  # just over 1%
    bad = pd.DataFrame({CONFIG.label_column: primary, CONFIG.label_fallback_column: fallback})
    with pytest.raises(ValueError, match="conflict"):
        labels.map_labels(bad)


def test_unknown_code_aborts():
    df = pd.DataFrame({CONFIG.label_column: [0, 2]})
    with pytest.raises(ValueError, match="unknown"):
        labels.map_labels(df)


def test_conflict_keeps_damage_val_and_flags_row():
    df = pd.DataFrame(
        {
            "building_id": [10, 11, 12],
            CONFIG.label_column: [0, 1, np.nan],
            CONFIG.label_fallback_column: [1, 1, 9],
        }
    )
    out = labels.map_labels(df, max_conflict_share=0.5)
    assert out["label_raw"].tolist() == [0, 1, 9]
    assert out["label_source"].tolist() == [
        CONFIG.label_column,
        CONFIG.label_column,
        CONFIG.label_fallback_column,
    ]
    assert out["label_conflict"].tolist() == [1, 0, 0]
    assert out["y_collapse"].iloc[0] == 0.0


def test_conflict_report_counts_and_lists_first_five_ids():
    n = 8
    df = pd.DataFrame(
        {
            "building_id": list(range(100, 100 + n)),
            CONFIG.label_column: [0] * n,
            CONFIG.label_fallback_column: [1] * n,
        }
    )
    count, examples = labels.conflict_report(df)
    assert count == n
    assert examples == [100, 101, 102, 103, 104]


def test_non_integer_code_aborts():
    df = pd.DataFrame({CONFIG.label_column: [0.0, 1.5]})
    with pytest.raises(ValueError, match="non-integer"):
        labels.map_labels(df)


def test_nan_in_both_columns_aborts():
    df = pd.DataFrame({CONFIG.label_column: [0, np.nan], CONFIG.label_fallback_column: [0, np.nan]})
    with pytest.raises(ValueError, match="missing"):
        labels.map_labels(df)


def test_code99_row_alone_has_nan_sens_under_config_meaning():
    out = labels.map_labels(pd.DataFrame({CONFIG.label_column: [99]}))
    assert np.isnan(out["y_collapse"].iloc[0])
    assert np.isnan(out["y_collapse_sens"].iloc[0])
