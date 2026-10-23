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
