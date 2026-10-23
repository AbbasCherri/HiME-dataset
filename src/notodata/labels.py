"""Damage code to collapse label (D4).

Pure mapping from the source damage codes to `y_collapse`, `observable` and
`y_collapse_sens`. `y_collapse_sens` is a sensitivity column only: it never
feeds geometry, folds, splits or clients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from notodata.config import CODE99_MEANING, CONFIG

# code -> (y_collapse, observable). Codes 9 and 99 cannot be judged from imagery.
_TABLE: dict[int, tuple[float, bool]] = {
    0: (0.0, True),
    1: (1.0, True),
    9: (np.nan, False),
    99: (np.nan, False),
}

# CODE99_MEANING -> y_collapse_sens for code 99.
_SENS_99: dict[str, float] = {
    "no_longer_present": 1.0,
    "missing_inconsistent": np.nan,
    "unknown": np.nan,
}

MAX_CONFLICT_SHARE = 0.01


def _resolve_codes(df: pd.DataFrame) -> pd.Series:
    primary = CONFIG.label_column
    fallback = CONFIG.label_fallback_column
    if primary not in df.columns:
        if fallback not in df.columns:
            raise KeyError(f"neither {primary!r} nor {fallback!r} is present")
        return df[fallback]
    codes = df[primary]
    if fallback not in df.columns:
        return codes
    both = codes.notna() & df[fallback].notna()
    n_conflict = int((both & (codes != df[fallback])).sum())
    if len(df) and n_conflict / len(df) > MAX_CONFLICT_SHARE:
        raise ValueError(
            f"{primary!r} and {fallback!r} conflict on {n_conflict} of {len(df)} rows "
            f"(limit {MAX_CONFLICT_SHARE:.0%})"
        )
    return codes.fillna(df[fallback])


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return `y_collapse`, `observable` and `y_collapse_sens`, indexed like `df`."""
    codes = _resolve_codes(df)
    if codes.isna().any():
        raise ValueError("unknown damage code: missing value")
    unknown = sorted(set(codes.astype(int)) - set(_TABLE))
    if unknown:
        raise ValueError(f"unknown damage code(s): {unknown}")
    ints = codes.astype(int)
    y = ints.map(lambda c: _TABLE[c][0]).astype("float64")
    observable = ints.map(lambda c: _TABLE[c][1]).astype(bool)
    sens = y.copy()
    sens[ints == 99] = _SENS_99[CODE99_MEANING]
    return pd.DataFrame({"y_collapse": y, "observable": observable, "y_collapse_sens": sens})
