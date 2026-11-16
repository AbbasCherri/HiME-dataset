"""Damage code to collapse label (D4).

Pure mapping from the source damage codes to `y_collapse`, `observable` and
`y_collapse_sens`. `y_collapse_sens` is a sensitivity column only: it never
feeds geometry, folds, splits or clients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from notodata.config import CODE99_MEANING, CONFIG, MAX_LABEL_CONFLICT_SHARE

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

ID_COLUMN = "building_id"
N_EXAMPLES = 5


def _conflict_mask(df: pd.DataFrame) -> pd.Series:
    primary, fallback = CONFIG.label_column, CONFIG.label_fallback_column
    if primary not in df.columns or fallback not in df.columns:
        return pd.Series(False, index=df.index)
    both = df[primary].notna() & df[fallback].notna()
    return both & (df[primary] != df[fallback])


def conflict_report(df: pd.DataFrame) -> tuple[int, list]:
    """Count rows where the two label columns disagree, with the first five ids."""
    mask = _conflict_mask(df)
    ids = df.index if ID_COLUMN not in df.columns else df[ID_COLUMN]
    return int(mask.sum()), ids[mask].tolist()[:N_EXAMPLES]


def _resolve_codes(df: pd.DataFrame, max_conflict_share: float) -> tuple[pd.Series, pd.Series]:
    primary, fallback = CONFIG.label_column, CONFIG.label_fallback_column
    if primary not in df.columns and fallback not in df.columns:
        raise KeyError(f"neither {primary!r} nor {fallback!r} is present")
    n_conflict, examples = conflict_report(df)
    if len(df) and n_conflict / len(df) > max_conflict_share:
        raise ValueError(
            f"{primary!r} and {fallback!r} conflict on {n_conflict} of {len(df)} rows "
            f"(limit {max_conflict_share:.0%}); first ids {examples}"
        )
    if primary not in df.columns:
        return df[fallback], pd.Series(fallback, index=df.index)
    if fallback not in df.columns:
        return df[primary], pd.Series(primary, index=df.index)
    use_primary = df[primary].notna()
    source = pd.Series(np.where(use_primary, primary, fallback), index=df.index)
    return df[primary].fillna(df[fallback]), source


def map_labels(
    df: pd.DataFrame, max_conflict_share: float = MAX_LABEL_CONFLICT_SHARE
) -> pd.DataFrame:
    """Return the mapped label columns, indexed like `df`.

    `damage_val` wins wherever it is present; `label_source` records which
    column supplied each code and `label_conflict` flags disagreements.
    """
    codes, source = _resolve_codes(df, max_conflict_share)
    if codes.isna().any():
        raise ValueError("missing damage code: neither label column has a value")
    if (codes != np.floor(codes)).any():
        raise ValueError(
            "non-integer damage code(s): " + str(sorted(set(codes[codes != np.floor(codes)])))
        )
    ints = codes.astype(int)
    unknown = sorted(set(ints) - set(_TABLE))
    if unknown:
        raise ValueError(f"unknown damage code(s): {unknown}")
    y = ints.map(lambda c: _TABLE[c][0]).astype("float64")
    observable = ints.map(lambda c: _TABLE[c][1]).astype(bool)
    sens = y.copy()
    sens[ints == 99] = _SENS_99[CODE99_MEANING]
    return pd.DataFrame(
        {
            "label_raw": ints.astype("int16"),
            "label_source": source,
            "label_conflict": _conflict_mask(df).astype("int8"),
            "y_collapse": y,
            "observable": observable,
            "y_collapse_sens": sens,
        }
    )
