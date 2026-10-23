"""D2: chip coverage per layer and the pure imagery-decision logic.

Everything here is a function of counts and layer ids; the blinded audit
(`audit.py`) supplies `n_seam`, `n_post` and `n_pre`. The decision table is
applied literally: no third branch, and a rejected A1 pre layer raises
`NeedsReaudit` rather than guessing (ruling R4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd

from notodata import chips, paths
from notodata.config import CONFIG
from notodata.layers import get_layer

BASELINE = CONFIG.baseline_imagery.layer_id

PRE_REJECT_DAMAGE = 3
"""A1: reject the pre layer when this many of the 20 collapsed show damage in it."""

COVERAGE_COLUMNS = ["layer_id", "group", "value", "n", *chips.STATUSES, "coverage"]


class NeedsReaudit(Exception):
    """The A1 pre layer was rejected; the next layer was never audited (R4)."""

    def __init__(self, next_layer: str | None):
        self.next_layer = next_layer
        super().__init__(f"A1 pre layer rejected; audit {next_layer or 'a new layer'} next")


def meets_threshold(n_ok: int, n: int) -> bool:
    """`n_ok / n >= coverage_threshold`, in exact arithmetic so 80.0% passes and 79.9% fails."""
    return n > 0 and Fraction(n_ok, n) >= Fraction(str(CONFIG.coverage_threshold))


def _row(layer_id: str, group: str, value: str, status: pd.Series) -> dict:
    n = len(status)
    counts = {s: int((status == s).sum()) for s in chips.STATUSES}
    return {
        "layer_id": layer_id,
        "group": group,
        "value": value,
        "n": n,
        **counts,
        "coverage": counts["ok"] / n if n else float("nan"),
    }


def coverage_by_layer(indexes: dict[str, pd.DataFrame], buildings: pd.DataFrame) -> pd.DataFrame:
    """Per layer: status counts and coverage (share `ok`), overall and per `label_raw`.

    Every building in `buildings` counts; one with no index row is `missing`.
    """
    ids = buildings["building_id"].astype("int64")
    raw = buildings["label_raw"].astype(str).to_numpy()
    rows = []
    for layer_id, index in indexes.items():
        status = index.set_index("building_id")["status"].reindex(ids).fillna("missing")
        status = status.reset_index(drop=True)
        if (status == "error").any():
            raise ValueError(f"{layer_id}: {(status == 'error').sum()} chips have status error")
        rows.append(_row(layer_id, "overall", "all", status))
        for value in sorted(set(raw)):
            rows.append(_row(layer_id, "label_raw", value, status[raw == value]))
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def write_coverage_by_layer(table: pd.DataFrame, root: Path | None = None) -> Path:
    out = paths.reports_dir(root) / "coverage_by_layer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    return out


def _first_date_key(layer_id: str):
    dates = get_layer(layer_id).capture_dates
    return (0, min(dates)) if dates else (1, None)


def best_post_layer(n_ok: dict[str, int]) -> str:
    """Highest `ok` count; ties to the earliest capture date, then the lower `layer_id`."""
    top = max(n_ok.values())
    tied = [k for k, v in n_ok.items() if v == top]
    return min(tied, key=lambda k: (_first_date_key(k), k))


def _year(layer_id: str) -> int:
    m = re.search(r"(\d{4})$", layer_id)
    if not m:
        raise ValueError(f"no fiscal year in {layer_id!r}")
    return int(m.group(1))


def rank_pre_layers(n_ok: dict[str, int]) -> list[str]:
    """Pre layers best first: most `ok`, then most recent year, then lower `layer_id`."""
    return sorted(n_ok, key=lambda k: (-n_ok[k], -_year(k), k))


def rule_row(n_seam: int, n_post: int) -> int:
    """The row of the D2 decision table that fires (1, 2 or 3)."""
    if n_seam < CONFIG.audit.decision_threshold:
        return 3
    return 1 if n_post > n_seam else 2


def choose_layers(row: int, best_post: str, a1_pre: str | None) -> tuple[str, str | None]:
    """(post layer, pre layer) for a table row; `a1_pre` is None when A1 does not apply."""
    post = BASELINE if row == 2 else best_post
    pre = a1_pre if a1_pre is not None else (BASELINE if row == 3 else None)
    return post, pre


def check_pre_layer(a1_pre: str, n_pre: int, ranked: list[str]) -> None:
    """Raise `NeedsReaudit(next layer)` if `n_pre` collapsed buildings show damage in `a1_pre`."""
    if n_pre >= PRE_REJECT_DAMAGE:
        later = ranked[ranked.index(a1_pre) + 1 :]
        raise NeedsReaudit(later[0] if later else None)


@dataclass(frozen=True)
class CoverageResult:
    kept: pd.DataFrame
    dropped_by_class: pd.DataFrame
    pre_available: bool
    post_coverage: float
    pre_coverage: float | None


def apply_coverage_rules(
    buildings: pd.DataFrame, post_ok: set[int], pre_ok: set[int] | None
) -> CoverageResult:
    """Spec step 4, in order: restrict to the post layer's buildings if it is under the
    threshold, then drop `pre_available` if the pre layer is under it on the remainder.
    `pre_ok` is None when there is no pre layer."""
    has_post = buildings["building_id"].isin(post_ok)
    post_cov = float(has_post.mean()) if len(buildings) else float("nan")
    if meets_threshold(int(has_post.sum()), len(buildings)):
        kept = buildings
    else:
        kept = buildings[has_post]
    dropped = buildings[~buildings["building_id"].isin(kept["building_id"])]
    by_class = (
        dropped.groupby(dropped["label_raw"], dropna=False)
        .size()
        .rename("dropped")
        .reset_index()
        .sort_values("label_raw", ignore_index=True)
    )
    pre_cov = None
    pre_available = False
    if pre_ok is not None:
        n_pre = int(kept["building_id"].isin(pre_ok).sum())
        pre_cov = n_pre / len(kept) if len(kept) else float("nan")
        pre_available = meets_threshold(n_pre, len(kept))
    return CoverageResult(kept, by_class, pre_available, post_cov, pre_cov)


def write_dropped_by_class(table: pd.DataFrame, root: Path | None = None) -> Path:
    out = paths.reports_dir(root) / "dropped_by_post_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    return out


def _dates(layer_id: str | None) -> list[str]:
    return [] if layer_id is None else [d.isoformat() for d in get_layer(layer_id).capture_dates]


def build_decision(
    *,
    n_seam: int,
    n_post: int,
    n_pre: int,
    best_post: str,
    a1_pre: str | None,
    n_buildings: int,
    n_kept: int,
    pre_available: bool,
    post_coverage: float,
    pre_coverage: float | None,
) -> dict:
    """The `imagery_decision.json` payload: every count, the rule row, layers and dates."""
    row = rule_row(n_seam, n_post)
    post, pre = choose_layers(row, best_post, a1_pre)
    return {
        "n_seam": n_seam,
        "n_post": n_post,
        "n_pre": n_pre,
        "n_audited_collapsed": CONFIG.audit.n_collapsed,
        "decision_threshold": CONFIG.audit.decision_threshold,
        "rule_row": row,
        "best_post_layer": best_post,
        "a1_pre_layer": a1_pre,
        "post_layer": post,
        "post_capture_dates": _dates(post),
        "pre_layer": pre,
        "pre_capture_dates": _dates(pre),
        "n_buildings": n_buildings,
        "n_kept": n_kept,
        "n_dropped": n_buildings - n_kept,
        "post_coverage": post_coverage,
        "pre_coverage": pre_coverage,
        "coverage_threshold": CONFIG.coverage_threshold,
        "pre_available": pre_available,
    }


def write_decision(decision: dict, root: Path | None = None) -> Path:
    out = paths.reports_dir(root) / "imagery_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return out
