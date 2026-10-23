"""D6 image flags: `has_post`, `has_pre`, the pattern `G`, and coverage by pattern.

Flags come only from chip statuses, so a building whose chip is not `ok` has
no path, no hash and no array anywhere; nothing here can make a placeholder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from notodata import chips, paths
from notodata.tiles import TileFetcher

log = logging.getLogger(__name__)

PATTERNS = ("IP", "I", "none")

FLAG_COLUMNS = [
    "building_id",
    "has_post",
    "has_pre",
    "G",
    "post_chip",
    "post_chip_sha256",
    "pre_chip",
    "pre_chip_sha256",
]


def image_pattern(has_post: int, has_pre: int) -> str:
    """`IP` if both, `I` if post only; a pre chip without a post chip is `none`."""
    if not has_post:
        return "none"
    return "IP" if has_pre else "I"


def _ok_lookup(ids: list[int], index: pd.DataFrame | None, usable: bool) -> pd.DataFrame:
    """Per id: `has`, `path`, `sha256`; absent unless the chip is `ok` and `usable`."""
    out = pd.DataFrame(
        {"has": 0, "path": "", "sha256": ""}, index=pd.Index(ids, name="building_id")
    )
    if not usable or index is None:
        return out
    ok = index[index["status"] == "ok"].set_index("building_id")
    ok = ok.loc[ok.index.intersection(out.index)]
    out.loc[ok.index, "has"] = 1
    out.loc[ok.index, "path"] = ok["path"].to_numpy()
    out.loc[ok.index, "sha256"] = ok["sha256"].to_numpy()
    return out


def flags_from_indexes(
    building_ids,
    post_index: pd.DataFrame,
    pre_index: pd.DataFrame | None,
    pre_available: bool,
) -> pd.DataFrame:
    """Image flags from the post and pre chip indexes (columns of `chips.build_chips`)."""
    ids = [int(b) for b in building_ids]
    post = _ok_lookup(ids, post_index, True)
    pre = _ok_lookup(ids, pre_index, pre_available)
    out = pd.DataFrame(
        {
            "building_id": pd.Series(ids, dtype="int64"),
            "has_post": post["has"].to_numpy(dtype="int8"),
            "has_pre": pre["has"].to_numpy(dtype="int8"),
        }
    )
    out["G"] = [image_pattern(a, b) for a, b in zip(out["has_post"], out["has_pre"], strict=True)]
    out["post_chip"] = post["path"].to_numpy()
    out["post_chip_sha256"] = post["sha256"].to_numpy()
    out["pre_chip"] = pre["path"].to_numpy()
    out["pre_chip_sha256"] = pre["sha256"].to_numpy()
    return out[FLAG_COLUMNS]


def build_image_flags(
    fetcher: TileFetcher,
    buildings: pd.DataFrame,
    post_layer: str,
    pre_layer: str | None,
    pre_available: bool,
) -> pd.DataFrame:
    """Chip the post layer (and the pre layer if available) and derive the flags."""
    post_index = chips.build_chips(fetcher, post_layer, buildings)
    if pre_available and pre_layer is None:
        raise ValueError("pre_available is true but no pre layer was given")
    pre_index = None
    if pre_available:
        pre_index = chips.build_chips(fetcher, pre_layer, buildings)
    flags = flags_from_indexes(buildings["building_id"], post_index, pre_index, pre_available)
    log.info("image patterns: %s", flags["G"].value_counts().to_dict())
    return flags


def _value_label(v) -> str:
    if pd.isna(v):
        return "unlabelled"
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _pattern_rows(group: str, value: str, sub: pd.DataFrame) -> list[tuple]:
    n = len(sub)
    rows = []
    for p in PATTERNS:
        c = int((sub["G"] == p).sum())
        rows.append((group, value, p, c, c / n if n else float("nan")))
    return rows


def _group_rows(flags: pd.DataFrame, col: str) -> list[tuple]:
    keys = flags[col].map(_value_label)
    rows = []
    for value in sorted(keys.unique()):
        rows.extend(_pattern_rows(col, value, flags[keys == value]))
    return rows


def _status_rows(status_counts: dict[str, pd.Series]) -> list[tuple]:
    return [
        ("chip_status", layer, status, int(counts.get(status, 0)), float("nan"))
        for layer, counts in status_counts.items()
        for status in chips.STATUSES
    ]


def coverage_by_pattern(
    flags: pd.DataFrame, status_counts: dict[str, pd.Series] | None = None
) -> pd.DataFrame:
    """Counts and shares of each pattern overall, per `label_raw` and per `y_collapse`.

    `flags` needs `G`; `label_raw` and `y_collapse` groups are added when those
    columns exist. `status_counts` maps layer id to chip status counts and adds
    `chip_status` rows (no share).
    """
    rows = _pattern_rows("overall", "all", flags)
    for col in ("label_raw", "y_collapse"):
        if col in flags:
            rows.extend(_group_rows(flags, col))
    rows.extend(_status_rows(status_counts or {}))
    return pd.DataFrame(rows, columns=["group", "value", "pattern", "count", "share"])


def write_coverage_report(
    flags: pd.DataFrame,
    status_counts: dict[str, pd.Series] | None = None,
    root: Path | None = None,
) -> Path:
    out = paths.reports_dir(root) / "coverage_by_pattern.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    coverage_by_pattern(flags, status_counts).to_csv(out, index=False, lineterminator="\n")
    return out
