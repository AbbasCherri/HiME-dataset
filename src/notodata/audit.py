"""D2: the blinded imagery audit, export and ingest.

Export draws 20 collapsed (raw code 1) and 20 survived (raw code 0) buildings
from those with an `ok` chip in all three audited layers, then writes their
120 chips under salted-hash names so the rater cannot tell layer or building.
Ingest validates the marks, unblinds and counts. The decision itself is
`coverage.build_decision`; nothing here applies the rule table.
"""

from __future__ import annotations

import hashlib
import html
import json
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from notodata import paths
from notodata.config import CONFIG

ALLOWED_MARKS = ("visible_damage", "no_visible_damage", "cannot_tell")
KEY_NAME = "key.csv"
MARKS_NAME = "marks.csv"
KEY_COLUMNS = ["file", "building_id", "layer_id", "label_raw", "salt"]
COLLAPSED_RAW = 1
SURVIVED_RAW = 0
NAME_HEX_CHARS = 16
SALT_BYTES = 32
CONTACT_SCALE = 4


@dataclass(frozen=True)
class ExportResult:
    key_path: Path
    marks_path: Path
    contact_sheet_path: Path
    blind_dir: Path


@dataclass(frozen=True)
class IngestResult:
    table: pd.DataFrame
    counts: dict[str, int]


def _draw(
    buildings: pd.DataFrame,
    post_ok: set[int],
    seam_ok: set[int],
    pre_ok: set[int],
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Collapsed then survived, without replacement, from the sorted candidates."""
    covered = buildings["building_id"].isin(post_ok & seam_ok & pre_ok)
    cand = buildings[covered].sort_values("building_id")
    groups = []
    for raw, n in (
        (COLLAPSED_RAW, CONFIG.audit.n_collapsed),
        (SURVIVED_RAW, CONFIG.audit.n_survived),
    ):
        pool = cand.loc[cand["label_raw"].astype(int) == raw, "building_id"].to_numpy()
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} candidates with raw code {raw}, need {n}")
        groups.append(tuple(int(i) for i in rng.choice(pool, n, replace=False)))
    return groups[0], groups[1]


def draw_sample(
    buildings: pd.DataFrame, post_ok: set[int], seam_ok: set[int], pre_ok: set[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The audit sample alone, from a fresh generator seeded by the config."""
    return _draw(buildings, post_ok, seam_ok, pre_ok, np.random.default_rng(CONFIG.audit.seed))


def blind_name(salt: bytes, building_id: int, layer_id: str) -> str:
    digest = hashlib.sha256(salt + str(building_id).encode() + layer_id.encode()).hexdigest()
    return f"{digest[:NAME_HEX_CHARS]}.png"


def _contact_sheet(files: list[str]) -> str:
    size = CONFIG.tile_chip.chip_size_px * CONTACT_SCALE
    cells = "\n".join(
        f'<div class="c"><img src="blind/{name}" alt=""><span>{name}</span></div>'
        for name in map(html.escape, files)
    )
    return (
        "<!doctype html><meta charset=utf-8><title>audit contact sheet</title>\n"
        "<style>.c{position:relative;display:inline-block;margin:6px;vertical-align:top}"
        f"img{{width:{size}px;height:{size}px;image-rendering:pixelated;display:block}}"
        ".c::before,.c::after{content:'';position:absolute;background:rgba(255,255,255,.35)}"
        f".c::before{{left:{size // 2}px;top:{size // 2 - 8}px;width:1px;height:16px}}"
        f".c::after{{left:{size // 2 - 8}px;top:{size // 2}px;width:16px;height:1px}}"
        "span{font:11px monospace}</style>\n" + cells + "\n"
    )


def export(
    buildings: pd.DataFrame,
    *,
    post_layer: str,
    pre_layer: str,
    post_ok: set[int],
    seam_ok: set[int],
    pre_ok: set[int],
    salt: bytes | None = None,
) -> ExportResult:
    """Write the 120 blinded chips, `key.csv`, a blank `marks.csv` and `contact_sheet.html`."""
    rng = np.random.default_rng(CONFIG.audit.seed)
    collapsed, survived = _draw(buildings, post_ok, seam_ok, pre_ok, rng)
    salt = salt if salt is not None else secrets.token_bytes(SALT_BYTES)
    raw = buildings.set_index("building_id")["label_raw"]
    layers = (CONFIG.baseline_imagery.layer_id, post_layer, pre_layer)
    rows = [
        (blind_name(salt, b, layer), b, layer, int(raw[b]), salt.hex())
        for b in (*collapsed, *survived)
        for layer in layers
    ]
    key = pd.DataFrame(rows, columns=KEY_COLUMNS).iloc[rng.permutation(len(rows))]
    key = key.reset_index(drop=True)
    if key["file"].duplicated().any():
        raise RuntimeError("blind file name collision; draw a new salt")
    root = paths.audit_dir()
    blind = root / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    for f, b, layer in zip(key["file"], key["building_id"], key["layer_id"], strict=True):
        shutil.copyfile(paths.chip_path(layer, int(b)), blind / f)
    key_path, marks_path, sheet = root / KEY_NAME, root / MARKS_NAME, root / "contact_sheet.html"
    key.to_csv(key_path, index=False, lineterminator="\n")
    pd.DataFrame({"file": key["file"], "mark": ""}).to_csv(
        marks_path, index=False, lineterminator="\n"
    )
    sheet.write_text(_contact_sheet(key["file"].tolist()))
    return ExportResult(key_path, marks_path, sheet, blind)


def read_marks(path: Path) -> pd.DataFrame:
    """Read as strings and keep blanks as '' so an unmarked row is rejected, not NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_marks(marks: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Exactly one allowed mark per key file; raises ValueError naming the first problem."""
    if list(marks.columns) != ["file", "mark"]:
        raise ValueError(f"marks columns must be ['file', 'mark'], got {list(marks.columns)}")
    dup = marks.loc[marks["file"].duplicated(), "file"]
    if len(dup):
        raise ValueError(f"duplicate marks for {sorted(set(dup))[:3]}")
    unknown = set(marks["file"]) - set(key["file"])
    if unknown:
        raise ValueError(f"unknown files in marks: {sorted(unknown)[:3]}")
    missing = set(key["file"]) - set(marks["file"])
    if missing:
        raise ValueError(f"{len(missing)} files missing from marks, e.g. {sorted(missing)[:3]}")
    bad = marks[~marks["mark"].isin(ALLOWED_MARKS)]
    if len(bad):
        raise ValueError(
            f"marks must be one of the allowed {ALLOWED_MARKS}; bad: {bad.iloc[0].to_dict()}"
        )
    return marks.copy()


def unblind(marks: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Counts of each mark by layer and raw class (columns layer_id, label_raw, mark, n)."""
    joined = marks.merge(key[["file", "layer_id", "label_raw"]], on="file", validate="1:1")
    table = joined.groupby(["layer_id", "label_raw", "mark"]).size().rename("n").reset_index()
    return table.sort_values(["layer_id", "label_raw", "mark"], ignore_index=True)


def damage_counts(
    table: pd.DataFrame, *, seam_layer: str, post_layer: str, pre_layer: str
) -> dict[str, int]:
    """Collapsed buildings marked `visible_damage` per layer, as the decision rule's inputs."""
    hit = table[
        (table["label_raw"].astype(int) == COLLAPSED_RAW) & (table["mark"] == "visible_damage")
    ]

    def count(layer: str) -> int:
        return int(hit.loc[hit["layer_id"] == layer, "n"].sum())

    return {
        "n_seam": count(seam_layer),
        "n_post": count(post_layer),
        "n_pre": count(pre_layer),
    }


def ingest(
    marks_path: Path,
    key_path: Path,
    *,
    rater: str,
    date: str,
    seam_layer: str,
    post_layer: str,
    pre_layer: str,
    root: Path | None = None,
) -> IngestResult:
    """Validate, unblind, write `reports/audit.csv` and `reports/audit/rater.json`."""
    key = pd.read_csv(key_path)
    marks = validate_marks(read_marks(marks_path), key)
    table = unblind(marks, key)
    counts = damage_counts(table, seam_layer=seam_layer, post_layer=post_layer, pre_layer=pre_layer)
    reports = paths.reports_dir(root)
    (reports / "audit").mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "audit.csv", index=False, lineterminator="\n")
    meta = {"rater": rater, "date": date}
    (reports / "audit" / "rater.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return IngestResult(table, counts)
