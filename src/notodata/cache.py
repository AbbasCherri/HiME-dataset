"""D3 cache keying: a `.meta.json` sidecar beside every derived cache file.

`ensure` is the only validator. It recomputes when the sidecar is missing,
unreadable or differs in any key, and writes the sidecar last, so a crash
between data and sidecar leaves a file that no run will trust.

`PIPELINE_VERSION` is bumped by hand; the version guard (a recorded hash of
`chips.py`) makes forgetting to do so a test failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa

from notodata import chips, config, paths, tiles
from notodata.config import CONFIG
from notodata.io import atomic_write, sha256_bytes, write_parquet_deterministic

log = logging.getLogger(__name__)

CHIPS_SOURCE = Path(chips.__file__)
GUARDED_SOURCES = (CHIPS_SOURCE, Path(tiles.__file__))
"""Files whose maths defines a chip: a change to either needs a version bump."""
LOCK_PATH = Path(__file__).with_name("pipeline_version.json")
"""Records the `PIPELINE_VERSION` and the `chips.py` hash that go with it."""

_CHIP_INDEX_SCHEMA = pa.schema(
    [
        ("building_id", pa.int64()),
        ("status", pa.string()),
        ("path", pa.string()),
        ("sha256", pa.string()),
    ]
)


def sidecar_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.meta.json")


def building_ids_sha256(building_ids: Iterable[int]) -> str:
    ids = sorted(int(b) for b in building_ids)
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())


def expected_meta(
    *,
    layer_ids: list[str],
    zoom: int,
    chip_size: int,
    encoder_id: str | None,
    building_ids: Iterable[int],
) -> dict:
    """The sidecar a valid cache file must carry (the brief's keys plus the id-list hash)."""
    ids = list(building_ids)
    return {
        "chip_pipeline_version": config.PIPELINE_VERSION,
        "layer_ids": list(layer_ids),
        "zoom": int(zoom),
        "chip_size": int(chip_size),
        "encoder_id": encoder_id,
        "row_count": len(ids),
        "building_ids_sha256": building_ids_sha256(ids),
    }


_ABSENT = object()


def _differing_keys(found: dict, expected: dict) -> list[str]:
    return [
        k
        for k in sorted(set(found) | set(expected))
        if found.get(k, _ABSENT) != expected.get(k, _ABSENT)
    ]


def _invalid_reason(path: Path, side: Path, expected: dict) -> str | None:
    """Why `path` cannot be trusted, or None if its sidecar matches `expected`."""
    if not path.exists():
        return "data file missing"
    if not side.exists():
        return "sidecar missing"
    try:
        found = json.loads(side.read_text())
        if not isinstance(found, dict):
            raise ValueError("sidecar is not an object")
    except (OSError, ValueError):
        return "sidecar unreadable"
    diff = _differing_keys(found, expected)
    if not diff:
        return None
    return "keys differ: " + "; ".join(
        f"{k}: {found.get(k, '<absent>')!r} -> {expected.get(k, '<absent>')!r}" for k in diff
    )


def ensure(path: Path | str, expected: dict, compute: Callable[[Path], None]) -> bool:
    """Make `path` valid for `expected`; return True if `compute` ran.

    `compute(path)` must write the data file itself (atomically). The sidecar
    is then written last. Any old sidecar is removed before `compute`, so it
    can never vouch for half-replaced data.
    """
    path = Path(path)
    side = sidecar_path(path)
    # round-trip through JSON so tuples/lists compare the way they are stored
    expected = json.loads(json.dumps(expected))
    reason = _invalid_reason(path, side, expected)
    if reason is None:
        return False
    log.info("recompute %s (%s)", path, reason)
    side.unlink(missing_ok=True)
    compute(path)
    atomic_write(side, (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode())
    return True


def ensure_chip_index(
    layer_id: str, buildings: pd.DataFrame, build: Callable[[], pd.DataFrame]
) -> bool:
    """Keep `chips/<layer>/index.parquet` and its sidecar valid; `build` returns the index."""
    path = paths.chip_index_path(layer_id)
    meta = expected_meta(
        layer_ids=[layer_id],
        zoom=CONFIG.tile_chip.zoom,
        chip_size=CONFIG.tile_chip.chip_size_px,
        encoder_id=None,
        building_ids=buildings["building_id"],
    )

    def compute(p: Path) -> None:
        write_parquet_deterministic(build(), p, schema=_CHIP_INDEX_SCHEMA, sort_by=["building_id"])

    return ensure(path, meta, compute)


# -- version guard ---------------------------------------------------------


class PipelineVersionError(RuntimeError):
    """The hashed sources changed but `PIPELINE_VERSION` was not bumped."""


def sources_sha256(source_paths: tuple[Path, ...] = GUARDED_SOURCES) -> str:
    """One hash over the guarded files' names and bytes, in the given order."""
    digest = hashlib.sha256()
    for p in source_paths:
        digest.update(Path(p).name.encode() + b"\0" + Path(p).read_bytes() + b"\0")
    return digest.hexdigest()


def _read_lock(lock_path: Path) -> dict | None:
    try:
        record = json.loads(Path(lock_path).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def update_pipeline_lock(
    lock_path: Path = LOCK_PATH, source_paths: tuple[Path, ...] = GUARDED_SOURCES
) -> None:
    """Record the current `PIPELINE_VERSION` and source hash (run after a version bump).

    Refuses when the sources changed but `PIPELINE_VERSION` equals the recorded one.
    """
    digest = sources_sha256(source_paths)
    old = _read_lock(lock_path)
    if (
        old is not None
        and "sources_sha256" in old
        and old["sources_sha256"] != digest
        and old.get("pipeline_version") == config.PIPELINE_VERSION
    ):
        raise PipelineVersionError(
            f"guarded sources changed but PIPELINE_VERSION is still {config.PIPELINE_VERSION}; "
            "bump it in config.py first"
        )
    record = {"pipeline_version": config.PIPELINE_VERSION, "sources_sha256": digest}
    atomic_write(lock_path, (json.dumps(record, indent=2) + "\n").encode())


def pipeline_guard_problem(
    lock_path: Path = LOCK_PATH, source_paths: tuple[Path, ...] = GUARDED_SOURCES
) -> str | None:
    """None if the guarded sources and `PIPELINE_VERSION` match the record, else what to do."""
    fix = (
        "bump PIPELINE_VERSION in config.py if the maths changed, "
        "then run scripts/update_pipeline_hash.py"
    )
    record = _read_lock(lock_path)
    if record is None:
        return f"{lock_path} missing or unreadable; run scripts/update_pipeline_hash.py"
    if record.get("pipeline_version") != config.PIPELINE_VERSION:
        return f"PIPELINE_VERSION changed without a recorded hash: {fix}"
    if record.get("sources_sha256") != sources_sha256(source_paths):
        return f"chips.py or tiles.py changed but PIPELINE_VERSION did not: {fix}"
    return None
