"""Phase 1: the building table from the Zenodo gpkg and the USGS ShakeMap grid.

Buildings and damage codes come from the pinned ESSD GeoPackage (MD5 checked
before it is read). The seven seismic columns are bilinearly sampled from the
pinned ShakeMap `grid.xml` at each footprint centroid. Units are taken from the
grid's own header, never from a fixed factor: PGA and the three PSA fields are
`%g` there and are converted to g; PGV (cm/s) and MMI keep their native units.
`MMI_shape` is defined in none of the sources checked (ruling P12), so it is a
null float column. Only this module imports geopandas; the one network call is
`http_get`, injected everywhere else so tests stay offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.interpolate import RegularGridInterpolator

from notodata import paths
from notodata.config import CONFIG
from notodata.io import atomic_write, sha256_file, write_parquet_deterministic

log = logging.getLogger(__name__)

SOURCE_FILES_NAME = "source_files.json"

# grid.xml field -> (output column, accepted header units -> factor to the output unit)
_FIELDS: dict[str, tuple[str, dict[str, float]]] = {
    "MMI": ("MMI_original", {"intensity": 1.0}),
    "PGA": ("PGA", {"%g": 0.01, "g": 1.0}),
    "PGV": ("PGV", {"cm/s": 1.0}),
    "PSA03": ("SA_0_3", {"%g": 0.01, "g": 1.0}),
    "PSA10": ("SA_1_0", {"%g": 0.01, "g": 1.0}),
    "PSA30": ("SA_3_0", {"%g": 0.01, "g": 1.0}),
}
_UNDEFINED_COLUMN = "MMI_shape"

SCHEMA = pa.schema(
    [
        ("building_id", pa.int64()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        *[(c, pa.float64()) for c in CONFIG.seismic_columns],
        ("damage_val", pa.float64()),
        ("label", pa.float64()),
        ("municipality", pa.string()),
        ("grid_outside", pa.int8()),
        ("bbox_outside", pa.int8()),
    ]
)


class SourceError(RuntimeError):
    """A source file is not the pinned one, or is not shaped as expected."""


# -- files -----------------------------------------------------------------


def md5_file(path: Path | str) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_get(url: str) -> bytes:
    """The only network call in this module."""
    req = urllib.request.Request(url, headers={"User-Agent": CONFIG.fetch.user_agent})
    with urllib.request.urlopen(req, timeout=CONFIG.fetch.timeout_s) as resp:
        return resp.read()


def source_files_path(raw: Path | None = None) -> Path:
    return (raw or paths.raw_dir()) / SOURCE_FILES_NAME


def read_source_files(raw: Path | None = None) -> dict[str, dict]:
    path = source_files_path(raw)
    return json.loads(path.read_text()) if path.exists() else {}


def record_source_file(path: Path, raw: Path | None = None) -> dict:
    """Record name, size and sha256 of a raw file in `raw/source_files.json`."""
    record = read_source_files(raw)
    record[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    out = json.dumps(record, indent=2, sort_keys=True) + "\n"
    atomic_write(source_files_path(raw), out.encode())
    return record[path.name]


def grid_url() -> str:
    s = CONFIG.shakemap
    return f"{s.base_url}/download/{s.grid_file}"


def fetch_grid(raw: Path | None = None, get: Callable[[str], bytes] = http_get) -> Path:
    """Download `grid.xml` once into `raw`, cache it and record its sha256."""
    raw = raw or paths.raw_dir()
    path = raw / CONFIG.shakemap.grid_file
    if not path.exists():
        log.info("downloading %s", grid_url())
        atomic_write(path, get(grid_url()))
    record_source_file(path, raw)
    return path


# -- buildings -------------------------------------------------------------


def _centroids(geometry: gpd.GeoSeries) -> tuple[np.ndarray, np.ndarray]:
    """Planar centroid in degrees; footprints are metres across, so the error is far below 1e-9."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        points = geometry.centroid
    return points.x.to_numpy(dtype="float64"), points.y.to_numpy(dtype="float64")


def read_buildings(path: Path | str, expected_md5: str = CONFIG.buildings.md5) -> pd.DataFrame:
    """Read the gpkg after verifying its MD5; one row per footprint, centroid lon/lat.

    `building_id` is the GeoPackage `fid`. `damage_val` is the primary code column; the
    shipped file's second code column `damage_2` is carried as `label`, the fallback.
    """
    found = md5_file(path)
    if found != expected_md5:
        raise SourceError(f"MD5 of {path} is {found}, expected {expected_md5}")
    gdf = gpd.read_file(path, engine="pyogrio", fid_as_index=True)
    lon, lat = _centroids(gdf.geometry)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        bad = int((~(np.isfinite(lon) & np.isfinite(lat))).sum())
        raise SourceError(f"{bad} footprints have non-finite centroid coordinates")
    second = CONFIG.label_fallback_column
    return pd.DataFrame(
        {
            "building_id": gdf.index.to_numpy(dtype="int64"),
            "lat": lat,
            "lon": lon,
            CONFIG.label_column: gdf[CONFIG.label_column].to_numpy(dtype="float64"),
            second: gdf["damage_2"].to_numpy(dtype="float64")
            if "damage_2" in gdf
            else np.full(len(gdf), np.nan),
            "municipality": gdf["municipality"].astype("string").to_numpy(dtype=object),
        }
    )


# -- ShakeMap grid ---------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    lon: np.ndarray
    lat: np.ndarray
    fields: dict[str, np.ndarray]
    """Header field name -> array of shape (nlat, nlon), south to north, in native units."""
    units: dict[str, str]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _check_pin(root: ET.Element) -> None:
    pin = CONFIG.shakemap
    event = root.get("event_id") or root.get("shakemap_id")
    if event != pin.event_id:
        raise SourceError(f"grid event is {event!r}, pinned {pin.event_id!r}")
    if root.get("shakemap_version") != str(pin.version):
        raise SourceError(f"grid version is {root.get('shakemap_version')!r}, pinned {pin.version}")


def _units(fields: list[tuple[str, str]]) -> dict[str, str]:
    units = dict(fields)
    for name, (_, accepted) in _FIELDS.items():
        if name not in units:
            raise SourceError(f"grid.xml has no {name} field")
        if units[name] not in accepted:
            raise SourceError(f"grid.xml declares units {units[name]!r} for {name}")
    return units


def _axes(spec: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(float(spec.get("lon_min")), float(spec.get("lon_max")), int(spec.get("nlon")))
    lat = np.linspace(float(spec.get("lat_min")), float(spec.get("lat_max")), int(spec.get("nlat")))
    return lon, lat


def _node_index(coord: np.ndarray, axis: np.ndarray) -> np.ndarray:
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    idx = np.rint((coord - axis[0]) / step).astype(int)
    if (idx < 0).any() or (idx >= len(axis)).any() or not np.allclose(axis[idx], coord, atol=1e-6):
        raise SourceError("grid nodes do not lie on the declared lattice")
    return idx


def parse_grid(xml: bytes) -> Grid:
    """Parse `grid.xml`: pin check, header units, and every node placed by its LON/LAT."""
    root = ET.fromstring(xml)
    _check_pin(root)
    kids = {_local(k.tag): k for k in root}
    header = [(f.get("name"), f.get("units")) for f in root if _local(f.tag) == "grid_field"]
    units = _units(header)
    lon, lat = _axes(kids["grid_specification"])
    data = np.array(kids["grid_data"].text.split(), dtype="float64").reshape(-1, len(header))
    if len(data) != len(lon) * len(lat):
        raise SourceError(f"grid has {len(data)} nodes, header says {len(lon) * len(lat)}")
    names = [n for n, _ in header]
    ix = _node_index(data[:, names.index("LON")], lon)
    iy = _node_index(data[:, names.index("LAT")], lat)
    fields = {}
    for name in _FIELDS:
        arr = np.full((len(lat), len(lon)), np.nan)
        arr[iy, ix] = data[:, names.index(name)]
        fields[name] = arr
    return Grid(lon=lon, lat=lat, fields=fields, units=units)


def sample(grid: Grid, lon: np.ndarray, lat: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Bilinear samples of the seismic columns plus a mask of points outside the lattice.

    Outside points get NaN (never clamped to the edge). Units follow the header.
    """
    pts = np.column_stack([lat, lon])
    out = {}
    for name, (column, accepted) in _FIELDS.items():
        interp = RegularGridInterpolator(
            (grid.lat, grid.lon), grid.fields[name], method="linear", bounds_error=False
        )
        out[column] = interp(pts) * accepted[grid.units[name]]
    out[_UNDEFINED_COLUMN] = np.full(len(lon), np.nan)
    outside = (
        (lon < grid.lon[0]) | (lon > grid.lon[-1]) | (lat < grid.lat[0]) | (lat > grid.lat[-1])
    )
    return pd.DataFrame(out)[list(CONFIG.seismic_columns)], outside


# -- table -----------------------------------------------------------------


def _outside_bbox(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    b = CONFIG.bounding_box
    return (lat < b.lat_min) | (lat > b.lat_max) | (lon < b.lon_min) | (lon > b.lon_max)


def build_source_table(buildings: pd.DataFrame, grid: Grid) -> tuple[pd.DataFrame, dict]:
    """Attach the sampled seismic columns; keep every row and count the awkward ones."""
    lon, lat = buildings["lon"].to_numpy(), buildings["lat"].to_numpy()
    seismic, outside = sample(grid, lon, lat)
    out = buildings.reset_index(drop=True).copy()
    for column in CONFIG.seismic_columns:
        out[column] = seismic[column].to_numpy()
    out["grid_outside"] = outside.astype("int8")
    out["bbox_outside"] = _outside_bbox(lat, lon).astype("int8")
    out = out[[f.name for f in SCHEMA]]
    checks = {
        "n_rows": len(out),
        "n_outside_bbox": int(out["bbox_outside"].sum()),
        "n_outside_grid": int(out["grid_outside"].sum()),
        "n_nonpositive": {
            c: int((out[c] <= 0).sum()) for c in CONFIG.seismic_columns if out[c].notna().any()
        },
        "units": dict(grid.units),
    }
    if checks["n_outside_grid"]:
        log.warning("%d buildings fall outside the ShakeMap grid", checks["n_outside_grid"])
    return out, checks


def source_table_path() -> Path:
    return paths.derived_dir() / "source.parquet"


def write_source_table(table: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or source_table_path()
    write_parquet_deterministic(table, path, schema=SCHEMA, sort_by=["building_id"])
    return path


def gpkg_path(raw: Path | None = None) -> Path:
    return (raw or paths.raw_dir()) / CONFIG.buildings.filename
