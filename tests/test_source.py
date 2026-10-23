"""Phase 1 source: gpkg MD5, ShakeMap grid parsing, bilinear sampling, units.

Everything here uses a tiny synthetic gpkg and grid.xml; nothing touches the network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from notodata import source
from notodata.config import CONFIG

LON0, LAT0, STEP = 136.0, 37.0, 0.5
NLON, NLAT = 4, 3

# field -> (header name, units, a, b, c, d) with value = a + b*u + c*v + d*u*v, u,v in grid cells
_FIELDS = {
    "MMI": ("intensity", 5.0, 1.0, 0.5, 0.25),
    "PGA": ("%g", 10.0, 20.0, 5.0, 2.0),
    "PGV": ("cm/s", 30.0, 4.0, 8.0, 1.0),
    "PSA03": ("%g", 20.0, 10.0, 3.0, 1.0),
    "PSA10": ("%g", 15.0, 6.0, 2.0, 0.5),
    "PSA30": ("%g", 8.0, 3.0, 1.0, 0.25),
}
_ORDER = ["MMI", "PGA", "PGV", "PSA03", "PSA10", "PSA30"]


def _value(field: str, u: float, v: float) -> float:
    _, a, b, c, d = _FIELDS[field]
    return a + b * u + c * v + d * u * v


def make_grid_xml(
    version: str = "10", event: str = "us6000m0xl", units: dict | None = None
) -> bytes:
    units = {k: v[0] for k, v in _FIELDS.items()} | (units or {})
    fields = "".join(
        f'<grid_field index="{i}" name="{n}" units="{u}"/>\n'
        for i, (n, u) in enumerate(
            [("LON", "dd"), ("LAT", "dd"), *[(f, units[f]) for f in _ORDER], ("SVEL", "m/s")], 1
        )
    )
    rows = []
    for j in range(NLAT - 1, -1, -1):  # real grids run north to south
        for i in range(NLON):
            vals = " ".join(f"{_value(f, i, j):.6f}" for f in _ORDER)
            rows.append(f"{LON0 + i * STEP:.4f} {LAT0 + j * STEP:.4f} {vals} 600.0")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<shakemap_grid xmlns="http://earthquake.usgs.gov/eqcenter/shakemap" '
        f'event_id="{event}" shakemap_id="{event}" shakemap_version="{version}" '
        'shakemap_originator="us">\n'
        f'<grid_specification lon_min="{LON0}" lat_min="{LAT0}" '
        f'lon_max="{LON0 + (NLON - 1) * STEP}" lat_max="{LAT0 + (NLAT - 1) * STEP}" '
        f'nominal_lon_spacing="{STEP}" nominal_lat_spacing="{STEP}" '
        f'nlon="{NLON}" nlat="{NLAT}"/>\n' + fields + "<grid_data>\n" + "\n".join(rows) + "\n"
        "</grid_data>\n</shakemap_grid>\n"
    ).encode()


def make_gpkg(path, points, damage_val=None, damage_2=None):
    n = len(points)
    gdf = gpd.GeoDataFrame(
        {
            "damage_val": damage_val if damage_val is not None else [0] * n,
            "damage_2": damage_2 if damage_2 is not None else [0] * n,
            "municipality": ["m"] * n,
        },
        geometry=[Point(x, y).buffer(1e-5) for x, y in points],
        crs="EPSG:4326",
    )
    gdf.to_file(path, layer="v2.5", driver="GPKG", engine="pyogrio")
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- MD5 ------------------------------------------------------------------


def test_md5_mismatch_aborts(tmp_path):
    path = tmp_path / "b.gpkg"
    make_gpkg(path, [(136.2, 37.2)])
    with pytest.raises(source.SourceError, match="MD5"):
        source.read_buildings(path, expected_md5="0" * 32)


def test_md5_match_reads_buildings(tmp_path):
    path = tmp_path / "b.gpkg"
    md5 = make_gpkg(path, [(136.2, 37.2), (136.4, 37.4)], damage_val=[1, 0], damage_2=[1, 0])
    df = source.read_buildings(path, expected_md5=md5)
    assert len(df) == 2
    assert df["building_id"].is_unique
    assert df["lon"].iloc[0] == pytest.approx(136.2, abs=1e-6)
    assert df["lat"].iloc[1] == pytest.approx(37.4, abs=1e-6)


def test_labels_are_normalised_to_float_columns(tmp_path):
    path = tmp_path / "b.gpkg"
    md5 = make_gpkg(path, [(136.2, 37.2), (136.4, 37.4)], damage_val=[99, 1], damage_2=[99, 1])
    df = source.read_buildings(path, expected_md5=md5)
    assert df["damage_val"].dtype == np.float64
    assert df["label"].dtype == np.float64
    assert df["damage_val"].tolist() == [99.0, 1.0]
    assert df["label"].tolist() == [99.0, 1.0]


def test_non_finite_coordinates_abort(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"damage_val": [0, 0], "damage_2": [0, 0], "municipality": ["m", "m"]},
        geometry=[Point(136.2, 37.2).buffer(1e-5), None],
        crs="EPSG:4326",
    )
    path = tmp_path / "b.gpkg"
    gdf.to_file(path, layer="v2.5", driver="GPKG", engine="pyogrio")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    with pytest.raises(source.SourceError, match="non-finite"):
        source.read_buildings(path, expected_md5=md5)


# --- grid parsing ---------------------------------------------------------


def test_parse_grid_reads_fields_and_units():
    grid = source.parse_grid(make_grid_xml())
    assert grid.units["PGA"] == "%g"
    assert grid.units["PGV"] == "cm/s"
    assert grid.units["MMI"] == "intensity"
    assert grid.lon[0] == pytest.approx(LON0)
    assert grid.lat[-1] == pytest.approx(LAT0 + (NLAT - 1) * STEP)
    assert grid.fields["PGA"].shape == (NLAT, NLON)
    # north-to-south data order must not matter: row 0 is the southernmost latitude
    assert grid.fields["PGA"][2, 3] == pytest.approx(_value("PGA", 3, 2))
    assert grid.fields["PGA"][0, 1] == pytest.approx(_value("PGA", 1, 0))


def test_parse_grid_pins_event_and_version():
    with pytest.raises(source.SourceError, match="version"):
        source.parse_grid(make_grid_xml(version="9"))
    with pytest.raises(source.SourceError, match="event"):
        source.parse_grid(make_grid_xml(event="us0000000"))


def test_parse_grid_rejects_unknown_units():
    with pytest.raises(source.SourceError, match="units"):
        source.parse_grid(make_grid_xml(units={"PGA": "furlongs"}))


# --- sampling -------------------------------------------------------------


def _sample(lon, lat):
    grid = source.parse_grid(make_grid_xml())
    return source.sample(grid, np.asarray(lon, float), np.asarray(lat, float))


def test_bilinear_is_exact_at_nodes_and_inside_cells():
    lon = [LON0 + 1 * STEP, LON0 + 1.25 * STEP, LON0 + 2.5 * STEP]
    lat = [LAT0 + 1 * STEP, LAT0 + 0.5 * STEP, LAT0 + 1.75 * STEP]
    values, outside = _sample(lon, lat)
    assert not outside.any()
    us, vs = [1.0, 1.25, 2.5], [1.0, 0.5, 1.75]
    for k, (u, v) in enumerate(zip(us, vs, strict=True)):
        assert values["MMI_original"][k] == pytest.approx(_value("MMI", u, v))
        assert values["PGV"][k] == pytest.approx(_value("PGV", u, v))


def test_percent_g_is_converted_to_g_from_the_header_units():
    values, _ = _sample([LON0 + STEP], [LAT0 + STEP])
    assert values["PGA"][0] == pytest.approx(_value("PGA", 1, 1) / 100.0)
    assert values["SA_0_3"][0] == pytest.approx(_value("PSA03", 1, 1) / 100.0)
    assert values["SA_1_0"][0] == pytest.approx(_value("PSA10", 1, 1) / 100.0)
    assert values["SA_3_0"][0] == pytest.approx(_value("PSA30", 1, 1) / 100.0)


def test_conversion_follows_the_declared_units_not_a_fixed_factor():
    grid = source.parse_grid(make_grid_xml(units={"PGA": "g"}))
    values, _ = source.sample(grid, np.array([LON0 + STEP]), np.array([LAT0 + STEP]))
    assert values["PGA"][0] == pytest.approx(_value("PGA", 1, 1))  # already g: factor 1
    assert values["SA_0_3"][0] == pytest.approx(_value("PSA03", 1, 1) / 100.0)


def test_pgv_and_mmi_keep_native_units():
    values, _ = _sample([LON0 + STEP], [LAT0 + STEP])
    assert values["PGV"][0] == pytest.approx(_value("PGV", 1, 1))
    assert values["MMI_original"][0] == pytest.approx(_value("MMI", 1, 1))


def test_outside_points_are_flagged_not_clamped():
    values, outside = _sample([LON0 - 0.1, LON0 + STEP, LON0 + 1.6], [LAT0 + STEP] * 3)
    assert outside.tolist() == [True, False, True]
    assert np.isnan(values["PGA"][0])
    assert np.isnan(values["MMI_original"][2])
    assert np.isfinite(values["PGA"][1])


def test_mmi_shape_is_an_all_nan_float_column():
    values, _ = _sample([LON0 + STEP, LON0 + 2 * STEP], [LAT0 + STEP] * 2)
    assert values["MMI_shape"].dtype == np.float64
    assert np.isnan(values["MMI_shape"]).all()
    assert list(values.columns) == list(CONFIG.seismic_columns)


# --- table + download ------------------------------------------------------


def test_build_table_counts_and_keeps_out_of_box_rows(tmp_path):
    path = tmp_path / "b.gpkg"
    pts = [(136.2, 37.2), (136.4, 37.4), (140.0, 40.0)]  # last: outside the grid and the box
    md5 = make_gpkg(path, pts, damage_val=[0, 1, 0], damage_2=[0, 1, 0])
    buildings = source.read_buildings(path, expected_md5=md5)
    table, checks = source.build_source_table(buildings, source.parse_grid(make_grid_xml()))
    assert len(table) == 3
    assert checks["n_rows"] == 3
    assert checks["n_outside_grid"] == 1
    assert checks["n_outside_bbox"] == 3  # the synthetic grid sits far west of the Noto box
    assert table["grid_outside"].tolist() == [0, 0, 1]
    assert checks["n_nonpositive"]["PGA"] == 0
    assert table["MMI_shape"].isna().all()


def test_fetch_grid_downloads_once_caches_and_records_sha256(tmp_path):
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return make_grid_xml()

    p1 = source.fetch_grid(tmp_path, get=fake_get)
    p2 = source.fetch_grid(tmp_path, get=fake_get)
    assert p1 == p2 == tmp_path / CONFIG.shakemap.grid_file
    assert len(calls) == 1
    assert calls[0] == (f"{CONFIG.shakemap.base_url}/download/{CONFIG.shakemap.grid_file}")
    record = source.read_source_files(tmp_path)
    assert (
        record[CONFIG.shakemap.grid_file]["sha256"] == hashlib.sha256(p1.read_bytes()).hexdigest()
    )


def test_source_never_uses_hf_loaders():
    text = Path(source.__file__).read_text()
    assert "load_dataset" not in text
    assert "snapshot_download" not in text


def test_source_table_frame_has_the_expected_columns(tmp_path):
    path = tmp_path / "b.gpkg"
    md5 = make_gpkg(path, [(136.2, 37.2)])
    buildings = source.read_buildings(path, expected_md5=md5)
    table, _ = source.build_source_table(buildings, source.parse_grid(make_grid_xml()))
    assert isinstance(table, pd.DataFrame)
    expected = [f.name for f in source.SCHEMA]
    assert list(table.columns) == expected
