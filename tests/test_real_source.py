"""Checks that need the real Zenodo gpkg and the real ShakeMap grid (run by `make test`).

These fail, not skip, when the files are missing: they are part of the Definition
of Done. Task 18 runs them after `notodata source`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from notodata import paths, source
from notodata.config import CONFIG

pytestmark = pytest.mark.real

N_BUILDINGS = 140_208
"""Row count of the v2.5 layer (notes/phase0.md A.2)."""


def _need(path):
    assert path.exists(), f"{path} is missing; run `notodata source` first (see Task 18)"
    return path


def test_real_gpkg_matches_the_pinned_md5():
    path = _need(source.gpkg_path())
    assert source.md5_file(path) == CONFIG.buildings.md5


def test_real_grid_is_the_pinned_version_with_percent_g_headers():
    grid = source.parse_grid(_need(paths.raw_dir() / CONFIG.shakemap.grid_file).read_bytes())
    assert grid.units["PGA"] == "%g"
    assert grid.units["PGV"] == "cm/s"
    assert grid.fields["PGA"].shape == (324, 416)


def test_real_source_table_shape_and_label_columns():
    table = pd.read_parquet(_need(source.source_table_path()))
    assert len(table) == N_BUILDINGS
    assert set(table["damage_val"].dropna().unique()) <= {0.0, 1.0, 9.0, 99.0}
    both = table["damage_val"].notna() & table["label"].notna()
    conflict = (table.loc[both, "damage_val"] != table.loc[both, "label"]).mean()
    # damage_2 as the fallback column is an assumption about the shipped file (Task 18 reports it)
    assert conflict <= 0.01, f"damage_val and damage_2 disagree on {conflict:.1%} of rows"


def test_real_seismic_columns_are_in_g_and_inside_the_documented_ranges():
    table = pd.read_parquet(_need(source.source_table_path()))
    inside = table[table["bbox_outside"] == 0]
    assert inside["grid_outside"].sum() == 0
    # grid.xml maxima inside the box: PGA 166.9 %g -> 1.669 g, MMI 8.8 (notes/phase0.md A.3)
    assert 0.1 < inside["PGA"].max() <= 1.7
    assert inside["MMI_original"].max() <= 8.81
    assert inside["PGV"].max() <= 121.0
    assert (inside["SA_0_3"] > 0).all()


def test_real_mmi_shape_is_null():
    table = pd.read_parquet(_need(source.source_table_path()))
    assert table["MMI_shape"].dtype == np.float64
    assert table["MMI_shape"].isna().all()


def test_real_shakemap_download_url_and_pin_are_recorded():
    record = source.read_source_files()
    assert CONFIG.shakemap.grid_file in record, "grid.xml sha256 was not recorded"
    assert len(record[CONFIG.shakemap.grid_file]["sha256"]) == 64
    json.dumps(record)  # serialisable
