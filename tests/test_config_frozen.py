import dataclasses

import pytest

from notodata import config


def test_building_source():
    b = config.CONFIG.buildings
    assert b.concept_doi == "10.5281/zenodo.11055711"
    assert b.filename == "Noto_Peninsula_Damage_2_5.gpkg"
    assert b.md5 == "280e9c53cb45086786c20ed0b2c21ffb"
    assert b.licence == "CC-BY-4.0"


def test_shakemap_source():
    m = config.CONFIG.shakemap
    assert (m.event_id, m.version) == ("us6000m0xl", 10)
    assert m.base_url.endswith("/us6000m0xl/us/1710009202676")


def test_bounding_box():
    bb = config.CONFIG.bounding_box
    assert (bb.lat_min, bb.lat_max, bb.lon_min, bb.lon_max) == (
        37.005,
        37.526,
        136.677,
        137.356,
    )


def test_seismic_columns():
    assert config.CONFIG.seismic_columns == (
        "MMI_original",
        "MMI_shape",
        "PGA",
        "PGV",
        "SA_0_3",
        "SA_1_0",
        "SA_3_0",
    )


def test_label_columns():
    assert config.CONFIG.label_column == "damage_val"
    assert config.CONFIG.label_fallback_column == "label"


def test_old_class_counts():
    c = config.CONFIG.old_class_counts
    assert (c.code_0, c.code_1, c.code_9, c.code_99, c.total) == (
        104_917,
        3_407,
        8_365,
        12_154,
        128_843,
    )


def test_baseline_imagery():
    b = config.CONFIG.baseline_imagery
    assert b.layer_id == "seamlessphoto"
    assert b.url_template == "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"


def test_tile_and_chip_sizes():
    t = config.CONFIG.tile_chip
    assert (t.zoom, t.tile_size_px, t.chip_size_px) == (18, 256, 128)


def test_audit_config():
    a = config.CONFIG.audit
    assert (a.n_collapsed, a.n_survived, a.seed, a.decision_threshold) == (20, 20, 0, 10)


def test_coverage_threshold():
    assert config.CONFIG.coverage_threshold == pytest.approx(0.80)


def test_moran_config():
    m = config.CONFIG.moran
    assert m.sample_size == 20_000
    assert m.seed == 0
    assert m.bin_edges_km == (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    assert m.threshold == pytest.approx(0.05)
    assert m.l_clamp_km == (2.0, 10.0)


def test_fold_config():
    f = config.CONFIG.folds
    assert f.n_folds == 5
    assert f.buffer_m == pytest.approx(500.0)


def test_client_build_config():
    c = config.CONFIG.clients
    assert c.kmeans_n_init == 10
    assert c.kmeans_random_state_base == 5309
    assert c.medoid_sample_max == 2000
    assert c.min_client_size == 20
    assert c.calibration_share == pytest.approx(0.20)


def test_test_seeds():
    assert config.CONFIG.test_seeds == (1000, 1001, 1002, 1003, 1004)


def test_d1_test_stub():
    d = config.CONFIG.d1_test_stub
    assert (d.mod_u, d.mod_v, d.channel_b, d.n_points, d.seed) == (251, 241, 0, 1000, 0)


def test_pipeline_version_and_code99_meaning():
    assert config.PIPELINE_VERSION == 1
    assert config.CODE99_MEANING == "missing_inconsistent"


@pytest.mark.parametrize(
    ("get_instance", "field_name"),
    [
        (lambda: config.CONFIG, "coverage_threshold"),
        (lambda: config.CONFIG.buildings, "md5"),
        (lambda: config.CONFIG.bounding_box, "lat_min"),
        (lambda: config.CONFIG.moran, "threshold"),
        (lambda: config.CONFIG.clients, "min_client_size"),
    ],
)
def test_assignment_raises(get_instance, field_name):
    instance = get_instance()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, object())


def test_blank_chip_thresholds():
    b = config.CONFIG.blank_chip
    assert (b.std_below, b.black_level, b.black_share) == (2.0, 3, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.std_below = 1.0  # type: ignore[misc]
