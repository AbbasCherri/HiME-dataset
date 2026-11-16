import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import StubFetcher
from notodata import cache, chips, embeddings, paths
from notodata.config import CONFIG

LAYER = "seamlessphoto"


def _buildings(n=5):
    rng = np.random.default_rng(7)
    bb = CONFIG.bounding_box
    return pd.DataFrame(
        {
            "building_id": np.arange(n, dtype=np.int64) + 40,
            "lat": rng.uniform(bb.lat_min, bb.lat_max, n),
            "lon": rng.uniform(bb.lon_min, bb.lon_max, n),
        }
    )


@pytest.fixture
def index(tmp_home):
    """Real chip index from the stub fetcher; building 42 is then made `missing`."""
    idx = chips.build_chips(StubFetcher(), LAYER, _buildings())
    assert (idx["status"] == "ok").all()
    gone = idx["building_id"] == 42
    paths.notodata_home().joinpath(idx.loc[gone, "path"].iloc[0]).unlink()
    idx.loc[gone, ["status", "path", "sha256"]] = ["missing", "", ""]
    return idx


def _build(index, enc=None, **kw):
    return embeddings.ensure_embeddings(enc or embeddings.MeanColourEncoder(), LAYER, index, **kw)


def _png(index, bid):
    return paths.notodata_home() / index.loc[index["building_id"] == bid, "path"].iloc[0]


def test_mean_colour_encoder():
    enc = embeddings.MeanColourEncoder()
    assert enc.dim == 3
    batch = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    batch[0, :, :, 0] = 255
    batch[1, :2, :, 1] = 255
    out = enc.embed(batch)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1.0, 0.0, 0.0]
    assert out[1].tolist() == [0.0, 0.5, 0.0]


def test_rows_for_ok_chips_only_and_absent_is_none(index):
    assert _build(index) is True
    store = embeddings.load("mean_colour", LAYER)
    ok_ids = sorted(index.loc[index["status"] == "ok", "building_id"])
    assert store.ids.tolist() == ok_ids
    assert store.vectors.shape == (len(ok_ids), 3)
    assert store.get(42) is None
    assert store.get(99999) is None
    got = store.get(41)
    pixels = np.asarray(Image.open(_png(index, 41)).convert("RGB"))
    assert np.allclose(got, pixels.reshape(-1, 3).mean(axis=0) / 255, atol=1e-6)
    many = store.lookup([41, 42, 40])
    assert many[1] is None
    assert np.array_equal(many[0], got)


def test_sidecar_keys_and_second_build_skips(index):
    assert _build(index) is True
    assert _build(index) is False
    path = embeddings.embedding_path("mean_colour", LAYER)
    assert path == paths.embeddings_dir("mean_colour") / f"{LAYER}.npy"
    meta = json.loads(cache.sidecar_path(path).read_text())
    assert meta["encoder_id"] == "mean_colour"
    assert meta["deterministic"] is True
    assert meta["row_count"] == 4
    assert meta["chip_index_sha256"] == embeddings.chip_index_sha256(index)
    assert meta["building_ids_sha256"] == cache.building_ids_sha256([40, 41, 43, 44])
    assert meta["layer_ids"] == [LAYER]


def test_embeddings_follow_chips(index):
    """Changing one chip's pixels invalidates the cache even at the same row count."""
    _build(index)
    before = embeddings.load("mean_colour", LAYER).get(43)
    png = _png(index, 43)
    new = np.full((128, 128, 3), 200, dtype=np.uint8)
    Image.fromarray(new, "RGB").save(png)
    index = index.copy()
    index.loc[index["building_id"] == 43, "sha256"] = chips.pixel_sha256(new)
    assert _build(index) is True
    after = embeddings.load("mean_colour", LAYER).get(43)
    assert not np.allclose(before, after)
    assert np.allclose(after, 200 / 255)
    assert _build(index) is False


def test_status_change_changes_rows(index):
    _build(index)
    # a previously missing chip becoming ok must add a row
    idx2 = chips.build_chips(StubFetcher(), LAYER, _buildings())
    assert _build(idx2) is True
    assert embeddings.load("mean_colour", LAYER).get(42) is not None


def test_stale_pixels_detected(index):
    png = _png(index, 41)
    Image.fromarray(np.full((128, 128, 3), 9, dtype=np.uint8), "RGB").save(png)
    with pytest.raises(ValueError, match="41"):
        _build(index)


def test_missing_ids_file_recomputes(index):
    _build(index)
    embeddings.ids_path("mean_colour", LAYER).unlink()
    assert _build(index) is True
    assert embeddings.ids_path("mean_colour", LAYER).exists()


def test_nondeterministic_encoder_is_flagged(index):
    class Gpu(embeddings.MeanColourEncoder):
        encoder_id = "gpu_stub"
        deterministic = False

    _build(index, Gpu())
    path = embeddings.embedding_path("gpu_stub", LAYER)
    assert json.loads(cache.sidecar_path(path).read_text())["deterministic"] is False


def test_batches_do_not_change_result(index):
    _build(index, batch_size=1)
    one = embeddings.load("mean_colour", LAYER).vectors.copy()
    embeddings.embedding_path("mean_colour", LAYER).unlink()
    _build(index, batch_size=64)
    assert np.array_equal(one, embeddings.load("mean_colour", LAYER).vectors)


def test_real_encoders_are_registered_stubs():
    assert {
        "resnet50.tv_in1k_v2",
        "dinov2_vits14",
        "openclip_vitb32_laion2b",
        "satlas_aerial_swinb_si",
    } <= set(embeddings.REGISTRY)
    assert embeddings.REGISTRY["resnet50.tv_in1k_v2"].dim == 2048
    assert embeddings.REGISTRY["dinov2_vits14"].dim == 384
    assert embeddings.REGISTRY["openclip_vitb32_laion2b"].dim == 512
    for spec in embeddings.REGISTRY.values():
        assert spec.licence_note
    assert isinstance(embeddings.get_encoder("mean_colour"), embeddings.MeanColourEncoder)
    with pytest.raises(NotImplementedError, match="Task 27"):
        embeddings.get_encoder("dinov2_vits14")
    with pytest.raises(KeyError):
        embeddings.get_encoder("nope")
