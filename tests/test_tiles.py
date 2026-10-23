import csv
import dataclasses
import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from notodata import tiles
from notodata.config import CONFIG

LAYER = "testlayer"
FAST = dataclasses.replace(
    CONFIG.fetch,
    backoff_base_s=0.0,
    throttle_backoff_base_s=0.0,
    max_requests_per_s=1000.0,
    timeout_s=2.0,
    precheck_zoom=0,  # off, so request counts below are exact
)


def _png(size=256, rgba=False) -> bytes:
    arr = np.full((size, size, 4 if rgba else 3), 120, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests: list[tuple[str, float]] = []
        self.responses: dict[str, list[int]] = {}  # path -> status queue; last repeats
        self.default = 200
        self.delay = 0.0
        self.active = 0
        self.max_active = 0
        self.body = _png()
        self.user_agents: list[str] = []


@pytest.fixture
def server():
    state = _State()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with state.lock:
                state.requests.append((self.path, time.monotonic()))
                state.user_agents.append(self.headers.get("User-Agent", ""))
                state.active += 1
                state.max_active = max(state.max_active, state.active)
                queue = state.responses.get(self.path)
                status = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else state.default
            if state.delay:
                time.sleep(state.delay)
            with state.lock:
                state.active -= 1
            body = state.body if status == 200 else b"x"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.template = f"http://127.0.0.1:{httpd.server_port}/{{z}}/{{x}}/{{y}}.png"
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def sleeps():
    return []


def _fetcher(state, tmp_home, sleeps=None, settings=FAST, template=None):
    return tiles.HttpTileFetcher(
        LAYER,
        template or state.template,
        "png",
        settings=settings,
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
    )


# ---- tile maths (D1) -------------------------------------------------------


def test_global_pixel_matches_brief_formula():
    lat, lon, z = 37.3, 136.9, 18
    gx, gy = tiles.global_pixel(lat, lon, z)
    ex = (lon + 180) / 360 * 2**z * 256
    phi = math.radians(lat)
    ey = (1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2 * 2**z * 256
    assert gx == pytest.approx(ex, abs=1e-6)
    assert gy == pytest.approx(ey, abs=1e-6)
    assert np.asarray(gx).dtype == np.float64


def test_global_pixel_known_tile():
    # lon 0 / equator sits at the centre of the world at any zoom.
    gx, gy = tiles.global_pixel(0.0, 0.0, 1)
    assert gx == pytest.approx(256.0)
    assert gy == pytest.approx(256.0)


def test_window_origin_and_covering_tiles():
    x0, y0 = tiles.window_origin(1000.5, 2000.25)
    assert (x0, y0) == (936, 1936)
    assert tiles.covering_tiles(800, 1800) == [(3, 7)]
    assert tiles.covering_tiles(896, 1800) == [(3, 7)]  # last pixel 1023
    # straddling a boundary in x, then in both
    assert tiles.covering_tiles(1000, 1800) == [(3, 7), (4, 7)]
    assert tiles.covering_tiles(1000, 1936) == [(3, 7), (4, 7), (3, 8), (4, 8)]


@pytest.mark.parametrize("u", [0, 1, 127, 128, 129, 255])
def test_covering_tiles_count_is_1_2_or_4(u):
    for v in (0, 100, 129, 255):
        assert len(tiles.covering_tiles(5 * 256 + u, 9 * 256 + v)) in (1, 2, 4)


def test_parent_tile():
    assert tiles.parent_tile(18, 232000, 104000, 14) == (232000 >> 4, 104000 >> 4)
    assert tiles.parent_tile(14, 5, 6, 14) == (5, 6)


# ---- cache layout ----------------------------------------------------------


def test_tile_cache_path_has_layer(server, tmp_home):
    f = _fetcher(server, tmp_home)
    arr = f.get(LAYER, 18, 10, 20)
    assert arr is not None
    assert arr.shape == (256, 256, 3)
    path = tmp_home / "tiles" / LAYER / "18" / "10" / "20.png"
    assert path.read_bytes() == server.body  # raw server bytes, never re-encoded


def test_cache_hit_does_not_rerequest(server, tmp_home):
    f = _fetcher(server, tmp_home)
    f.get(LAYER, 18, 1, 2)
    f.get(LAYER, 18, 1, 2)
    assert len(server.requests) == 1


def test_template_mismatch_raises(server, tmp_home):
    _fetcher(server, tmp_home)
    meta = json.loads((tmp_home / "tiles" / LAYER / "layer.meta.json").read_text())
    assert meta == {"url_template": server.template}
    with pytest.raises(tiles.TemplateMismatchError):
        _fetcher(server, tmp_home, template=server.template.replace("png", "jpg"))
    _fetcher(server, tmp_home)  # same template is fine


def test_wrong_layer_id_rejected(server, tmp_home):
    with pytest.raises(ValueError, match="layer"):
        _fetcher(server, tmp_home).get("other", 18, 1, 2)


def test_alpha_channel_preserved(server, tmp_home):
    server.body = _png(rgba=True)
    arr = _fetcher(server, tmp_home).get(LAYER, 18, 1, 2)
    assert arr.shape == (256, 256, 4)


def test_non_256_tile_raises(server, tmp_home):
    server.body = _png(size=255)
    with pytest.raises(tiles.TileShapeError):
        _fetcher(server, tmp_home).get(LAYER, 18, 1, 2)


# ---- HTTP behaviour --------------------------------------------------------


def test_404_cached_as_missing_and_not_rerequested(server, tmp_home):
    server.default = 404
    f = _fetcher(server, tmp_home)
    assert f.get(LAYER, 18, 3, 4) is None
    assert (tmp_home / "tiles" / LAYER / "18" / "3" / "4.missing").exists()
    assert f.get(LAYER, 18, 3, 4) is None
    # a fresh fetcher (new process) also honours the negative entry
    assert _fetcher(server, tmp_home).get(LAYER, 18, 3, 4) is None
    assert len(server.requests) == 1


def test_5xx_retries_5_times_with_backoff(server, tmp_home, sleeps):
    server.default = 500
    settings = dataclasses.replace(FAST, backoff_base_s=1.0)
    f = _fetcher(server, tmp_home, sleeps, settings)
    with pytest.raises(tiles.TileFetchError):
        f.get(LAYER, 18, 5, 6)
    assert len(server.requests) == 6  # first attempt + 5 retries
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert not (tmp_home / "tiles" / LAYER / "18" / "5" / "6.missing").exists()


def test_transient_5xx_then_success(server, tmp_home):
    server.responses["/18/7/8.png"] = [500, 502, 200]
    arr = _fetcher(server, tmp_home).get(LAYER, 18, 7, 8)
    assert arr is not None
    assert len(server.requests) == 3


@pytest.mark.parametrize("status", [429, 503])
def test_429_and_503_use_throttle_backoff(server, tmp_home, sleeps, status):
    server.responses["/18/1/1.png"] = [status, 200]
    settings = dataclasses.replace(FAST, backoff_base_s=1.0, throttle_backoff_base_s=5.0)
    assert _fetcher(server, tmp_home, sleeps, settings).get(LAYER, 18, 1, 1) is not None
    assert sleeps == [5.0]


def test_backoff_is_capped(server, tmp_home, sleeps):
    server.default = 429
    settings = dataclasses.replace(FAST, throttle_backoff_base_s=5.0, backoff_max_s=12.0)
    with pytest.raises(tiles.TileFetchError):
        _fetcher(server, tmp_home, sleeps, settings).get(LAYER, 18, 1, 1)
    assert sleeps == [5.0, 10.0, 12.0, 12.0, 12.0]


def test_other_4xx_fails_without_retry(server, tmp_home):
    server.default = 403
    with pytest.raises(tiles.TileFetchError):
        _fetcher(server, tmp_home).get(LAYER, 18, 1, 1)
    assert len(server.requests) == 1


def test_persistent_failure_listed_in_fetch_errors_csv(server, tmp_home):
    server.default = 500
    f = _fetcher(server, tmp_home)
    for y in (10, 11):
        with pytest.raises(tiles.TileFetchError):
            f.get(LAYER, 18, 9, y)
    rows = list(csv.DictReader((tmp_home / "logs" / "fetch_errors.csv").open()))
    assert [(r["layer_id"], r["z"], r["x"], r["y"]) for r in rows] == [
        (LAYER, "18", "9", "10"),
        (LAYER, "18", "9", "11"),
    ]
    assert rows[0]["error"]


def test_timeout_is_retried(server, tmp_home):
    server.delay = 0.5
    settings = dataclasses.replace(FAST, timeout_s=0.05, max_retries=2)
    with pytest.raises(tiles.TileFetchError):
        _fetcher(server, tmp_home, settings=settings).get(LAYER, 18, 1, 1)
    assert len(server.requests) == 3


def test_at_most_8_concurrent_requests(server, tmp_home):
    server.delay = 0.05
    f = _fetcher(server, tmp_home)
    results = f.fetch_many([(LAYER, 18, i, 0) for i in range(40)], workers=20)
    assert all(r is not None for r in results.values())
    assert 2 <= server.max_active <= 8
    assert CONFIG.fetch.max_concurrent == 8


def test_request_rate_is_limited(server, tmp_home):
    settings = dataclasses.replace(FAST, max_requests_per_s=40.0)
    f = _fetcher(server, tmp_home, settings=settings)
    f.fetch_many([(LAYER, 18, i, 0) for i in range(13)], workers=8)
    stamps = sorted(t for _, t in server.requests)
    assert stamps[-1] - stamps[0] >= 12 / 40.0 * 0.9
    assert CONFIG.fetch.max_requests_per_s == 10.0


def test_token_bucket_spacing_with_fake_clock():
    now = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    bucket = tiles.TokenBucket(10.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(5):
        bucket.acquire()
    assert sum(slept) == pytest.approx(0.4)
    assert all(s == pytest.approx(0.1) for s in slept[1:] if s)


def test_user_agent_is_set(server, tmp_home):
    _fetcher(server, tmp_home).get(LAYER, 18, 1, 1)
    assert server.user_agents == [CONFIG.fetch.user_agent]
    assert "python-requests" not in server.user_agents[0]


def test_token_never_logged(server, tmp_home, monkeypatch, caplog):
    secret = "hf_FAKE_secret_token_value_123"
    monkeypatch.setenv("HF_TOKEN", secret)
    caplog.set_level(logging.DEBUG)
    server.default = 500
    f = _fetcher(server, tmp_home)
    with pytest.raises(tiles.TileFetchError) as exc:
        f.get(LAYER, 18, 1, 1)
    server.default = 404
    f.get(LAYER, 18, 1, 2)
    assert secret not in caplog.text
    assert secret not in str(exc.value)
    assert secret not in (tmp_home / "logs" / "fetch_errors.csv").read_text()
    assert all(secret not in ua for ua in server.user_agents)
    for p in (tmp_home / "tiles").rglob("*.json"):
        assert secret not in p.read_text()


# ---- zoom-14 parent pre-check ---------------------------------------------


def test_z14_precheck_safe(server, tmp_home, tmp_path, monkeypatch):
    """With the pre-check, results equal an unconditional fetch, with fewer requests."""
    rng = np.random.default_rng(0)
    z, pz = 18, CONFIG.fetch.precheck_zoom
    shift = z - pz
    px = rng.integers(10000, 10010, 200)
    py = rng.integers(20000, 20010, 200)
    absent_parents = {
        (a, b) for a in range(10000, 10010) for b in range(20000, 20010) if (a + b) % 3 == 0
    }
    tiles_ = [
        (
            int(a) << shift | int(rng.integers(0, 1 << shift)),
            int(b) << shift | int(rng.integers(0, 1 << shift)),
        )
        for a, b in zip(px, py, strict=True)
    ]
    # consistent world: a z18 tile exists iff its parent exists
    for a, b in absent_parents:
        server.responses[f"/{pz}/{a}/{b}.png"] = [404]
    for x, y in tiles_:
        if tiles.parent_tile(z, x, y, pz) in absent_parents:
            server.responses[f"/{z}/{x}/{y}.png"] = [404]

    with_check = _fetcher(server, tmp_home, settings=dataclasses.replace(FAST, precheck_zoom=pz))
    got = [with_check.get(LAYER, z, x, y) is None for x, y in tiles_]
    expected = [tiles.parent_tile(z, x, y, pz) in absent_parents for x, y in tiles_]
    assert got == expected
    assert any(expected)
    assert not all(expected)

    z18_asked = {p for p, _ in server.requests if p.startswith(f"/{z}/")}
    for x, y in tiles_:
        if tiles.parent_tile(z, x, y, pz) in absent_parents:
            assert f"/{z}/{x}/{y}.png" not in z18_asked  # saved, never requested

    # same answers with the pre-check off
    off = FAST
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("NOTODATA_HOME", str(other))
    plain = _fetcher(server, other, settings=off)
    got_off = [plain.get(LAYER, z, x, y) is None for x, y in tiles_]
    assert got_off == got
