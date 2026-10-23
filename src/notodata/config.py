"""Single source of truth for every pinned constant in the build plan.

No other module may hard-code a value that appears here: import it from
`notodata.config` instead. Every dataclass is frozen, so a stray attempt to
mutate a field raises `dataclasses.FrozenInstanceError` right at the call
site that tried it, rather than silently drifting from the pinned value.
"""

from __future__ import annotations

from dataclasses import dataclass

PIPELINE_VERSION: int = 2
"""Bumped by hand whenever chip maths or another cache-defining step changes."""

EARTH_RADIUS_M: float = 6_371_008.8
"""IUGG mean Earth radius in metres, used by the equirectangular projection."""

CODE99_MEANING: str = "missing_inconsistent"
"""One of `no_longer_present`, `missing_inconsistent` or `unknown`.

Sourced from Vescovo et al. 2025 (ESSD 17, 5259-5276), section 2.2: code 9 is
"obstructed view through human inspection" and code 99 is a "missing or
inconsistent" footprint. Quoted in DATASET.md and notes/phase0.md.
"""

CLIENT_COUNTS: tuple[int, ...] = (50, 100, 200)
"""Client counts N built by default (DATASET.md deviation row 4)."""

GSI_XYZ_BASE: str = "https://cyberjapandata.gsi.go.jp/xyz"
"""Host and path prefix of every GSI tile layer: `<base>/<layer_id>/{z}/{x}/{y}.<ext>`."""

GSI_TILE_LIST_URL: str = "https://maps.gsi.go.jp/development/ichiran.html"
"""The GSI tile list page every layer in the registry was taken from."""

LABEL_CODES: frozenset[int] = frozenset({0, 1, 9, 99})
"""The raw damage codes a building may carry in `label_raw`."""

MAX_LABEL_CONFLICT_SHARE: float = 0.01
"""Abort the load if `damage_val` and `label` disagree on more than this share of rows."""


@dataclass(frozen=True)
class BuildingSource:
    """Per-building labels and footprints: the Zenodo ESSD GeoPackage."""

    concept_doi: str
    paper_doi: str
    filename: str
    md5: str
    licence: str


@dataclass(frozen=True)
class ShakeMapSource:
    """Seismic columns: sampled from a version-pinned USGS ShakeMap grid."""

    event_id: str
    version: int
    base_url: str
    grid_file: str
    licence: str


@dataclass(frozen=True)
class BoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class OldClassCounts:
    """The old repository's class counts, kept as a comparison figure only."""

    code_0: int
    code_1: int
    code_9: int
    code_99: int
    total: int


@dataclass(frozen=True)
class BaselineImagery:
    layer_id: str
    url_template: str


@dataclass(frozen=True)
class TileChipConfig:
    zoom: int
    tile_size_px: int
    chip_size_px: int


@dataclass(frozen=True)
class BlankChipConfig:
    """When a chip counts as `blank` (DATASET.md deviation row 7)."""

    std_below: float
    """Pixel standard deviation over all channels below this is blank."""
    black_level: int
    """A pixel is near-black when every channel is at most this."""
    black_share: float
    """Blank when at least this share of pixels is near-black."""


@dataclass(frozen=True)
class FetchConfig:
    """Network etiquette for the GSI tile server."""

    user_agent: str
    max_concurrent: int
    max_requests_per_s: float
    max_retries: int
    """Retries after the first attempt, so a failing tile is tried `max_retries + 1` times."""
    backoff_base_s: float
    """Delay before retry n is `backoff_base_s * 2**n` (timeouts and 5xx)."""
    throttle_backoff_base_s: float
    """Same, but used after a 429 or 503."""
    backoff_max_s: float
    timeout_s: float
    precheck_zoom: int
    """A 404 parent at this zoom marks every descendant tile missing, unrequested."""


@dataclass(frozen=True)
class AuditConfig:
    n_collapsed: int
    n_survived: int
    seed: int
    decision_threshold: int
    pre_reject_damage: int
    """Reject the A1 pre layer when this many audited collapsed buildings show damage in it."""
    """The "10 of 20" line in the decision rule."""


@dataclass(frozen=True)
class MoranConfig:
    sample_size: int
    seed: int
    bin_edges_km: tuple[float, ...]
    threshold: float
    l_clamp_km: tuple[float, float]


@dataclass(frozen=True)
class FoldConfig:
    n_folds: int
    buffer_m: float


@dataclass(frozen=True)
class ClientBuildConfig:
    kmeans_n_init: int
    kmeans_random_state_base: int
    medoid_sample_max: int
    min_client_size: int
    calibration_share: float


@dataclass(frozen=True)
class D1TestStub:
    """The exact arithmetic the brief's D1 regression test stub must use."""

    mod_u: int
    mod_v: int
    channel_b: int
    n_points: int
    seed: int


@dataclass(frozen=True)
class Config:
    buildings: BuildingSource
    shakemap: ShakeMapSource
    bounding_box: BoundingBox
    seismic_columns: tuple[str, ...]
    label_column: str
    label_fallback_column: str
    old_class_counts: OldClassCounts
    baseline_imagery: BaselineImagery
    tile_chip: TileChipConfig
    fetch: FetchConfig
    blank_chip: BlankChipConfig
    audit: AuditConfig
    coverage_threshold: float
    moran: MoranConfig
    folds: FoldConfig
    clients: ClientBuildConfig
    test_seeds: tuple[int, ...]
    d1_test_stub: D1TestStub


CONFIG = Config(
    buildings=BuildingSource(
        concept_doi="10.5281/zenodo.11055711",
        paper_doi="10.5194/essd-17-5259-2025",
        filename="Noto_Peninsula_Damage_2_5.gpkg",
        md5="280e9c53cb45086786c20ed0b2c21ffb",
        licence="CC-BY-4.0",
    ),
    shakemap=ShakeMapSource(
        event_id="us6000m0xl",
        version=10,
        base_url="https://earthquake.usgs.gov/product/shakemap/us6000m0xl/us/1710009202676",
        grid_file="grid.xml",
        licence="US Public Domain",
    ),
    bounding_box=BoundingBox(
        lat_min=37.005,
        lat_max=37.526,
        lon_min=136.677,
        lon_max=137.356,
    ),
    seismic_columns=(
        "MMI_original",
        "MMI_shape",
        "PGA",
        "PGV",
        "SA_0_3",
        "SA_1_0",
        "SA_3_0",
    ),
    label_column="damage_val",
    label_fallback_column="label",
    old_class_counts=OldClassCounts(
        code_0=104_917,
        code_1=3_407,
        code_9=8_365,
        code_99=12_154,
        total=128_843,
    ),
    baseline_imagery=BaselineImagery(
        layer_id="seamlessphoto",
        url_template="https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
    ),
    tile_chip=TileChipConfig(zoom=18, tile_size_px=256, chip_size_px=128),
    fetch=FetchConfig(
        user_agent="notodata/0.1 (Noto 2024 research dataset build; polite fetcher)",
        max_concurrent=8,
        max_requests_per_s=10.0,
        max_retries=5,
        backoff_base_s=1.0,
        throttle_backoff_base_s=5.0,
        backoff_max_s=60.0,
        timeout_s=30.0,
        precheck_zoom=14,
    ),
    blank_chip=BlankChipConfig(std_below=2.0, black_level=3, black_share=0.5),
    audit=AuditConfig(
        n_collapsed=20, n_survived=20, seed=0, decision_threshold=10, pre_reject_damage=3
    ),
    coverage_threshold=0.80,
    moran=MoranConfig(
        sample_size=20_000,
        seed=0,
        bin_edges_km=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0),
        threshold=0.05,
        l_clamp_km=(2.0, 10.0),
    ),
    folds=FoldConfig(n_folds=5, buffer_m=500.0),
    clients=ClientBuildConfig(
        kmeans_n_init=10,
        kmeans_random_state_base=5309,
        medoid_sample_max=2000,
        min_client_size=20,
        calibration_share=0.20,
    ),
    test_seeds=(1000, 1001, 1002, 1003, 1004),
    d1_test_stub=D1TestStub(mod_u=251, mod_v=241, channel_b=0, n_points=1000, seed=0),
)
