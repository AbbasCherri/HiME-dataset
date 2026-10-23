"""Single source of truth for every pinned constant in the build plan.

No other module may hard-code a value that appears here: import it from
`notodata.config` instead. Every dataclass is frozen, so a stray attempt to
mutate a field raises `dataclasses.FrozenInstanceError` right at the call
site that tried it, rather than silently drifting from the pinned value.
"""

from __future__ import annotations

from dataclasses import dataclass

PIPELINE_VERSION: int = 1
"""Bumped by hand whenever chip maths or another cache-defining step changes."""

CODE99_MEANING: str = "unknown"
"""One of `no_longer_present`, `missing_inconsistent` or `unknown`.

Default `unknown` until a named source (the dataset card, or a document it
cites, quoted in DATASET.md) states the meaning of raw label code 99 in
words that can be quoted. A code comment, a column name or the old
repository do not count as that source.
"""


@dataclass(frozen=True)
class HFSource:
    dataset_id: str
    revision: str


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
class AuditConfig:
    n_collapsed: int
    n_survived: int
    seed: int
    decision_threshold: int
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
    hf_source: HFSource
    bounding_box: BoundingBox
    seismic_columns: tuple[str, ...]
    label_column: str
    label_fallback_column: str
    old_class_counts: OldClassCounts
    baseline_imagery: BaselineImagery
    tile_chip: TileChipConfig
    audit: AuditConfig
    coverage_threshold: float
    moran: MoranConfig
    folds: FoldConfig
    clients: ClientBuildConfig
    test_seeds: tuple[int, ...]
    d1_test_stub: D1TestStub


CONFIG = Config(
    hf_source=HFSource(
        # Overridden: the brief's original pin (`AbbasABC/HFL-Dataset` @
        # 6cf97c900445e080e61cb45e1aa72515d3ff1de8) belongs to a different
        # repository. The new revision is resolved once the dataset is
        # inspected; "PENDING-TASK-2" is a literal placeholder, logged in
        # DATASET.md's Deviations table, and is never used to fetch anything
        # before it is replaced with the real pin.
        dataset_id="AbbasABC/HiME-dataset",
        revision="PENDING-TASK-2",
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
    audit=AuditConfig(n_collapsed=20, n_survived=20, seed=0, decision_threshold=10),
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
