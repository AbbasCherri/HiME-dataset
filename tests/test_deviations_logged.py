"""Checks DATASET.md's Deviations table against config.py and the project's
own decisions -- never against the spec document, which isn't part of this
repository.
"""

from _fsscan import repo_root
from notodata import config

# Numbers the brief leaves open, decided by this project and logged here
# before first use (plus the dataset-source change).
_REQUIRED_SNIPPETS = [
    config.CONFIG.hf_source.dataset_id,  # "AbbasABC/HiME-dataset"
    config.CONFIG.hf_source.revision,  # "PENDING-TASK-2"
    "AbbasABC/HFL-Dataset",  # the old, overridden dataset id
    "6371008.8",  # Earth radius, IUGG mean, in metres
    "50",
    "100",
    "200",  # client counts N
    "1000-1004",
    "2000-2004",
    "3000-3019",
    "4000-4019",  # disjoint seed ranges
    "default_rng([s, client_id])",  # calibration draw
    "2.0",  # blank-chip standard-deviation threshold (grey levels)
    "3 levels",  # blank-chip near-black threshold
    "8 concurrent",
    "10 requests",  # GSI request limits
    "resnet50.tv_in1k_v2",
    "dinov2_vits14",
    "openclip_vitb32_laion2b",
    "satlas_aerial_swinb_si",  # candidate encoders
    "2026-10-23",
]


def test_deviations_table_logs_every_non_brief_number():
    text = (repo_root() / "DATASET.md").read_text()
    missing = [snippet for snippet in _REQUIRED_SNIPPETS if snippet not in text]
    assert not missing, f"DATASET.md is missing: {missing}"


def test_deviations_table_accepts_the_pending_revision_literal():
    # Ruling: the revision is resolved once the dataset is inspected; until
    # then this literal placeholder is what config.py and DATASET.md agree on.
    assert config.CONFIG.hf_source.revision == "PENDING-TASK-2"
