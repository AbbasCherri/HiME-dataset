"""Publish the finished dataset to the Hugging Face dataset repo.

Only this module imports `huggingface_hub`, and only when a real upload runs.
The upload set is explicit and passes through `is_publishable`, which refuses
anything under an `audit` directory and any file called `key.csv`: the audit key
unblinds the audit sample and must never leave the machine.

Repo paths are the manifest names: relative to `NOTODATA_HOME` for data files
(`derived/...`, `embeddings/...`, optionally `chips/...` and `tiles/...`), and
`reports/...`, `MANIFEST.sha256` and `README.md` at the top level.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

from notodata import paths
from notodata.config import (
    CARD_LICENCE,
    CARD_PRETTY_NAME,
    CONFIG,
    HF_DATASET_REPO_ID,
    HF_REPO_PRIVATE_ON_CREATE,
)

MANIFEST_NAME = "MANIFEST.sha256"
CARD_NAME = "README.md"
VERIFY_RESULT = "verify_result.json"
IMAGERY_DIRS = ("chips", "tiles")
_NEVER = ("audit/**", "**/key.csv")
_TEMP = ("*.tmp",)
_SIZE_BANDS = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
)


class Item(NamedTuple):
    """One file to upload: its name in the repo, size in bytes and local path."""

    rel: str
    size: int
    path: Path


class ImageryPart(NamedTuple):
    """A whole imagery directory, uploaded as a folder: file count and total bytes."""

    rel: str
    n_files: int
    size: int


def is_publishable(rel: str) -> bool:
    """False for anything under an `audit` directory and for any `key.csv`."""
    parts = rel.split("/")
    return "audit" not in parts and parts[-1] != "key.csv"


def _items(base: Path, rels: Iterable[Path], prefix: str = "") -> list[Item]:
    out = []
    for p in rels:
        rel = prefix + p.relative_to(base).as_posix()
        if p.is_file() and is_publishable(rel):
            out.append(Item(rel, p.stat().st_size, p))
    return out


def small_files(root: Path) -> list[Item]:
    """The small artifacts, sorted by repo path (the dataset card is added by `plan`)."""
    home = paths.notodata_home()
    derived, reports = paths.derived_dir(), paths.reports_dir(root)
    found = _items(home, [paths.buildings_table_path(), derived / "folds.parquet"])
    for sub in ("splits", "clients"):
        found += _items(home, sorted((derived / sub).glob("*")))
    found += _items(home, sorted((home / "embeddings").rglob("*")))
    found += _items(root, [root / MANIFEST_NAME])
    for pattern in ("*.csv", "*.json"):
        found += _items(root, sorted(reports.glob(pattern)))
    return sorted(found)


def plan(root: Path) -> list[Item]:
    """Every file `publish` uploads as a commit, the generated card included."""
    card = build_card(root)
    return sorted([*small_files(root), Item(CARD_NAME, len(card.encode()), Path(CARD_NAME))])


def _imagery_part(name: str) -> ImageryPart | None:
    files = [p for p in (paths.notodata_home() / name).rglob("*") if p.is_file()]
    files = [p for p in files if p.suffix != ".tmp"]
    if not files:
        return None
    return ImageryPart(name, len(files), sum(p.stat().st_size for p in files))


def imagery_parts(include: bool) -> list[ImageryPart]:
    """The imagery directories that exist under `NOTODATA_HOME` (empty unless `include`)."""
    found = [_imagery_part(name) for name in IMAGERY_DIRS] if include else []
    return [part for part in found if part is not None]


def gate(root: Path) -> tuple[str, str] | None:
    """None when the release may be published, else (`missing` | `failed`, message)."""
    result = paths.reports_dir(root) / VERIFY_RESULT
    manifest = root / MANIFEST_NAME
    hint = "run `notodata verify` and fix any difference first"
    if not result.exists():
        return "missing", f"{result} is missing; {hint}"
    if not manifest.exists():
        return "missing", f"{manifest} is missing; run `notodata manifest`, then verify"
    try:
        status = json.loads(result.read_text()).get("status")
    except (ValueError, AttributeError):
        return "failed", f"{result} is not a verify result; {hint}"
    if status != "pass":
        return "failed", f"{result} does not say pass; {hint}"
    if result.stat().st_mtime < manifest.stat().st_mtime:
        return "failed", f"{result} is older than {manifest}; {hint}"
    return None


def _size_category(root: Path) -> list[str]:
    decision = paths.reports_dir(root) / "imagery_decision.json"
    if not decision.exists():
        return []
    n = int(json.loads(decision.read_text())["n_kept"])
    return [label for limit, label in _SIZE_BANDS if n < limit][:1]


def _front_matter(root: Path) -> str:
    lines = [
        "---",
        f"license: {CARD_LICENCE}",
        "language:",
        "- en",
        f"pretty_name: {CARD_PRETTY_NAME}",
        "task_categories:",
        "- image-classification",
        "tags:",
        "- earthquake",
        "- building-damage",
        "- remote-sensing",
    ]
    cats = _size_category(root)
    if cats:
        lines += ["size_categories:", f"- {cats[0]}"]
    return "\n".join([*lines, "---", ""])


def _files_section() -> str:
    return "\n".join(
        [
            "## Files",
            "",
            "- `derived/buildings.parquet`: the building table",
            "- `derived/folds.parquet`: spatial blocks and folds",
            "- `derived/splits/`: per-seed roles",
            "- `derived/clients/`: per-(N, seed) client assignments and metadata",
            "- `embeddings/<encoder>/`: chip embeddings (`<layer>.npy`, `<layer>.ids.npy` and"
            " JSON sidecars)",
            "- `MANIFEST.sha256`: SHA-256 of every derived file, `sha256sum` format",
            "- `reports/`: CSV and JSON reports, including `verify_result.json`",
            "- `chips/` and `tiles/`: the GSI imagery (`chips/` crops, `tiles/` source tiles)",
            "",
        ]
    )


def _sources_section() -> str:
    b, s = CONFIG.buildings, CONFIG.shakemap
    return "\n".join(
        [
            "## Sources and licences",
            "",
            f"- Zenodo `{b.concept_doi}`, `{b.filename}` ({b.licence}); Vescovo et al. 2025,"
            f" doi:{b.paper_doi}",
            f"- USGS ShakeMap `{s.event_id}`, version {s.version} ({s.licence})",
            "- Imagery: Geospatial Information Authority of Japan (GSI) tiles, Public Data"
            " License 1.0 (compatible with CC-BY-4.0). Source: GSI website"
            " (https://maps.gsi.go.jp/). Chips were created by editing GSI Tiles; the full"
            " attribution is in DATASET.md, section 10",
            "",
        ]
    )


def build_card(root: Path) -> str:
    """The dataset card: YAML front matter, a short Files section and DATASET.md embedded."""
    dataset_md = (root / "DATASET.md").read_text()
    body = [
        f"# {CARD_PRETTY_NAME}",
        "",
        "Building-level collapse table, spatial folds, per-seed splits, client assignments and"
        " chip embeddings, built by the `notodata` pipeline. DATASET.md, embedded below, is the"
        " full documentation.",
        "",
        _files_section(),
        _sources_section(),
        "## DATASET.md",
        "",
        dataset_md,
    ]
    return _front_matter(root) + "\n" + "\n".join(body)


def default_api() -> Any:
    """`HfApi()`, which reads the token from the `HF_TOKEN` environment variable itself."""
    from huggingface_hub import HfApi

    return HfApi()


def _commit(api: Any, root: Path, items: list[Item]) -> None:
    from huggingface_hub import CommitOperationAdd

    card = build_card(root).encode()
    ops = [
        CommitOperationAdd(path_in_repo=CARD_NAME, path_or_fileobj=card)
        if i.rel == CARD_NAME
        else CommitOperationAdd(path_in_repo=i.rel, path_or_fileobj=str(i.path))
        for i in items
    ]
    api.create_commit(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="Publish the HiME Noto dataset",
    )


def upload(
    root: Path,
    items: list[Item],
    include_imagery: bool,
    api_factory: Callable[[], Any] | None = None,
) -> None:
    """Create the repo if absent (private), commit `items`, then the imagery folders if asked."""
    api = (api_factory or default_api)()
    api.create_repo(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        private=HF_REPO_PRIVATE_ON_CREATE,
        exist_ok=True,
    )
    _commit(api, root, items)
    if include_imagery:
        api.upload_large_folder(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            folder_path=str(paths.notodata_home()),
            allow_patterns=[f"{name}/**" for name in IMAGERY_DIRS],
            ignore_patterns=[*_NEVER, *_TEMP],
        )
