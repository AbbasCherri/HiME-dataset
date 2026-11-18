"""`notodata publish`: dry run by default, gated on a passing verify, never the audit key."""

from __future__ import annotations

import json

import pytest
import yaml

from notodata import cli, publish
from notodata.config import HF_DATASET_REPO_ID

SENTINEL = "sentinel-not-a-real-credential"


class FakeApi:
    """Records every call; never touches the network."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit", kwargs))

    def upload_large_folder(self, **kwargs):
        self.calls.append(("upload_large_folder", kwargs))

    def named(self, name: str) -> list[dict]:
        return [kw for n, kw in self.calls if n == name]

    def committed_paths(self) -> set[str]:
        return {op.path_in_repo for kw in self.named("create_commit") for op in kw["operations"]}


def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def release(tmp_home, tmp_path):
    """A finished-looking release: root with reports and manifest, plus a populated home."""
    root = tmp_path / "repo"
    reports = root / "reports"
    _touch(reports / "verify_result.json", json.dumps({"status": "pass"}))
    _touch(reports / "imagery_decision.json", json.dumps({"n_kept": 140208}))
    _touch(reports / "folds.csv", "a,b\n")
    _touch(reports / "publish.log", "log")
    _touch(reports / "run.jsonl", "log")
    _touch(reports / "audit" / "key.csv", "secret")
    _touch(root / "MANIFEST.sha256", "0" * 64 + "  derived/buildings.parquet\n")
    _touch(root / "DATASET.md", "# Noto\n\nbody text\n")
    for rel in (
        "derived/buildings.parquet",
        "derived/folds.parquet",
        "derived/labels.parquet",
        "derived/source.parquet",
        "derived/splits/seed_1000.parquet",
        "derived/clients/N50_seed_1000.parquet",
        "derived/clients/N50_seed_1000_meta.parquet",
        "embeddings/enc/seamlessphoto.npy",
        "embeddings/enc/seamlessphoto.ids.npy",
        "embeddings/enc/seamlessphoto.npy.meta.json",
        "audit/key.csv",
        "audit/marks.csv",
        "chips/seamlessphoto/1/1001.png",
        "tiles/seamlessphoto/18/1/2.jpg",
    ):
        _touch(tmp_home / rel)
    api = FakeApi()
    ctx = cli.Context(root=root, hf_api_factory=lambda: api)
    return ctx, api, root


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)


def run(ctx, *argv) -> int:
    return cli.main(["publish", *argv], ctx)


def test_dry_run_lists_files_and_total_and_uploads_nothing(release, capsys, token):
    ctx, api, _ = release
    assert run(ctx) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "derived/buildings.parquet" in out
    assert "derived/splits/seed_1000.parquet" in out
    assert "total" in out.lower()
    assert "dry run" in out.lower()
    assert api.calls == []


def test_dry_run_needs_no_token(release, monkeypatch):
    ctx, api, _ = release
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert run(ctx) == cli.EXIT_OK
    assert api.calls == []


def test_refuses_without_verify_result(release, capsys):
    ctx, api, root = release
    (root / "reports" / "verify_result.json").unlink()
    assert run(ctx, "--yes") == cli.EXIT_MISSING_INPUT
    assert "verify" in capsys.readouterr().err
    assert api.calls == []


def test_refuses_when_verify_did_not_pass(release, capsys):
    ctx, api, root = release
    (root / "reports" / "verify_result.json").write_text(json.dumps({"status": "fail"}))
    assert run(ctx, "--yes") == cli.EXIT_VERIFY_FAILED
    assert "verify" in capsys.readouterr().err
    assert api.calls == []


def test_refuses_without_manifest(release, capsys):
    ctx, api, root = release
    (root / "MANIFEST.sha256").unlink()
    assert run(ctx) == cli.EXIT_MISSING_INPUT
    err = capsys.readouterr().err
    assert "verify" in err
    assert "MANIFEST.sha256" in err
    assert api.calls == []


def test_expected_small_set_and_exclusions(release):
    ctx, _, _ = release
    names = {f.rel for f in publish.small_files(ctx.root)}
    assert names == {
        "derived/buildings.parquet",
        "derived/folds.parquet",
        "derived/splits/seed_1000.parquet",
        "derived/clients/N50_seed_1000.parquet",
        "derived/clients/N50_seed_1000_meta.parquet",
        "embeddings/enc/seamlessphoto.npy",
        "embeddings/enc/seamlessphoto.ids.npy",
        "embeddings/enc/seamlessphoto.npy.meta.json",
        "MANIFEST.sha256",
        "reports/imagery_decision.json",
        "reports/folds.csv",
        "reports/verify_result.json",
    }


def test_key_csv_can_never_be_listed(release, tmp_home):
    ctx, _, root = release
    # Plant a key everywhere a glob might reach it.
    _touch(tmp_home / "embeddings" / "audit" / "key.csv")
    _touch(tmp_home / "derived" / "splits" / "key.csv")
    _touch(root / "reports" / "key.csv")
    _touch(root / "reports" / "key.json")
    names = [f.rel for f in publish.small_files(ctx.root)]
    assert not [n for n in names if n.endswith("key.csv")]
    assert not [n for n in names if "audit" in n.split("/")]
    assert not publish.is_publishable("audit/key.csv")
    assert not publish.is_publishable("audit/anything.json")
    assert not publish.is_publishable("reports/audit/key.csv")
    assert not publish.is_publishable("derived/splits/key.csv")


def test_yes_creates_private_repo_and_commits_expected_paths(release, token):
    ctx, api, _ = release
    assert run(ctx, "--yes") == cli.EXIT_OK
    (create,) = api.named("create_repo")
    assert create["repo_id"] == HF_DATASET_REPO_ID == "AbbasABC/HiME-dataset"
    assert create["repo_type"] == "dataset"
    assert create["private"] is True
    assert create["exist_ok"] is True
    paths = api.committed_paths()
    assert "README.md" in paths
    assert "derived/buildings.parquet" in paths
    assert "MANIFEST.sha256" in paths
    assert not [p for p in paths if "audit" in p.split("/") or p.endswith("key.csv")]
    assert not [p for p in paths if p.startswith(("chips/", "tiles/"))]
    assert api.named("upload_large_folder") == []


def test_imagery_only_with_flag_and_never_the_audit_dir(release, token, tmp_home):
    ctx, api, _ = release
    assert run(ctx, "--yes", "--include-imagery") == cli.EXIT_OK
    (big,) = api.named("upload_large_folder")
    assert big["repo_id"] == HF_DATASET_REPO_ID
    assert big["repo_type"] == "dataset"
    assert big["folder_path"] == str(tmp_home)
    assert sorted(big["allow_patterns"]) == ["chips/**", "tiles/**"]
    assert "audit/**" in big["ignore_patterns"]
    assert "**/key.csv" in big["ignore_patterns"]
    assert not [p for p in api.committed_paths() if p.startswith(("chips/", "tiles/"))]


def test_dry_run_with_imagery_summarises_without_uploading(release, capsys):
    ctx, api, _ = release
    assert run(ctx, "--include-imagery") == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "chips" in out
    assert "tiles" in out
    assert api.calls == []


def test_real_upload_without_token_is_a_clear_error(release, capsys, monkeypatch):
    ctx, api, _ = release
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert run(ctx, "--yes") == cli.EXIT_MISSING_INPUT
    assert "HF_TOKEN" in capsys.readouterr().err
    assert api.calls == []


def test_token_never_appears_in_output(release, capsys, token):
    ctx, _, _ = release
    assert run(ctx) == cli.EXIT_OK
    assert run(ctx, "--yes", "--include-imagery") == cli.EXIT_OK
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err


def test_token_not_in_the_run_log(release, token, tmp_home):
    ctx, _, _ = release
    assert run(ctx, "--yes") == cli.EXIT_OK
    for p in (tmp_home / "logs").glob("*.jsonl"):
        assert SENTINEL not in p.read_text()


def test_card_front_matter_parses_as_yaml_and_embeds_dataset_md(release):
    ctx, _, _ = release
    card = publish.build_card(ctx.root)
    assert card.startswith("---\n")
    front, body = card[4:].split("\n---\n", 1)
    meta = yaml.safe_load(front)
    assert meta["license"] == "cc-by-4.0"
    assert meta["language"] == ["en"]
    assert meta["pretty_name"] == "HiME Noto 2024 building collapse"
    assert meta["size_categories"] == ["100K<n<1M"]
    assert meta["task_categories"]
    assert "body text" in body
    assert "## Files" in body
    for fact in ("10.5194/essd-17-5259-2025", "10.5281/zenodo.11055711", "Vescovo"):
        assert fact in body
    assert "CC-BY-4.0" in body
    assert "US Public Domain" in body
    assert "GSI" in body


def test_card_is_uploaded_as_readme_with_the_card_text(release, token):
    ctx, api, root = release
    assert run(ctx, "--yes") == cli.EXIT_OK
    ops = {op.path_in_repo: op for kw in api.named("create_commit") for op in kw["operations"]}
    assert ops["README.md"].path_or_fileobj == publish.build_card(root).encode()
