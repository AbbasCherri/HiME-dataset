"""`notodata report` rewrites only the AUTO blocks of DATASET.md."""

from __future__ import annotations

import json

import pandas as pd

from notodata import report

DOC = """# Title

Hand-written intro, with <!-- not a marker --> and a table | a | b |.

## 3. Sources
<!-- AUTO:source_files --> file, size, sha256 <!-- /AUTO -->
Hand text after.

## 6. Class counts
<!-- AUTO:class_counts_raw -->
old content
<!-- /AUTO -->
<!-- AUTO:moran --> bin, n <!-- /AUTO -->
Final hand-written line.
"""


def test_replaces_only_the_named_blocks():
    out = report.replace_auto_blocks(DOC, {"class_counts_raw": "| x |\n| 1 |"})
    assert "| 1 |" in out
    assert "old content" not in out
    # untouched: other AUTO block, and every hand-written line
    assert "<!-- AUTO:moran --> bin, n <!-- /AUTO -->" in out
    assert "<!-- AUTO:source_files --> file, size, sha256 <!-- /AUTO -->" in out
    for line in DOC.splitlines():
        if "AUTO" not in line and line != "old content":
            assert line in out.splitlines()


def test_unknown_block_names_are_ignored_and_replace_is_idempotent():
    blocks = {"moran": "| a |", "not_in_doc": "zzz"}
    once = report.replace_auto_blocks(DOC, blocks)
    assert "zzz" not in once
    assert report.replace_auto_blocks(once, blocks) == once


def test_hand_text_outside_markers_is_byte_identical():
    out = report.replace_auto_blocks(DOC, {"moran": "M", "source_files": "S"})
    assert report._AUTO.sub("", out) == report._AUTO.sub("", DOC)


def test_update_dataset_md_from_reports(tmp_path, tmp_home):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"bin_hi_km": [0.5, 1.0], "I": [0.12, 0.03]}).to_csv(
        reports / "moran.csv", index=False
    )
    (tmp_home / "raw").mkdir()
    (tmp_home / "raw" / "source_files.json").write_text(
        json.dumps({"grid.xml": {"size": 10, "sha256": "ab" * 32}})
    )
    doc = tmp_path / "DATASET.md"
    doc.write_text(DOC)
    changed = report.update_dataset_md(doc, root=tmp_path)
    text = doc.read_text()
    assert changed
    assert "| bin_hi_km | I |" in text
    assert "grid.xml" in text
    assert "old content" in text  # class_counts_raw.csv absent: block left as it was
    assert report.update_dataset_md(doc, root=tmp_path) is False


def test_layers_block_lists_the_registry(tmp_path):
    blocks = report.render_blocks(tmp_path)
    assert "seamlessphoto" in blocks["layers"]


def test_seed_register_comes_from_config(tmp_path):
    assert "1000" in report.render_blocks(tmp_path)["seed_register"]
