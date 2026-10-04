"""The model inventory: every file pinned and its hashes in agreement."""

import json
import re  # noqa: F401
from pathlib import Path

from execubench import models

INV = json.loads((Path(__file__).resolve().parent.parent / "data" / "models" / "xnnpack.json").read_text())["files"]


def test_inventory_is_not_empty():
    assert len(INV) >= 60


def test_every_file_is_pinned_and_hashes_agree():
    # models.problems compares the hashes themselves; the hashes_agree flag is not trusted.
    assert models.problems(INV) == []


def test_a_tampered_entry_is_caught():
    bad = dict(INV[0], report_sha256="0" * 64)  # flag still says the hashes agree
    assert any("differs" in p for p in models.problems([bad]))
    assert any("flag" in p for p in models.problems([bad]))
    assert any("tokenizer_sha256" in p for p in models.problems([dict(INV[0], tokenizer_sha256=None)]))


def test_windows_parse_from_names():
    for f in INV:
        assert f["window"] in {2048, 4096, 8192, 16384, 32768}, f["file"]


def test_every_repo_has_an_8k_file():
    """docs/PLAN.md runs the quality and speed tracks at 8k for every model."""
    repos = {f["repo"] for f in INV}
    with_8k = {f["repo"] for f in INV if f["window"] == 8192}
    assert repos == with_8k


def test_empty_duplicate_and_conflicting_scans_are_refused():
    assert models.problems([]) == ["the scan found no files"]
    assert any("listed twice" in p for p in models.problems([INV[0], INV[0]]))
    assert any("disagree" in p for p in models.problems([dict(INV[0], report_conflict=True)]))
    assert any("source_revision" in p for p in models.problems([dict(INV[0], source_revision="main")]))
