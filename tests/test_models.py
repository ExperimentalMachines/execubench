"""The model inventory: every file pinned and its hashes in agreement."""

import json
import re
from pathlib import Path

INV = json.loads((Path(__file__).resolve().parent.parent / "data" / "models" / "xnnpack.json").read_text())["files"]


def test_inventory_is_not_empty():
    assert len(INV) >= 60


def test_every_file_is_pinned_and_hashes_agree():
    for f in INV:
        assert re.fullmatch(r"[0-9a-f]{40}", f["revision"]), f["file"]
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f["file"]
        assert f["hashes_agree"], f["file"]
        assert re.fullmatch(r"[0-9a-f]{64}", f["tokenizer_sha256"]), f["file"]


def test_windows_parse_from_names():
    for f in INV:
        assert f["window"] in {2048, 4096, 8192, 16384, 32768}, f["file"]


def test_every_repo_has_an_8k_file():
    """docs/PLAN.md runs the quality and speed tracks at 8k for every model."""
    repos = {f["repo"] for f in INV}
    with_8k = {f["repo"] for f in INV if f["window"] == 8192}
    assert repos == with_8k
