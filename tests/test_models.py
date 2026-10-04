"""The model inventory: every file pinned and its hashes in agreement."""

import hashlib
import json
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


class _Sib:
    def __init__(self, name, sha=None, size=1):
        self.rfilename, self.size = name, size
        self.lfs = type("L", (), {"sha256": sha})() if sha else None


def _fake_hub(monkeypatch, tmp_path, reports: dict, pte_sha: str):
    """A Hub with one repo, one 8k .pte and the given export reports, no network."""
    import huggingface_hub

    siblings = [_Sib("xnnpack/M-8da4w-8k.pte", pte_sha), _Sib("tokenizer.json")]
    siblings += [_Sib(name) for name in reports]
    files = {name: json.dumps(r) for name, r in reports.items()} | {"tokenizer.json": "{}"}

    class Api:
        revisions: list = []

        def __init__(self, token=None):
            pass

        def list_models(self, author):
            return [type("M", (), {"id": f"{author}/M-ExecuTorch"})()]

        def model_info(self, repo, files_metadata, revision=None):
            Api.revisions.append(revision)
            return type("I", (), {"sha": "a" * 40, "siblings": siblings})()

    def download(repo, name, revision, token=None):
        path = tmp_path / name.replace("/", "_")
        path.write_text(files[name])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return Api


def _report(sha, executorch="1.4.0"):
    return {
        "files": [{"path": "xnnpack/M-8da4w-8k.pte", "sha256": sha}],
        "source": {"id": "org/M", "sha": "b" * 40},
        "toolchain": {"executorch": executorch},
        "tokenizer": "tokenizer.json",
        "recipe": {"qmode": "8da4w"},
    }


def test_scan_reads_reports_and_hashes_git_tokenizers(monkeypatch, tmp_path):
    sha = "c" * 64
    _fake_hub(monkeypatch, tmp_path, {"xnnpack/export-report-a.json": _report(sha)}, sha)
    inv = models.scan()
    assert len(inv) == 1 and inv[0]["hashes_agree"] and not inv[0]["report_conflict"]
    assert inv[0]["tokenizer_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert models.problems(inv) == []


def test_scan_flags_reports_that_disagree_on_anything_but_the_hash(monkeypatch, tmp_path):
    sha = "c" * 64
    reports = {"xnnpack/export-report-a.json": _report(sha), "xnnpack/export-report-b.json": _report(sha, "1.5.1")}
    _fake_hub(monkeypatch, tmp_path, reports, sha)
    inv = models.scan()
    assert inv[0]["report_conflict"]
    assert any("disagree" in p for p in models.problems(inv))


def test_a_report_that_lists_a_file_twice_is_a_conflict(monkeypatch, tmp_path):
    sha = "c" * 64
    report = _report(sha)
    report["files"].append({"path": "xnnpack/M-8da4w-8k.pte", "sha256": "d" * 64})
    _fake_hub(monkeypatch, tmp_path, {"xnnpack/export-report-a.json": report}, sha)
    assert models.scan()[0]["report_conflict"]


def test_reports_that_disagree_on_sizing_conflict(monkeypatch, tmp_path):
    sha = "c" * 64
    a, b = _report(sha), _report(sha)
    a["window"] = {"table": [{"context": 8192, "device_resident_bytes": 1}]}
    b["window"] = {"table": [{"context": 8192, "device_resident_bytes": 2}]}
    _fake_hub(monkeypatch, tmp_path, {"xnnpack/export-report-a.json": a, "xnnpack/export-report-b.json": b}, sha)
    assert models.scan()[0]["report_conflict"]


def test_pins_are_checked_at_their_revision(monkeypatch, tmp_path):
    sha = "c" * 64
    api = _fake_hub(monkeypatch, tmp_path, {"xnnpack/export-report-a.json": _report(sha)}, sha)
    tok_sha = hashlib.sha256(b"{}").hexdigest()
    pin = {"repo": "experimentalmachines/M-ExecuTorch", "revision": "e" * 40, "tokenizer": "tokenizer.json"}
    pin |= {"file": "xnnpack/M-8da4w-8k.pte", "tokenizer_sha256": tok_sha}
    good = models.check_pins([{**pin, "sha256": sha}])[0]
    assert good["resolves"] and good["sha256_matches"] and good["tokenizer_matches"]
    assert api.revisions == ["e" * 40]  # asked at the pinned revision, not the head
    moved = models.check_pins([{**pin, "sha256": "d" * 64}])[0]
    assert moved["resolves"] and not moved["sha256_matches"]
    other_tok = models.check_pins([{**pin, "sha256": sha, "tokenizer_sha256": "0" * 64}])[0]
    assert not other_tok["tokenizer_matches"] and other_tok["tokenizer_sha256_observed"] == tok_sha


def test_the_committed_pin_check_covers_the_inventory():
    check = json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "models" / "pin-check-2026-10-04.json").read_text()
    )
    rows = {(r["repo"], r["revision"], r["file"]): r for r in check["files"]}
    assert set(rows) == {(f["repo"], f["revision"], f["file"]) for f in INV}
    for f in INV:
        r = rows[(f["repo"], f["revision"], f["file"])]
        assert r["sha256_expected"] == f["sha256"] and r["tokenizer_sha256_expected"] == f["tokenizer_sha256"]
        assert r["resolves"] and r["sha256_matches"] and r["tokenizer_matches"]


def test_a_scan_that_loses_files_is_refused():
    assert any("missing from the scan" in p for p in models.problems(INV[:1], INV))
    assert models.problems(INV, INV) == []
    no_8k = [f for f in INV if not (f["repo"] == INV[0]["repo"] and f["window"] == 8192)]
    assert any("no 8k file" in p for p in models.problems(no_8k))
