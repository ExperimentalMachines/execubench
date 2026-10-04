"""data/pilot/p1.yaml: every pin real, every device and model known, the worst case affordable."""

import json
from pathlib import Path

import yaml

from execubench import versions

ROOT = Path(__file__).resolve().parent.parent
PILOT = yaml.safe_load((ROOT / "data" / "pilot" / "p1.yaml").read_text())
INVENTORY = {
    (f["repo"], f["file"]): f for f in json.loads((ROOT / "data" / "models" / "xnnpack.json").read_text())["files"]
}
DEVICES = {
    r["device_model_id"]: r
    for r in json.loads((ROOT / "data" / "devices" / "devices.json").read_text())["devices"]
    if r["latest_for_model"]
}
ENV = versions.load()


def test_models_match_the_inventory():
    for key, m in PILOT["models"].items():
        inv = INVENTORY[(m["repo"], m["file"])]
        for field in ("revision", "sha256", "tokenizer", "tokenizer_sha256"):
            assert m[field] == inv[field], (key, field)
        assert m["export_executorch"] == inv["executorch"]


def test_devices_match_the_registry():
    for model_id, arn in PILOT["devices"].items():
        assert DEVICES[model_id]["provider"]["device_arn"] == arn


def test_experiments_reference_known_names():
    for name, e in PILOT["experiments"].items():
        assert set(e["devices"]) <= set(PILOT["devices"]), name
        assert set(e["models"]) <= set(PILOT["models"]), name


def test_worst_case_is_under_the_ceiling():
    worst = sum(
        len(e["devices"]) * len(e["models"]) * len(e["arms"]) * e["jobs_per_cell"] * e["job_timeout_min"]
        for e in PILOT["experiments"].values()
    )
    assert worst <= PILOT["max_device_minutes"], worst


def test_runtime_matches_the_pin():
    assert PILOT["runtime"]["executorch"] == ENV["EXECUTORCH_VERSION"]


def test_every_experiment_has_a_known_workload():
    for name, e in PILOT["experiments"].items():
        assert e["workload"] in PILOT["workloads"], name


def test_contract_acceptance_covers_every_tier_a_vendor():
    standard = yaml.safe_load((ROOT / "data" / "devices" / "standard.yaml").read_text())["tiers"]["A"]["devices"]
    assert set(PILOT["experiments"]["contract_acceptance"]["devices"]) == set(standard)


def test_workloads_are_pinned_bytes():
    from execubench import pilot

    for name, w in PILOT["workloads"].items():
        problems = pilot.workload_problems(name, w)
        # Only the speed source text may still be open, and the gate reports it.
        assert all("source text" in p for p in problems), problems


def test_the_gate_lists_what_blocks_scheduling():
    from execubench import pilot

    blocked = pilot.blockers(PILOT)
    assert any("apk_sha256" in b for b in blocked)
    assert any("orchestration" in b for b in blocked)
    assert pilot.worst_case_minutes(PILOT) <= PILOT["max_device_minutes"]


def test_a_changed_prompt_file_is_caught(tmp_path):
    from execubench import pilot

    w = dict(PILOT["workloads"]["smoke-v1"])
    (tmp_path / "data" / "pilot").mkdir(parents=True)
    (tmp_path / w["prompts_file"]).write_text('{"id": "x", "turns": ["changed"]}\n')
    assert any("prompts_sha256" in p for p in pilot.workload_problems("smoke-v1", w, tmp_path))


def test_the_ledger_enforces_one_budget_across_runs(tmp_path):
    import pytest

    from execubench import ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.reserve(path, 100, "run-a", 60) == 60
    with pytest.raises(ledger.OverBudget, match="above its budget"):
        ledger.reserve(path, 100, "run-b", 60)
    with pytest.raises(ledger.OverBudget, match="already holds"):
        ledger.reserve(path, 1000, "run-a", 1)
    assert ledger.reserve(path, 100, "run-c", 40) == 100
    assert ledger.reserved(path) == 100
    for budget, worst in ((float("nan"), 1), (100, float("nan")), (float("inf"), 1), (100, 0)):
        with pytest.raises(ValueError):
            ledger.reserve(tmp_path / "other.jsonl", budget, "x", worst)
    (tmp_path / "bad.jsonl").write_text('{"name": "a", "worst_minutes": NaN}\n')
    with pytest.raises(ValueError):
        ledger.reserved(tmp_path / "bad.jsonl")


def test_the_cli_refuses_nonfinite_minutes():
    import argparse

    import pytest

    from execubench.__main__ import finite_minutes

    for bad in ("nan", "inf", "-1", "0"):
        with pytest.raises(argparse.ArgumentTypeError):
            finite_minutes(bad)
    assert finite_minutes("30") == 30


def test_compat_evidence_is_checked_not_assumed(tmp_path):
    import shutil

    from execubench import pilot

    model = PILOT["models"]["qwen3-1.7b"]
    assert pilot.compat_problems(model, "1.5.1") == []
    stem = Path(model["file"]).stem
    for f in pilot.COMPAT.glob(f"{stem}.*"):
        shutil.copy(f, tmp_path / f.name)
    compare = tmp_path / f"{stem}.compare.json"
    compare.write_text(compare.read_text().replace('"identical": false', '"identical": true', 1))
    assert any("derives" in p for p in pilot.compat_problems(model, "1.5.1", tmp_path))
    assert any("missing" in p for p in pilot.compat_problems(dict(model, file="xnnpack/nothing.pte"), "1.5.1"))
    assert any("another .pte" in p for p in pilot.compat_problems(dict(model, sha256="0" * 64), "1.5.1"))


def test_compat_evidence_must_show_execution(tmp_path):
    import shutil

    from execubench import pilot

    model = PILOT["models"]["smollm2-360m"]
    stem = Path(model["file"]).stem
    for f in pilot.COMPAT.glob(f"{stem}.*"):
        shutil.copy(f, tmp_path / f.name)
    assert pilot.compat_problems(model, "1.5.1", tmp_path) == []
    for v in ("1.4.0", "1.5.1"):
        path = tmp_path / f"{stem}.executorch-{v}.json"
        report = json.loads(path.read_text())
        report["max_new_tokens"] = 0
        for row in report["results"]:
            row.update({"pieces": [], "stats": {}})
        path.write_text(json.dumps(report))
    problems = pilot.compat_problems(model, "1.5.1", tmp_path)
    assert any("max_new_tokens" in p for p in problems) and any("completed generation" in p for p in problems)
    (tmp_path / f"{stem}.executorch-1.4.0.json").write_text("[]")
    assert pilot.compat_problems(model, "1.5.1", tmp_path) == [
        "host compatibility evidence is not a set of JSON objects"
    ]
