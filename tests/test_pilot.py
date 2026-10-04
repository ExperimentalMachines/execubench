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
