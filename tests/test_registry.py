"""Device records: schema-valid, and no field filled from the wrong kind of source."""

import json
from pathlib import Path

import yaml

from execubench import registry, schemas

ROOT = Path(__file__).resolve().parent.parent
DEVICES_DOC = json.loads((ROOT / "data" / "devices" / "devices.json").read_text())
DEVICES = DEVICES_DOC["devices"]
STANDARD = yaml.safe_load((ROOT / "data" / "devices" / "standard.yaml").read_text())["tiers"]
STANDARD_IDS = {d for tier in STANDARD.values() for d in tier["devices"]}
SPECS = yaml.safe_load((ROOT / "data" / "devices" / "specs.yaml").read_text())


def test_every_record_is_schema_valid():
    for rec in DEVICES:
        assert schemas.errors_for("device.schema.json", rec) == [], rec["id"]


def test_devices_json_is_current():
    roots = [ROOT / r for r in DEVICES_DOC["generated_from"] if r.startswith("data/devices/probe/")]
    assert registry.build(roots) == DEVICES


def test_unpublished_figures_stay_unknown():
    for rec in DEVICES:
        assert rec["compute"]["bf16_tflops"]["provenance"] == "unknown"
        assert rec["memory"]["bandwidth_gbps"]["provenance"] == "unknown"
        assert rec["npu"]["int8_tops"]["provenance"] == "unknown"


def test_usable_ram_comes_from_the_phone_not_the_catalogue():
    for rec in DEVICES:
        ram = rec["memory"]["mem_total_gib"]
        assert ram["provenance"] == "reported"
        # Device Farm's catalogue "memory" is storage (tens to hundreds of GB).
        assert ram["value"] < 20
        assert rec["provider"]["catalogue"]["memory"] > 10e9


def test_every_published_value_has_a_maker_url():
    for key, soc in SPECS["socs"].items():
        assert soc["source"].startswith("https://"), key
        assert soc["checked_by"], key
    for key, phone in SPECS["phones"].items():
        assert phone["source"].startswith("https://"), key
        assert phone["checked_by"], key
        assert phone["soc"] in SPECS["socs"], key


def test_device_ids_change_with_the_build():
    ids = [r["id"] for r in DEVICES]
    assert len(set(ids)) == len(ids)
    for rec in DEVICES:
        assert rec["id"].startswith(rec["device_model_id"] + "-")
    # Three models were seen on two firmware builds (two across the first two probes, the
    # Pixel 2 XL across the two script checks); every build is kept.
    by_model: dict[str, list] = {}
    for rec in DEVICES:
        by_model.setdefault(rec["device_model_id"], []).append(rec)
    assert {m for m, recs in by_model.items() if len(recs) > 1} == {
        "sm-s928u1-android14",
        "23090ra98g-android15",
        "google-pixel-2-xl-android8-1-0",
    }
    assert all(sum(r["latest_for_model"] for r in recs) == 1 for recs in by_model.values())


def test_architecture_labels_follow_core_classes():
    assert registry.architecture(["efficiency"], 1) == "all-efficiency-core, single-cluster"
    assert registry.architecture(["performance", "performance"], 2) == "all-big-core, dual-cluster"
    assert registry.architecture(["performance", "efficiency"], 2) == "big.LITTLE, dual-cluster"
    assert registry.architecture(["performance", "unknown"], 2) == "core classes unknown, dual-cluster"


def test_every_standard_device_is_probed_and_has_published_specs():
    by_model = {r["device_model_id"]: r for r in DEVICES if r["latest_for_model"]}
    for model_id in STANDARD_IDS:
        rec = by_model[model_id]
        assert rec["soc"]["marketing_name"]["provenance"] == "published", model_id
        assert rec["memory"]["capacity_gb"]["provenance"] == "published", model_id
        assert rec["identity"]["arm64_userspace"] is True, model_id


def test_devices_doc_table_is_current():
    doc = (ROOT / "docs" / "DEVICES.md").read_text()
    start = doc.index(registry.TABLE_BEGIN) + len(registry.TABLE_BEGIN)
    end = doc.index(registry.TABLE_END)
    assert doc[start:end].strip() == registry.markdown_table(DEVICES).strip()
