"""Build device records (schemas/device.schema.json) from a probe run plus the published specs.

Reported values come from the probe's parsed profile; published values from
data/devices/specs.yaml. A field is never filled from the other kind. What nobody we can
cite states stays `unknown` with a null value.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "data" / "devices" / "specs.yaml"


CLUSTER_FIELDS = {
    "cores",
    "cpus",
    "design",
    "design_source",
    "implementer",
    "part",
    "max_mhz",
    "efficiency_class",
    "core_class",
}


def _unknown(note: str | None = None) -> dict:
    out = {"value": None, "provenance": "unknown"}
    if note:
        out["note"] = note
    return out


def _published(value, source: str, retrieved: str, unit: str | None = None, note: str | None = None) -> dict:
    out = {"value": value, "provenance": "published", "source": source, "retrieved": retrieved}
    if unit:
        out["unit"] = unit
    if note:
        out["note"] = note
    return out


def _reported(value, source: str, unit: str | None = None) -> dict:
    out = {"value": value, "provenance": "reported", "source": source}
    if unit:
        out["unit"] = unit
    return out


def architecture(classes: list[str], cluster_count: int) -> str:
    """Label from the full set of core classes: all-big-core, all-efficiency-core, big.LITTLE
    or unknown, plus the clock-domain count."""
    known = set(classes) - {"unknown"}
    if not classes or "unknown" in classes and not ("efficiency" in known and "performance" in known):
        kind = "core classes unknown"
    elif known == {"performance"}:
        kind = "all-big-core"
    elif known == {"efficiency"}:
        kind = "all-efficiency-core"
    else:
        kind = "big.LITTLE"
    shape = {1: "single-cluster", 2: "dual-cluster", 3: "tri-cluster", 4: "quad-cluster"}.get(
        cluster_count, f"{cluster_count}-cluster"
    )
    return f"{kind}, {shape}"


def record(job_dir: Path, specs: dict) -> dict:
    """One device record from one probed job folder (data/devices/probe/<date>/<slug>)."""
    job = json.loads((job_dir / "devicefarm-job.json").read_text())
    prof = json.loads((job_dir / "profile.json").read_text())
    cat = job["device"]
    rep = prof["reported"]
    artifacts = str((job_dir.resolve() / "artifacts" / "probe").relative_to(ROOT))
    retrieved = str(specs["retrieved"])

    soc_key = specs["platform_to_soc"].get(rep["soc_model"]) or specs["platform_to_soc"].get(rep["board_platform"])
    soc = specs["socs"].get(soc_key, {}) if soc_key else {}
    phone = specs["phones"].get(cat["modelId"].strip("{}"), {})

    soc_name = (
        _published(phone["soc_in_phone"], phone["source"], retrieved)
        if phone.get("soc_in_phone")
        else _published(soc["marketing_name"], soc["source"], retrieved)
        if soc
        else _unknown()
    )
    cpu = prof["cpu"]
    clusters = [{k: v for k, v in c.items() if k in CLUSTER_FIELDS} for c in cpu["clusters"]]
    gpu_name = prof["gpu"].get("vulkan_device")
    npu_tops = soc.get("npu_tops")
    mem_kib = prof["memory"]["mem_total_kib"]
    ddr = prof["memory"]["ddr"]

    if ddr.get("type"):
        mem_type = _reported(ddr["type"], f"{artifacts}/getprop.txt (ro.boot.hardware.ddr)")
    elif phone.get("memory_type_in_phone"):
        mem_type = _published(phone["memory_type_in_phone"], phone["source"], retrieved)
    else:
        mem_type = _unknown("Phone maker does not state it; the chip's supported types are not the fitted type")
    # Xiaomi states "8533 Mbps" per pin, which is the same quantity as MT/s.
    rate = (
        _published(phone["memory_rate_in_phone"], phone["source"], retrieved, "MT/s")
        if phone.get("memory_rate_in_phone")
        else _unknown("The phone maker does not state the fitted DRAM speed; the chip's supported maximum is not it")
    )

    form = (cat.get("formFactor") or "PHONE").lower()
    # A device id names one model at one firmware build: the same model on another build is
    # another device, and results are never pooled across ids (docs/DEVICES.md).
    build = hashlib.sha256(rep["fingerprint"].encode()).hexdigest()[:8]
    model_id = re.sub(r"[^a-z0-9]+", "-", job_dir.name.lower()).strip("-")
    return {
        "id": f"{model_id}-{build}",
        "device_model_id": model_id,
        "provider": {
            "name": "aws-device-farm",
            "region": "us-west-2",
            "device_arn": cat["arn"],
            "catalogue": cat,
        },
        "identity": {
            "manufacturer": rep["manufacturer"] or cat["manufacturer"],
            "marketing_name": phone.get("name") or cat["name"],
            "model_code": rep["model"] or cat["modelId"],
            "android": rep["android"],
            "sdk": rep["sdk"],
            "fingerprint": rep["fingerprint"],
            "security_patch": rep["security_patch"],
            "form_factor": "tablet" if form == "tablet" else "phone",
            "abilist": rep.get("abilist"),
            "arm64_userspace": rep.get("arm64_userspace"),
        },
        "soc": {
            "reported_manufacturer": rep["soc_manufacturer"],
            "reported_model": rep["soc_model"],
            "platform": rep["board_platform"],
            "marketing_name": soc_name,
            "process_node": _published(soc["process_node"], soc.get("process_node_source", soc["source"]), retrieved)
            if soc.get("process_node")
            else _unknown(),
        },
        "cpu": {
            "count": cpu["count"],
            "topology": cpu["topology"],
            "tiers": cpu["tiers"],
            "cluster_count": len(cpu["clock_domains"]),
            "all_big_cores": cpu["all_big_cores"],
            "architecture": architecture([c["core_class"] for c in cpu["clusters"]], len(cpu["clock_domains"])),
            "vendor_description": _published(
                soc["cpu_published"], soc.get("cpu_published_source", soc["source"]), retrieved
            )
            if soc.get("cpu_published")
            else _unknown(),
            "max_mhz": cpu["max_mhz"],
            "features": cpu["features"],
            "clusters": clusters,
            "clock_domains": cpu["clock_domains"],
        },
        "gpu": {
            "name": _reported(gpu_name, f"{artifacts}/vkjson.txt (Vulkan deviceName)") if gpu_name else _unknown(),
            "renderer": prof["gpu"].get("gles"),
            "vulkan_api": None,
        },
        "npu": {
            "name": _published(soc["npu_name"], soc["source"], retrieved) if soc.get("npu_name") else _unknown(),
            "int8_tops": _unknown(
                f"Maker states {npu_tops['value']} TOPS without a precision, so it is not an INT8 figure"
                if npu_tops
                else "Not found in the maker sources reviewed"
                if soc
                else "Maker sources not reviewed for this chip"
            ),
            "runtimes_present": sorted(prof["npu"]["runtimes_present"]),
        },
        "memory": {
            "mem_total_gib": _reported(round(mem_kib / 2**20, 2), f"{artifacts}/meminfo.txt (MemTotal)", "GiB"),
            "capacity_gb": _published(phone["ram_gb"], phone["source"], retrieved, "GB")
            if phone.get("ram_gb")
            else _unknown(),
            "type": mem_type,
            "data_rate_mtps": rate,
            "bus_width_bits": _unknown(
                "Not found in the maker sources reviewed" if soc else "Maker sources not reviewed for this chip"
            ),
            "bandwidth_gbps": _unknown("Not derivable: needs a sourced data rate and a sourced bus width"),
        },
        "compute": {
            "bf16_tflops": _unknown(
                "No BF16 TFLOPS figure for CPU, GPU or NPU in the maker sources reviewed; see docs/DEVICES.md"
                if soc
                else "Maker sources not reviewed for this chip"
            ),
        },
        "probe": {
            "run_arn": job["arn"].rsplit("/", 1)[0].replace(":job:", ":run:"),
            "artifact_dir": artifacts,
            "date": str(date.fromisoformat(job["created"][:10])),
            "fixed_performance_mode": prof["state_at_probe"]["fixed_performance_mode"],
            "simpleperf": prof["tooling"]["simpleperf"],
            "adb_push_mib_s": prof["staging"]["adb_push_mib_s"],
        },
    }


def build(probe_roots: list[Path], specs_path: Path = SPECS) -> list[dict]:
    """One record per device configuration (model, Android version, firmware build) across
    the given probe runs, which are given oldest first.

    A configuration probed twice keeps its latest probe (the newer script collects more). A
    model whose firmware changed between probes keeps one record per build, because jobs that
    ran on the older build still refer to it; `latest_for_model` marks the newest.
    """
    specs = yaml.safe_load(specs_path.read_text())
    by_id: dict[str, dict] = {}
    order: dict[str, int] = {}
    for rank, root in enumerate(probe_roots):
        for d in sorted(root.iterdir()):
            if (d / "profile.json").exists():
                rec = record(d, specs)
                by_id[rec["id"]] = rec
                order[rec["id"]] = rank
    latest: dict[str, str] = {}
    for rec_id, rec in by_id.items():
        model = rec["device_model_id"]
        if model not in latest or order[rec_id] >= order[latest[model]]:
            latest[model] = rec_id
    for rec_id, rec in by_id.items():
        rec["latest_for_model"] = latest[rec["device_model_id"]] == rec_id
    return [by_id[k] for k in sorted(by_id)]


TABLE_BEGIN = "<!-- devices-table:begin (generated by `execubench devices table`; do not edit) -->"
TABLE_END = "<!-- devices-table:end -->"


def _cell(v) -> str:
    return "" if v is None else str(v)


def markdown_table(records: list[dict], model_ids: set[str] | None = None) -> str:
    """The probe table in docs/DEVICES.md, rendered from device records so it cannot drift."""
    head = (
        "| Device (model id) | Android | SoC reported | SoC (published name) | Topology | Tiers | Architecture "
        "| Top MHz | GPU (reported) | MemTotal GiB | Marketed RAM GB | arm64 |\n"
        "|---|---|---|---|---|---|---|---:|---|---:|---|---|\n"
    )
    rows = []
    for r in sorted(records, key=lambda r: (r["identity"]["manufacturer"].lower(), r["identity"]["marketing_name"])):
        if not r.get("latest_for_model", True):
            continue
        if model_ids is not None and r["device_model_id"] not in model_ids:
            continue
        soc = r["soc"]
        cpu = r["cpu"]
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{r['identity']['marketing_name']} ({r['provider']['catalogue']['modelId'].strip('{}')})",
                    _cell(r["identity"]["android"]),
                    " ".join(x for x in (soc["reported_manufacturer"], soc["reported_model"]) if x)
                    or _cell(soc["platform"]),
                    _cell(soc["marketing_name"]["value"]) or "unknown",
                    _cell(cpu["topology"]) or "unknown",
                    _cell(cpu["tiers"]),
                    _cell(cpu["architecture"]),
                    _cell(cpu["max_mhz"]),
                    _cell(r["gpu"]["name"]["value"]) or "unknown",
                    f"{r['memory']['mem_total_gib']['value']:.2f}",
                    _cell(r["memory"]["capacity_gb"]["value"]) or "unknown",
                    "yes" if r["identity"]["arm64_userspace"] else "no",
                ]
            )
            + " |"
        )
    return head + "\n".join(rows) + "\n"
