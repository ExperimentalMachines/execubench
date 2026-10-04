"""Parse a probe's raw dumps into a device profile.

Only what the phone reports is parsed here. Vendor figures that a phone cannot report
(memory bandwidth, peak FLOPs, NPU) live in `data/devices/specs.yaml` with a source per
value, and are joined later; the two are never mixed in one field.

Core designs come from MIDR_EL1 (implementer and part number), named through the Linux
kernel's own table (`data/reference/linux-cputype.h`, pinned to a commit), never from a
marketing name. Clusters are cores that share a design and a maximum clock.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPUTYPE = ROOT / "data" / "reference" / "linux-cputype.h"

IMPLEMENTERS = {0x41: "ARM", 0x51: "QCOM", 0x53: "SAMSUNG", 0x48: "HISI", 0x61: "APPLE"}

# Core classes, each with its source. A design in neither set is "unknown", and one unknown
# core makes the phone's all-big-core answer unknown too: absence from a list is not evidence.
# Arm's own classification: Cortex-A5xx and C1-Nano are its efficiency cores; X, A7xx, C1-Ultra,
# C1-Premium and C1-Pro are its performance cores.
EFFICIENCY_PARTS = {
    (0x41, 0xD03): "Arm (Cortex-A53)",
    (0x41, 0xD05): "Arm (Cortex-A55)",
    (0x41, 0xD46): "Arm (Cortex-A510)",
    (0x41, 0xD80): "Arm (Cortex-A520)",
    (0x41, 0xD8A): "Arm (C1-Nano)",
    # Qualcomm's Kryo "Silver" cores are its efficiency cores (Arm Cortex-A55 derivatives).
    (0x51, 0x801): "Qualcomm (Kryo 2xx Silver)",
    (0x51, 0x803): "Qualcomm (Kryo 3xx Silver)",
    (0x51, 0x805): "Qualcomm (Kryo 4xx Silver)",
}
PERFORMANCE_PARTS = {
    **{
        (0x41, part): "Arm"
        for part in (0xD09, 0xD0A, 0xD0B, 0xD0D, 0xD41, 0xD44, 0xD47, 0xD48, 0xD4D, 0xD4E, 0xD81, 0xD82, 0xD85, 0xD87)
    },
    # Qualcomm's Kryo "Gold" cores are its performance cores.
    (0x51, 0x800): "Qualcomm (Kryo 2xx Gold)",
    (0x51, 0x802): "Qualcomm (Kryo 3xx Gold)",
    (0x51, 0x804): "Qualcomm (Kryo 4xx Gold)",
    (0x41, 0xD8B): "Arm (C1-Pro)",
    (0x41, 0xD8C): "Arm (C1-Ultra)",
    (0x41, 0xD90): "Arm (C1-Premium)",
    # Qualcomm calls every Oryon core in these phones Prime or Performance and lists no
    # efficiency cores (SM8750 and SM8850 product briefs, data/devices/specs.yaml).
    (0x51, 0x001): "Qualcomm SM8750 brief: Prime and Performance cores only",
    (0x51, 0x002): "Qualcomm SM8850 brief: 2 Prime + 6 Performance cores",
}


def core_class(implementer: int | None, part: int | None) -> str:
    if (implementer, part) in EFFICIENCY_PARTS:
        return "efficiency"
    if (implementer, part) in PERFORMANCE_PARTS:
        return "performance"
    return "unknown"


def kernel_part_names(path: Path = CPUTYPE) -> dict[tuple[int, int], str]:
    """(implementer, part) -> the kernel's macro name, e.g. (0x41, 0xD82) -> 'CORTEX_X4'."""
    imps = {}
    parts = {}
    for line in path.read_text().splitlines():
        m = re.match(r"#define (\w+)_CPU_IMP_(\w+)\s+0x([0-9A-Fa-f]+)", line)
        if m:
            imps[m.group(2)] = int(m.group(3), 16)
            continue
        m = re.match(r"#define (\w+)_CPU_PART_(\w+)\s+0x([0-9A-Fa-f]+)", line)
        if m:
            parts.setdefault(m.group(1), []).append((m.group(2), int(m.group(3), 16)))
    out = {}
    for vendor, items in parts.items():
        if vendor not in imps:
            continue
        for name, part in items:
            # Several Qualcomm Kryo macros reuse a part number; keep the first, which is
            # what the kernel matches first too.
            out.setdefault((imps[vendor], part), name)
    return out


def parse_getprop(text: str) -> dict[str, str]:
    # Values can span lines; one runs to the "]" before the next "[key]: [" line.
    pattern = r"^\[([^\]\n]+)\]: \[(.*?)\]\s*(?=^\[[^\]\n]+\]: \[|\Z)"
    return {k: v.strip() for k, v in re.findall(pattern, text, flags=re.M | re.S)}


def parse_meminfo(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, flags=re.M)}


def parse_cpus(text: str) -> list[dict]:
    cores = []
    for block in re.split(r"^== ", text, flags=re.M)[1:]:
        name, *lines = block.strip().splitlines()
        fields = dict(line.split("=", 1) for line in lines if "=" in line)
        midr = fields.get("regs/identification/midr_el1", "")
        core = {"cpu": int(name.removeprefix("cpu"))}
        if midr.startswith("0x"):
            v = int(midr, 16)
            core["implementer"] = (v >> 24) & 0xFF
            core["part"] = (v >> 4) & 0xFFF
            core["variant"] = (v >> 20) & 0xF
            core["revision"] = v & 0xF
        related = fields.get("cpufreq/related_cpus", "")
        if related and related != "<unreadable>":
            core["related_cpus"] = [int(c) for c in related.split() if c.isdigit()]
        for key, out in [
            ("cpufreq/cpuinfo_max_freq", "max_khz"),
            ("cpufreq/cpuinfo_min_freq", "min_khz"),
            ("cpu_capacity", "capacity"),
        ]:
            val = fields.get(key, "")
            if val.isdigit():
                core[out] = int(val)
        cores.append(core)
    return cores


def clock_domains(text: str) -> list[dict]:
    """cpufreq policies: groups of cores that share one clock. Fastest first."""
    out = []
    for block in re.split(r"^== ", text, flags=re.M)[1:]:
        head, *lines = block.strip().splitlines()
        fields = dict(line.split("=", 1) for line in lines if "=" in line)
        cpus = [int(c) for c in fields.get("affected_cpus", "").split() if c.isdigit()]
        khz = fields.get("cpuinfo_max_freq", "")
        if cpus:
            out.append(
                {
                    "policy": head.rsplit("/", 1)[-1],
                    "cpus": cpus,
                    "max_mhz": round(int(khz) / 1000, 1) if khz.isdigit() else None,
                    "governor": fields.get("scaling_governor"),
                }
            )
    return sorted(out, key=lambda d: (-(d["max_mhz"] or 0), d["cpus"][0]))


def cpu_features(cpuinfo: str) -> list[str]:
    m = re.search(r"^Features\s*:\s*(.+)$", cpuinfo, flags=re.M)
    return sorted(set(m.group(1).split())) if m else []


def clusters(cores: list[dict], names: dict[tuple[int, int], str]) -> list[dict]:
    """Group cores by (design, max clock), fastest first."""
    groups: OrderedDict[tuple, list[int]] = OrderedDict()
    for c in sorted(cores, key=lambda c: (-c.get("max_khz", 0), c["cpu"])):
        key = (c.get("implementer"), c.get("part"), c.get("max_khz"))
        groups.setdefault(key, []).append(c["cpu"])
    out = []
    for (imp, part, khz), cpus in groups.items():
        design = names.get((imp, part)) if imp is not None else None
        out.append(
            {
                "cores": len(cpus),
                "cpus": cpus,
                "implementer": f"0x{imp:02x}" if imp is not None else None,
                "part": f"0x{part:03x}" if part is not None else None,
                "design": design or (f"{IMPLEMENTERS.get(imp, hex(imp))}:0x{part:03x}" if imp is not None else None),
                "design_source": "linux cputype.h" if design else "unlisted in linux cputype.h",
                "max_mhz": round(khz / 1000, 1) if khz else None,
                "efficiency_class": core_class(imp, part) == "efficiency",
                "core_class": core_class(imp, part),
            }
        )
    return out


def ddr(props: dict[str, str]) -> dict:
    """What the bootloader says about DRAM, where the vendor exposes it (Pixels do)."""
    out = {}
    raw = props.get("ro.boot.hardware.ddr")
    if raw:
        parts = raw.split(",")
        out = {"raw": raw, "size": parts[0], "vendor": parts[1] if len(parts) > 1 else None}
        if len(parts) > 2:
            out["type"] = parts[2]
    for k in ("ro.boot.ddr_size", "ro.boot.ddr_info", "ro.boot.ddr_type", "ro.vendor.boot.ddr_type"):
        if k in props:
            out.setdefault("props", {})[k] = props[k]
    return out


NPU_FAMILIES = {
    "qualcomm-qnn-snpe": ("qnn", "snpe", "htp", "hexagon"),
    "mediatek-neuron-apusys": ("neuron", "apusys", "apuware"),
    "google-edgetpu": ("edgetpu", "darwinn"),
    "samsung-enn-eden": ("libenn", "eden_", "enn_aidl", "samsung_slsi"),
}


def npu_runtimes(listing: str) -> dict[str, list[str]]:
    """Vendor NPU runtime libraries present on the phone, grouped by family.

    The probe's filter also catches every library with "input" in its name; those are
    dropped here rather than in the probe, so the raw listing stays complete.
    """
    out: dict[str, list[str]] = {}
    for lib in sorted({line.strip() for line in listing.splitlines() if line.strip()}):
        low = lib.lower()
        # "input" libraries and ArcSoft's camera HDR library (libhdraid.npu.arcsoft.so, on
        # Samsung and Xiaomi alike) match the probe's broad filter but are not NPU runtimes.
        if "input" in low or "arcsoft" in low:
            continue
        for family, keys in NPU_FAMILIES.items():
            if any(k in low for k in keys):
                out.setdefault(family, []).append(lib)
                break
    return out


def vulkan_device(vkjson: str) -> str | None:
    m = re.search(r'"deviceName"\s*:\s*"([^"]+)"', vkjson)
    return m.group(1) if m else None


def thermal_hal(dump: str) -> dict:
    """Thermal status and the HAL's current temperatures, by sensor name.

    `dumpsys thermalservice` prints a "Cached temperatures" block (the last values pushed to
    listeners, possibly minutes old) before "Current temperatures from HAL". Only the current
    block is a reading; the cached one is kept separately for diagnosis, never used.
    """
    status = re.search(r"^Thermal Status: (\d+)", dump, flags=re.M)

    def block(title: str) -> dict[str, dict]:
        m = re.search(rf"^{title}:\n((?:\s+.*\n?)*)", dump, flags=re.M)
        out: dict[str, dict] = {}
        for value, kind, name, st in re.findall(
            r"Temperature\{mValue=(-?[\d.]+), mType=(-?\d+), mName=([^,]+), mStatus=(-?\d+)\}", m.group(1) if m else ""
        ):
            out.setdefault(name, {"c": float(value), "type": int(kind), "status": int(st)})
        return out

    return {
        "status": int(status.group(1)) if status else None,
        "current": block("Current temperatures from HAL"),
        "cached": block("Cached temperatures"),
    }


def dumpsys_total_ram_kib(text: str) -> int | None:
    m = re.search(r"Total RAM:\s*([\d,]+)K", text)
    return int(m.group(1).replace(",", "")) if m else None


def _read(folder: Path, name: str) -> str:
    p = folder / name
    return p.read_text(errors="replace") if p.exists() else ""


def battery_temp_c(battery: str) -> float | None:
    m = re.search(r"^\s*temperature:\s*(-?\d+)", battery, flags=re.M)
    return int(m.group(1)) / 10 if m else None


def push_rate(push: str) -> float | None:
    m = re.search(r"push_256MiB_seconds=([\d.]+)", push)
    return round(256 / float(m.group(1)), 1) if m else None


def profile(probe_dir: Path, names: dict | None = None) -> dict:
    names = names or kernel_part_names()
    props = parse_getprop(_read(probe_dir, "getprop.txt"))
    mem = parse_meminfo(_read(probe_dir, "meminfo.txt"))
    cores = parse_cpus(_read(probe_dir, "cpus.txt"))
    cl = clusters(cores, names)
    domains = clock_domains(_read(probe_dir, "cpufreq_policies.txt"))
    # affected_cpus lists only online CPUs; related_cpus lists every CPU of a policy. A probe
    # with an offline core would undercount a domain, so the two must agree with the cores
    # present, or the topology is left unknown.
    related = sorted({tuple(c["related_cpus"]) for c in cores if c.get("related_cpus")})
    covered = sorted(cpu for d in domains for cpu in d["cpus"])
    if sorted(c["cpu"] for c in cores) != covered or sorted(tuple(d["cpus"]) for d in domains) != related:
        domains = []
    design_of = {cpu: c["design"] for c in cl for cpu in c["cpus"]}
    for d in domains:
        d["designs"] = sorted({design_of.get(cpu) or "?" for cpu in d["cpus"]})
    # Phones that render through ANGLE on Vulkan (Pixel 11) print no GLES line; their GPU
    # is left unreported here rather than guessed (probe v2 reads `cmd gpu vkjson`).
    gles = re.search(r"GLES: ([^\n]+)", _read(probe_dir, "surfaceflinger_gles.txt"))
    caps = _read(probe_dir, "capabilities.txt")
    return {
        "reported": {
            "manufacturer": props.get("ro.product.manufacturer"),
            "model": props.get("ro.product.model"),
            "device": props.get("ro.product.device"),
            "soc_manufacturer": props.get("ro.soc.manufacturer"),
            "soc_model": props.get("ro.soc.model"),
            "board_platform": props.get("ro.board.platform"),
            "hardware": props.get("ro.hardware"),
            "android": props.get("ro.build.version.release"),
            "sdk": props.get("ro.build.version.sdk"),
            "fingerprint": props.get("ro.build.fingerprint"),
            "security_patch": props.get("ro.build.version.security_patch"),
            "page_size": props.get("ro.boot.hardware.cpu.pagesize"),
            "abilist": props.get("ro.product.cpu.abilist"),
            # ExecuTorch's Android builds are arm64-v8a; a phone with a 32-bit-only userspace
            # (Galaxy A13 5G, despite Cortex-A76 cores) cannot run them.
            "arm64_userspace": "arm64-v8a" in (props.get("ro.product.cpu.abilist") or "").split(","),
        },
        "cpu": {
            "count": len(cores),
            # Clusters are clock domains (cpufreq policies): the cores that must run at one
            # frequency. Core tiers group cores by design and top clock. The two differ on some
            # chips (Tensor G5: 1+5+2 tiers, four domains), and a vendor's own description may
            # match either, so all three are kept and the label uses the domains.
            "topology": "+".join(str(len(d["cpus"])) for d in domains) if domains else None,
            "clock_domains": domains,
            "tiers": "+".join(str(c["cores"]) for c in cl),
            "clusters": cl,
            # One known efficiency core settles "no"; "yes" needs every core known.
            "all_big_cores": False
            if any(c["core_class"] == "efficiency" for c in cl)
            else None
            if not cl or any(c["core_class"] == "unknown" for c in cl)
            else True,
            "max_mhz": max((c["max_mhz"] or 0) for c in cl) if cl else None,
            "features": cpu_features(_read(probe_dir, "cpuinfo.txt")),
        },
        "gpu": {
            "gles": gles.group(1).strip() if gles else None,
            "vulkan_device": vulkan_device(_read(probe_dir, "vkjson.txt")),
        },
        "npu": {"runtimes_present": npu_runtimes(_read(probe_dir, "npu_libs.txt"))},
        "memory": {
            "mem_total_kib": mem.get("MemTotal"),
            "dumpsys_total_ram_kib": dumpsys_total_ram_kib(_read(probe_dir, "meminfo_dumpsys.txt")),
            "ddr": ddr(props),
        },
        "state_at_probe": {
            "battery_temp_c": battery_temp_c(_read(probe_dir, "battery.txt")),
            "thermal": thermal_hal(_read(probe_dir, "thermalservice.txt")),
            "fixed_performance_mode": _fixed_perf(caps),
        },
        "tooling": {"simpleperf": _simpleperf(caps)},
        "staging": {
            "adb_push_mib_s": push_rate(_read(probe_dir, "push.txt")),
            "hf_fetch": _read(probe_dir, "hf_fetch.txt").splitlines()[:1],
        },
    }


def _fixed_perf(caps: str) -> str | None:
    """'accepted' when the shell command printed nothing and exited 0; otherwise what it said.

    Accepted means only that the PowerHAL took the hint. Whether clocks hold steadier under
    it is a measurement (docs/METRICS.md), not something this flag shows.
    """
    m = re.search(r"== cmd power set-fixed-performance-mode-enabled true\n((?:(?!==|exit=).*\n)*)exit=(\d+)", caps)
    if not m:
        return None
    out, code = m.group(1).strip(), int(m.group(2))
    if code == 0 and not out:
        return "accepted"
    return f"exit={code} {out.splitlines()[0][:160] if out else ''}".strip()


def _simpleperf(caps: str) -> dict:
    paranoid = re.search(r"== perf_event_paranoid\n(-?\d+)", caps)
    version = re.search(r"Simpleperf version (\S+)", caps)
    return {
        "present": "/system/bin/simpleperf" in caps,
        "version": version.group(1) if version else None,
        "perf_event_paranoid": int(paranoid.group(1)) if paranoid else None,
    }
