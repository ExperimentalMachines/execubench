"""The probe parser, checked against committed raw dumps from real Device Farm phones."""

import json
from pathlib import Path

import pytest

from execubench import devicefarm, devices

PROBE = Path(__file__).resolve().parent.parent / "data" / "devices" / "probe" / "2026-10-04-v2"


def prof(slug: str) -> dict:
    return devices.profile(PROBE / slug / "artifacts" / "probe")


def test_tensor_g5_has_four_clock_domains_but_three_tiers():
    p = prof("GLBW0-android16")
    assert p["reported"]["soc_model"] == "Tensor G5"
    assert p["cpu"]["topology"] == "1+3+2+2"
    assert p["cpu"]["tiers"] == "1+5+2"
    assert [c["design"] for c in p["cpu"]["clusters"]] == ["CORTEX_X4", "CORTEX_A725", "CORTEX_A520"]
    assert p["cpu"]["all_big_cores"] is False
    assert p["cpu"]["max_mhz"] == 3782.0


def test_tensor_g6_is_seven_cores_all_big():
    p = prof("GUJ0N-android17")
    assert p["cpu"]["count"] == 7
    assert p["cpu"]["topology"] == "1+4+2"
    assert {c["design"] for c in p["cpu"]["clusters"]} == {"C1_ULTRA", "C1_PRO"}
    assert p["cpu"]["all_big_cores"] is True
    assert p["gpu"]["gles"] is None  # renders through ANGLE: no GLES line
    assert p["gpu"]["vulkan_device"] == "PowerVR C-Series CXTP-48-1536 MC1"


def test_snapdragon_8_elite_reports_qualcomm_part_0x001():
    p = prof("SM-S938U1-android15")
    assert p["reported"]["soc_model"] == "SM8750"
    assert p["cpu"]["topology"] == "2+6"
    assert {(c["implementer"], c["part"]) for c in p["cpu"]["clusters"]} == {("0x51", "0x001")}
    assert p["cpu"]["max_mhz"] == 4473.6


def test_unlisted_part_is_named_by_vendor_and_number():
    p = prof("SM-S948U1-android16")
    assert {c["design"] for c in p["cpu"]["clusters"]} == {"QCOM:0x002"}
    assert all(c["design_source"] == "unlisted in linux cputype.h" for c in p["cpu"]["clusters"])


def test_snapdragon_8_gen_2_clock_domains_match_qualcomm_1_4_3():
    p = prof("2211133G-android13")
    assert p["cpu"]["topology"] == "1+4+3"
    assert p["cpu"]["tiers"] == "1+2+2+3"


def test_dimensity_9400_plus_clock():
    p = prof("SM-X730-android16")
    assert p["cpu"]["max_mhz"] == 3730.0
    assert p["cpu"]["all_big_cores"] is True


def test_pixel_bootloader_reports_dram():
    ddr = prof("G4QUR-android16")["memory"]["ddr"]
    assert ddr["type"] == "LPDDR5"
    assert ddr["size"] == "16GiB"


@pytest.mark.parametrize(
    "slug,family",
    [
        ("SM-S938U1-android15", "qualcomm-qnn-snpe"),
        ("SM-X730-android16", "mediatek-neuron-apusys"),
        ("GLBW0-android16", "google-edgetpu"),
        ("SM-A566U1-android15", "samsung-enn-eden"),
    ],
)
def test_one_npu_family_per_phone(slug, family):
    assert list(prof(slug)["npu"]["runtimes_present"]) == [family]


def test_fixed_performance_mode_parser():
    caps = "== cmd power set-fixed-performance-mode-enabled true\nexit=0\n== next\n"
    assert devices._fixed_perf(caps) == "accepted"
    caps = "== cmd power set-fixed-performance-mode-enabled true\nError: nope\nexit=255\n"
    assert devices._fixed_perf(caps) == "exit=255 Error: nope"


def test_scrub_replaces_serial_everywhere_and_drops_progress():
    files = {
        "probe/_host.txt": b"device_name=ABC123XYZ\nrun_arn=arn:aws:devicefarm:us-west-2:123456789012:run:x/y\n",
        "probe/getprop.txt": (
            b"[ro.serialno]: [ABC123XYZ]\n[ro.ril.oem.imei1]: [356789012345678]\n"
            b"[ro.boot.ap_serial]: [0x1A2B3C4D5E]\n[gsm.sim.preiccid_0]: [8901260123]\n"
            b"[ril.support.dynamic_imei]: [1]\n[ro.build.uuid]: [build-uuid-kept]\n"
        ),
        "probe/battery.txt": b"imei echo 356789012345678\n",
        "probe/push.txt": b"[  0%] /data/x\n[ 50%] /data/x\n/tmp/push.bin: 1 file pushed.\n",
    }
    out, unit = devicefarm.scrub(files, account="123456789012")
    assert unit == devicefarm.unit_hash("ABC123XYZ")
    joined = b"".join(out.values())
    for secret in (b"ABC123XYZ", b"356789012345678", b"0x1A2B3C4D5E", b"8901260123", b"123456789012"):
        assert secret not in joined
    # Flags and per-build ids are not identifiers and stay.
    assert b"[ril.support.dynamic_imei]: [1]" in out["probe/getprop.txt"]
    assert b"build-uuid-kept" in out["probe/getprop.txt"]
    assert out["probe/push.txt"] == b"/tmp/push.bin: 1 file pushed.\n"


def test_artifact_paths_cannot_escape():
    assert devicefarm._safe_member("Host_Machine_Files/$DEVICEFARM_LOG_DIR/probe/a.txt") == "probe/a.txt"
    for bad in ("Host_Machine_Files/$DEVICEFARM_LOG_DIR/../x", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError):
            devicefarm._safe_member(bad)


def test_unknown_core_design_makes_all_big_unknown():
    cores = [{"cpu": 0, "implementer": 0x41, "part": 0xFFF, "max_khz": 1000}]
    cl = devices.clusters(cores, {})
    assert cl[0]["core_class"] == "unknown"


def test_thermal_reading_comes_from_current_not_cached():
    dump = (
        "Thermal Status: 0\nCached temperatures:\n"
        "\tTemperature{mValue=90.0, mType=0, mName=BIG, mStatus=0}\n"
        "HAL Ready: true\nCurrent temperatures from HAL:\n"
        "\tTemperature{mValue=40.0, mType=0, mName=BIG, mStatus=0}\n"
    )
    t = devices.thermal_hal(dump)
    assert t["current"]["BIG"]["c"] == 40.0
    assert t["cached"]["BIG"]["c"] == 90.0


def test_committed_probe_data_carries_no_identifiers():
    for f in PROBE.parent.rglob("getprop.txt"):
        props = devices.parse_getprop(f.read_text(errors="replace"))
        for key, value in props.items():
            if devicefarm.IDENTIFIER_KEYS.match(key.encode()) and len(value) >= 6:
                assert value.startswith(("unit-", "<redacted>")), (f, key)


def test_committed_profiles_are_current():
    """profile.json files are generated; a parser change must regenerate them."""
    for folder in sorted(d for run in PROBE.parent.iterdir() if run.is_dir() for d in run.iterdir() if d.is_dir()):
        stored = json.loads((folder / "profile.json").read_text())
        assert stored == devices.profile(folder / "artifacts" / "probe"), folder.name
