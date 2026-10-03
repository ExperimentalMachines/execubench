#!/usr/bin/env bash
# Device profile probe: what the phone says about itself, read over adb from the Device Farm
# test host. Nothing here is inferred; every file is a raw dump, and parsing happens later on
# our own machine (`execubench devices parse`), so a parser fix never needs a new device run.
#
# Runs on the host (Amazon Linux 2, x86_64) in a custom test environment. Device Farm sets
# DEVICEFARM_DEVICE_UDID and DEVICEFARM_LOG_DIR. Locally: DEVICEFARM_DEVICE_UDID=<serial>
# DEVICEFARM_LOG_DIR=out bash probe.sh
set -u
UDID=${DEVICEFARM_DEVICE_UDID:?}
OUT=${DEVICEFARM_LOG_DIR:?}/probe
mkdir -p "$OUT"
A() { adb -s "$UDID" "$@"; }
S() { A shell "$@" 2>&1; }

# Record how each command went, so a missing file is told apart from an empty one.
run() {
  local name=$1; shift
  "$@" > "$OUT/$name" 2>&1
  echo "$name exit=$?" >> "$OUT/_index.txt"
}

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/_started_utc.txt"
{
  echo "host_os=$(. /etc/os-release && echo "$PRETTY_NAME") ($(uname -r))"
  echo "adb=$(adb version | head -1)"
  echo "device_arn=${DEVICEFARM_DEVICE_ARN:-}"
  echo "device_name=${DEVICEFARM_DEVICE_NAME:-}"
  echo "device_os=${DEVICEFARM_DEVICE_OS_VERSION:-}"
  echo "run_arn=${DEVICEFARM_RUN_ARN:-}"
} > "$OUT/_host.txt"

run getprop.txt S getprop
run uname.txt S uname -a
run cpuinfo.txt S cat /proc/cpuinfo
run meminfo.txt S cat /proc/meminfo
run cpu_present.txt S cat /sys/devices/system/cpu/present

# Per core: clocks, capacity (the scheduler's own big/little ranking), cluster and the
# MIDR, which names the core design (vendor and part number) without trusting a marketing name.
S 'for c in /sys/devices/system/cpu/cpu[0-9]*; do
  n=${c##*/}
  echo "== $n"
  for f in cpufreq/cpuinfo_max_freq cpufreq/cpuinfo_min_freq cpufreq/scaling_max_freq \
           cpufreq/scaling_cur_freq cpufreq/scaling_governor cpufreq/related_cpus \
           cpufreq/scaling_available_frequencies cpu_capacity topology/cluster_id \
           topology/physical_package_id topology/core_siblings_list regs/identification/midr_el1; do
    printf "%s=" "$f"; cat "$c/$f" 2>/dev/null || echo "<unreadable>"
  done
done' > "$OUT/cpus.txt"
echo "cpus.txt exit=$?" >> "$OUT/_index.txt"

S 'ls /sys/devices/system/cpu/cpufreq/ 2>&1; for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "== $p"; for f in affected_cpus cpuinfo_max_freq cpuinfo_min_freq scaling_governor; do
  printf "%s=" "$f"; cat "$p/$f" 2>/dev/null || echo "<unreadable>"; done; done' > "$OUT/cpufreq_policies.txt"

# GPU: the renderer string the driver reports, plus the vendor sysfs nodes where readable.
run surfaceflinger_gles.txt S "dumpsys SurfaceFlinger | grep -i -E 'GLES|vulkan' | head -20"
S 'for f in /sys/class/kgsl/kgsl-3d0/gpu_model /sys/class/kgsl/kgsl-3d0/max_gpuclk \
  /sys/class/kgsl/kgsl-3d0/devfreq/max_freq /sys/kernel/gpu/gpu_model /sys/kernel/gpu/gpu_max_clock \
  /sys/class/misc/mali0/device/gpuinfo /proc/gpufreq/gpufreq_opp_dump; do
  printf "%s=" "$f"; cat "$f" 2>/dev/null | head -5 || echo "<unreadable>"; done' > "$OUT/gpu_sysfs.txt"

# Thermal and power state: what a benchmark can later record, and whether shell may read it.
run battery.txt S dumpsys battery
run thermalservice.txt S dumpsys thermalservice
S 'for z in /sys/class/thermal/thermal_zone*; do printf "%s %s %s\n" "${z##*/}" "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"; done' > "$OUT/thermal_zones.txt"
S 'for f in /sys/class/power_supply/battery/temp /sys/class/power_supply/battery/capacity \
  /sys/class/power_supply/battery/status /sys/class/power_supply/usb/online; do
  printf "%s=" "$f"; cat "$f" 2>/dev/null || echo "<unreadable>"; done' > "$OUT/power_supply.txt"
run power.txt S "dumpsys power | grep -i -E 'mWakefulness|mIsPowered|mBatteryLevel|mScreenOn|Fixed|sustained|LowPower' | head -40"

# Can a benchmark ask for steadier clocks without root? Each attempt is recorded, then undone.
{
  echo "== cmd power set-fixed-performance-mode-enabled true"; S cmd power set-fixed-performance-mode-enabled true; echo "exit=$?"
  echo "== cmd power set-fixed-performance-mode-enabled false"; S cmd power set-fixed-performance-mode-enabled false; echo "exit=$?"
  echo "== simpleperf"; S 'ls -la /system/bin/simpleperf 2>&1; simpleperf --version 2>&1 | head -2'
  echo "== perf_event_paranoid"; S cat /proc/sys/kernel/perf_event_paranoid
  echo "== id"; S id
  echo "== storage"; S df -h /data/local/tmp /sdcard
} > "$OUT/capabilities.txt" 2>&1

# Staging cost: models are 0.1 to 2.5 GB, so the push rate decides how much of a 150-minute
# job goes to copying. 256 MiB of zeros, pushed and timed, then deleted.
dd if=/dev/zero of=/tmp/push.bin bs=1M count=256 status=none
S mkdir -p /data/local/tmp/execubench >/dev/null
t0=$(date +%s.%N); A push /tmp/push.bin /data/local/tmp/execubench/push.bin > "$OUT/push.txt" 2>&1; t1=$(date +%s.%N)
echo "push_256MiB_seconds=$(echo "$t1 - $t0" | bc)" >> "$OUT/push.txt"
S rm -f /data/local/tmp/execubench/push.bin >/dev/null
rm -f /tmp/push.bin

# Host egress: can the host fetch from Hugging Face, and how fast (one public tokenizer).
URL=https://huggingface.co/experimentalmachines/Qwen3-0.6B-ExecuTorch/resolve/main/tokenizer.json
curl -sS -L -o /tmp/tok.json -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total} speed_Bps=%{speed_download}\n' "$URL" > "$OUT/hf_fetch.txt" 2>&1
sha256sum /tmp/tok.json >> "$OUT/hf_fetch.txt" 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/_finished_utc.txt"
echo "probe done: $(ls "$OUT" | wc -l) files"
