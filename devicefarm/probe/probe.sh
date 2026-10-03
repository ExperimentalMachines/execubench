#!/usr/bin/env bash
# Device profile probe (v3.1): what the phone says about itself, read over adb from the Device
# Farm test host. Nothing here is inferred; every file is a raw dump, and parsing happens
# later on our own machine (`execubench devicefarm pull`), so a parser fix never needs a new
# device run.
#
# The committed probe data came from the exact scripts archived beside it
# (data/devices/probe/<run>/probe.sh). v3.1 differs from v3 only in treating the thermal
# service as optional below Android 10, where it does not exist. Changes in v3 from v2:
# no pipelines on the phone (output is filtered on the host, so exit codes are real), the
# push is verified by size before a rate is derived, time_in_state readability is recorded,
# and the script exits non-zero when a required dump is missing.
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
FAILED=()

# One device command, its raw output in a file, its real exit status in the index. Required
# dumps that fail make the job fail; optional ones only record that they failed.
run() {
  local need=$1 name=$2; shift 2
  "$@" > "$OUT/$name" 2>&1
  local code=$?
  echo "$name exit=$code" >> "$OUT/_index.txt"
  if [ "$need" = required ] && { [ $code -ne 0 ] || [ ! -s "$OUT/$name" ]; }; then FAILED+=("$name"); fi
}

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/_started_utc.txt"
{
  echo "probe_version=3.1"
  echo "host_os=$(. /etc/os-release && echo "$PRETTY_NAME") ($(uname -r))"
  echo "adb=$(adb version | head -1)"
  echo "device_arn=${DEVICEFARM_DEVICE_ARN:-}"
  echo "device_name=${DEVICEFARM_DEVICE_NAME:-}"
  echo "device_os=${DEVICEFARM_DEVICE_OS_VERSION:-}"
  echo "run_arn=${DEVICEFARM_RUN_ARN:-}"
} > "$OUT/_host.txt"

run required getprop.txt S getprop
run required uname.txt S uname -a
run required cpuinfo.txt S cat /proc/cpuinfo
run required meminfo.txt S cat /proc/meminfo
run required cpu_present.txt S cat /sys/devices/system/cpu/present

# Per core: clocks, capacity (the scheduler's own big/little ranking), clock domain and the
# MIDR, which names the core design without trusting a marketing name.
run required cpus.txt S 'for c in /sys/devices/system/cpu/cpu[0-9]*; do
  echo "== ${c##*/}"
  for f in cpufreq/cpuinfo_max_freq cpufreq/cpuinfo_min_freq cpufreq/scaling_max_freq \
           cpufreq/scaling_cur_freq cpufreq/scaling_governor cpufreq/related_cpus \
           cpufreq/scaling_available_frequencies cpu_capacity topology/cluster_id \
           topology/physical_package_id topology/core_siblings_list regs/identification/midr_el1; do
    printf "%s=" "$f"; cat "$c/$f" 2>/dev/null || echo "<unreadable>"
  done
done'
run required cpufreq_policies.txt S 'for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "== $p"; for f in affected_cpus related_cpus cpuinfo_max_freq cpuinfo_min_freq scaling_governor; do
  printf "%s=" "$f"; cat "$p/$f" 2>/dev/null || echo "<unreadable>"; done; done'
run optional time_in_state.txt S 'for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "== $p"; cat "$p/stats/time_in_state" 2>&1 | head -3; done'

# GPU: the GLES renderer where the driver prints one, and the Vulkan device name, which
# phones that render through ANGLE (Pixel 11) also report. Filtered on the host.
run optional surfaceflinger.raw S dumpsys SurfaceFlinger
grep -i -E 'GLES|vulkan' "$OUT/surfaceflinger.raw" | head -20 > "$OUT/surfaceflinger_gles.txt"; rm -f "$OUT/surfaceflinger.raw"
run optional vkjson.raw S cmd gpu vkjson
grep -E '"(deviceName|driverVersion|apiVersion|vendorID|deviceID)"' "$OUT/vkjson.raw" | head -10 > "$OUT/vkjson.txt"; rm -f "$OUT/vkjson.raw"
run optional gpu_sysfs.txt S 'for f in /sys/class/kgsl/kgsl-3d0/gpu_model /sys/class/kgsl/kgsl-3d0/max_gpuclk \
  /sys/class/kgsl/kgsl-3d0/devfreq/max_freq /sys/kernel/gpu/gpu_model /sys/kernel/gpu/gpu_max_clock \
  /sys/class/misc/mali0/device/gpuinfo; do printf "%s=" "$f"; cat "$f" 2>/dev/null || echo "<unreadable>"; done'

# NPU: userspace cannot name an NPU, but it can list which vendor runtimes and device nodes
# exist, which is what a later QNN, NeuroPilot or LiteRT backend needs to know.
run optional npu_libs.raw S ls /vendor/lib64 /system/lib64 /system_ext/lib64
grep -i -E 'htp|qnn|neuron|apusys|edgetpu|darwinn|eden|enn|tpu|npu|hexagon|snpe' "$OUT/npu_libs.raw" | sort -u > "$OUT/npu_libs.txt"; rm -f "$OUT/npu_libs.raw"
run optional npu_dev.raw S ls /dev
grep -i -E 'apusys|edgetpu|eden|npu|fastrpc|adsprpc|cdsp|tpu|vertex' "$OUT/npu_dev.raw" > "$OUT/npu_dev.txt"; rm -f "$OUT/npu_dev.raw"
run optional meminfo_dumpsys.raw S dumpsys meminfo
grep -E 'Total RAM|Free RAM|Used RAM|ZRAM' "$OUT/meminfo_dumpsys.raw" > "$OUT/meminfo_dumpsys.txt"; rm -f "$OUT/meminfo_dumpsys.raw"

# Counters: system-wide counting failed on every phone in v2 (three different errors), so v3
# also tries user-mode only. Per-process counting needs a profileable app and is P1's job.
run optional simpleperf_list.txt S simpleperf list hw
run optional simpleperf_idle.txt S simpleperf stat -a --per-core -e cpu-cycles,instructions --duration 1
run optional simpleperf_idle_user.txt S simpleperf stat -a --per-core -e cpu-cycles:u --duration 1

# Thermal and power state. The thermal dump's "Current temperatures from HAL" block is the
# reading; its "Cached temperatures" block can be minutes old (the parser keeps them apart).
run required battery.txt S dumpsys battery
# The thermal service exists from Android 10 (SDK 29); older phones cannot report it, which
# is a property of the phone, not a failed collection.
SDK=$(S getprop ro.build.version.sdk | tr -d '\r')
if [ "${SDK:-0}" -ge 29 ]; then
  run required thermalservice.txt S dumpsys thermalservice
else
  run optional thermalservice.txt S dumpsys thermalservice
fi
run optional thermal_zones.txt S 'for z in /sys/class/thermal/thermal_zone*; do printf "%s %s %s\n" "${z##*/}" "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"; done'
run optional power_supply.txt S 'for f in /sys/class/power_supply/battery/temp /sys/class/power_supply/battery/capacity \
  /sys/class/power_supply/battery/status /sys/class/power_supply/usb/online; do
  printf "%s=" "$f"; cat "$f" 2>/dev/null || echo "<unreadable>"; done'
run optional power.raw S dumpsys power
grep -i -E 'mWakefulness|mIsPowered|mBatteryLevel|mScreenOn|Fixed|sustained|LowPower' "$OUT/power.raw" | head -40 > "$OUT/power.txt"; rm -f "$OUT/power.raw"

# Can a benchmark ask for steadier clocks without root? Each attempt is recorded, then undone.
{
  echo "== cmd power set-fixed-performance-mode-enabled true"; S cmd power set-fixed-performance-mode-enabled true; echo "exit=$?"
  echo "== cmd power set-fixed-performance-mode-enabled false"; S cmd power set-fixed-performance-mode-enabled false; echo "exit=$?"
  echo "== simpleperf"; S ls -la /system/bin/simpleperf; S simpleperf --version
  echo "== perf_event_paranoid"; S cat /proc/sys/kernel/perf_event_paranoid
  echo "== id"; S id
  echo "== storage"; S df -h /data/local/tmp /sdcard
} > "$OUT/capabilities.txt" 2>&1

# Staging cost: 256 MiB pushed and timed. The rate is derived only when the push exits 0 and
# the size on the phone matches. Zeros are an upper bound on staging speed only if adb
# compressed them; adb 1.0.39 on these hosts predates adb's compression support.
BYTES=268435456
head -c $BYTES /dev/urandom > /tmp/push.bin
S mkdir -p /data/local/tmp/execubench >/dev/null
t0=$(date +%s.%N); A push /tmp/push.bin /data/local/tmp/execubench/push.bin > /tmp/push.log 2>&1; code=$?; t1=$(date +%s.%N)
tail -1 /tmp/push.log > "$OUT/push.txt"
size=$(S stat -c %s /data/local/tmp/execubench/push.bin | tr -d '\r')
echo "push_exit=$code bytes_expected=$BYTES bytes_on_device=$size" >> "$OUT/push.txt"
if [ $code -eq 0 ] && [ "$size" = "$BYTES" ]; then
  echo "push_256MiB_seconds=$(echo "$t1 - $t0" | bc)" >> "$OUT/push.txt"
else
  FAILED+=("push.txt")
fi
S rm -f /data/local/tmp/execubench/push.bin >/dev/null
rm -f /tmp/push.bin /tmp/push.log

# Host egress: can the host fetch from Hugging Face, and how fast (one public tokenizer).
URL=https://huggingface.co/experimentalmachines/Qwen3-0.6B-ExecuTorch/resolve/main/tokenizer.json
curl -sS -L --fail -o /tmp/tok.json -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total} speed_Bps=%{speed_download}\n' "$URL" > "$OUT/hf_fetch.txt" 2>&1
sha256sum /tmp/tok.json >> "$OUT/hf_fetch.txt" 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/_finished_utc.txt"
echo "probe done: $(ls "$OUT" | wc -l) files"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "required collection failed: ${FAILED[*]}" | tee "$OUT/_failed.txt"
  exit 1
fi
