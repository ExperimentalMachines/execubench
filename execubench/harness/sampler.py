"""Periodic readings of the server process and the phone, written to samples.jsonl.

Every sample carries the host's monotonic clock and the request being served, so a request's
samples can be selected by its host send and receive times (docs/METRICS.md, "Memory").
"""

from __future__ import annotations

import re
import threading
import time

from .adb import Adb, AdbError
from .records import JsonlWriter

STATUS_FIELDS = {"VmRSS": "vm_rss_kib", "RssAnon": "rss_anon_kib", "RssFile": "rss_file_kib", "VmHWM": "vm_hwm_kib"}


def parse_status(text: str) -> dict[str, int]:
    out = {}
    for key, field in STATUS_FIELDS.items():
        m = re.search(rf"^{key}:\s+(\d+) kB", text, flags=re.M)
        if m:
            out[field] = int(m.group(1))
    return out


def parse_freqs(text: str) -> dict[str, int]:
    """Output of `for c in cpu*; do echo $c $(cat .../scaling_cur_freq); done`."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^cpu(\d+) (\d+)$", text, flags=re.M)}


def parse_thermal(dump: str) -> dict:
    from ..devices import thermal_hal

    hal = thermal_hal(dump)
    return {"thermal_status": hal["status"], "temps_c": {k: v["c"] for k, v in hal["current"].items()}}


FREQ_CMD = "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ${c##*/} $(cat $c/cpufreq/scaling_cur_freq); done"


class Sampler:
    """Two loops on one thread: memory and clocks every `fast_s`, thermal every `slow_s`."""

    def __init__(self, adb: Adb, pid: int, writer: JsonlWriter, job_id: str, fast_s: float = 0.25, slow_s: float = 1.0):
        self.adb, self.pid, self.writer, self.job_id = adb, pid, writer, job_id
        self.fast_s, self.slow_s = fast_s, slow_s
        self.seq: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.errors = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _loop(self) -> None:
        next_slow = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                status = parse_status(self.adb.shell(f"cat /proc/{self.pid}/status", timeout_s=5))
                freqs = parse_freqs(self.adb.shell(FREQ_CMD, timeout_s=5))
                self.writer.write(
                    {
                        "job_id": self.job_id,
                        "seq": self.seq,
                        "t_ns": time.monotonic_ns(),
                        "kind": "memory_clock",
                        **status,
                        "cpu_cur_khz": freqs or None,
                    }
                )
                if now >= next_slow:
                    thermal = parse_thermal(self.adb.shell("dumpsys thermalservice", timeout_s=10))
                    self.writer.write(
                        {
                            "job_id": self.job_id,
                            "seq": self.seq,
                            "t_ns": time.monotonic_ns(),
                            "kind": "thermal",
                            **thermal,
                        }
                    )
                    next_slow = now + self.slow_s
            except AdbError:
                # A missed sample is recorded as a gap, never invented; the count is reported.
                self.errors += 1
            self._stop.wait(max(0.0, self.fast_s - (time.monotonic() - now)))
