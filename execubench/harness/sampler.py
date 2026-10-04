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
    m = re.search(r"^Current temperatures from HAL:\n((?:\s+.*\n?)*)", dump, flags=re.M)
    return {
        "thermal_status": hal["status"],
        "temps_c": {k: v["c"] for k, v in hal["current"].items()},
        # The raw lines behind temps_c, so a parser change can be checked against the reading.
        "raw": (m.group(1).strip() if m else None),
    }


FREQ_CMD = "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ${c##*/} $(cat $c/cpufreq/scaling_cur_freq); done"


class SamplerFailed(RuntimeError):
    pass


class Sampler:
    """Memory and clocks every `fast_s`, thermal every `slow_s`, on one thread.

    Each sample records when its read started and ended (host monotonic), so a request's samples
    are selected by time afterwards; a read that straddles a request's start or end belongs to
    neither. `stop()` waits for the thread to finish before the writer may be closed, and raises
    if the sampler failed in a way other than a missed adb read.
    """

    def __init__(self, adb: Adb, pid: int, writer: JsonlWriter, job_id: str, fast_s: float = 0.25, slow_s: float = 1.0):
        self.adb, self.pid, self.writer, self.job_id = adb, pid, writer, job_id
        self.fast_s, self.slow_s = fast_s, slow_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.missed = 0
        self.failure: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 60.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise SamplerFailed(f"sampler still running {timeout_s} s after stop")
        if self.failure is not None:
            raise SamplerFailed(f"sampler failed: {self.failure!r}") from self.failure

    def _read(self, kind: str, command: str, parse, timeout_s: float) -> None:
        t_start = time.monotonic_ns()
        text = self.adb.shell(command, timeout_s=timeout_s)
        t_end = time.monotonic_ns()
        self.writer.write(
            {
                "job_id": self.job_id,
                "seq": None,
                "t_ns": (t_start + t_end) // 2,
                "t_start_ns": t_start,
                "t_end_ns": t_end,
                "kind": kind,
                **parse(text),
            }
        )

    def _loop(self) -> None:
        next_slow = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                try:
                    self._read("memory", f"cat /proc/{self.pid}/status", parse_status, 5)
                    self._read("clock", FREQ_CMD, lambda t: {"cpu_cur_khz": parse_freqs(t) or None}, 5)
                    if now >= next_slow:
                        self._read("thermal", "dumpsys thermalservice", parse_thermal, 10)
                        next_slow = now + self.slow_s
                except AdbError:
                    # A missed read is a gap, never an invented value; the count is reported.
                    self.missed += 1
                self._stop.wait(max(0.0, self.fast_s - (time.monotonic() - now)))
        except BaseException as error:  # noqa: BLE001 - surfaced through stop()
            self.failure = error
