"""adb, called with argument lists (never a shell string on the host) and bounded timeouts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class AdbError(RuntimeError):
    """`code` is a fixed, identifier-free classification ("timeout" or "exit_<n>") that records may
    carry. The message also holds adb's stderr, which can name the device serial ("device
    '<serial>' not found"), so it is for the operator's log only and never goes into a record."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Adb:
    serial: str
    binary: str = "adb"
    timeout_s: float = 60.0

    def run(self, *args: str, timeout_s: float | None = None, check: bool = True) -> str:
        cmd = [self.binary, "-s", self.serial, *args]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s or self.timeout_s)
        except subprocess.TimeoutExpired as error:
            raise AdbError(f"timed out: {' '.join(args[:3])}", "timeout") from error
        if check and done.returncode != 0:
            raise AdbError(
                f"adb {' '.join(args[:3])} exited {done.returncode}: {done.stderr.strip()[:300]}",
                f"exit_{done.returncode}",
            )
        return done.stdout

    def shell(self, command: str, timeout_s: float | None = None, check: bool = True) -> str:
        """A command for the phone's shell. Callers pass fixed strings or quoted paths only."""
        return self.run("shell", command, timeout_s=timeout_s, check=check)
