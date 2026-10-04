"""A spend ledger: worst-case device minutes reserved against one budget, across every run.

`--max-device-minutes` bounds one run; a ledger bounds a whole phase (the P1 pilot's
`max_device_minutes`) across experiments, retries and re-runs. A reservation is written, and
fsynced, before the run is scheduled and is never released: a run that used less than its worst
case still counts its worst case, which is the conservative direction.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path


class OverBudget(RuntimeError):
    pass


def reserved(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["worst_minutes"] for line in path.read_text().splitlines() if line.strip())


def reserve(path: Path, budget_minutes: float, name: str, worst_minutes: float) -> float:
    """Reserve `worst_minutes` under `name`, or raise OverBudget; returns the new total.

    The ledger is locked while it is read and appended, so two scheduling commands cannot both
    spend the last minutes.
    """
    if worst_minutes <= 0:
        raise ValueError("a reservation must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        lines = [json.loads(line) for line in f.read().splitlines() if line.strip()]
        if any(entry["name"] == name for entry in lines):
            raise OverBudget(f"{name!r} already holds a reservation in {path}")
        total = sum(entry["worst_minutes"] for entry in lines) + worst_minutes
        if total > budget_minutes:
            raise OverBudget(
                f"reserving {worst_minutes:.0f} worst-case minutes for {name!r} would bring {path} to "
                f"{total:.0f}, above its budget of {budget_minutes:.0f}"
            )
        entry = {"name": name, "worst_minutes": worst_minutes, "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return total
