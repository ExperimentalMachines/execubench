"""A spend ledger: worst-case device minutes reserved against one budget, across every run.

`--max-device-minutes` bounds one run; a ledger bounds a whole phase (the P1 pilot's
`max_device_minutes`) across experiments, retries and re-runs. A reservation is written, and
fsynced, before the run is scheduled and is never released: a run that used less than its worst
case still counts its worst case, which is the conservative direction.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path


class OverBudget(RuntimeError):
    pass


def _finite_positive(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be a finite positive number, not {value!r}")
    return float(value)


def _entries(text: str) -> list[dict]:
    entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    for entry in entries:
        _finite_positive(entry.get("worst_minutes"), f"ledger entry {entry.get('name')!r}")
    return entries


def reserved(path: Path) -> float:
    return sum(e["worst_minutes"] for e in _entries(path.read_text())) if path.exists() else 0.0


def reserve(path: Path, budget_minutes: float, name: str, worst_minutes: float) -> float:
    """Reserve `worst_minutes` under `name`, or raise OverBudget; returns the new total.

    The ledger is locked while it is read and appended, so two scheduling commands cannot both
    spend the last minutes.
    """
    _finite_positive(budget_minutes, "the ledger budget")
    _finite_positive(worst_minutes, "a reservation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        lines = _entries(f.read())
        if any(entry["name"] == name for entry in lines):
            raise OverBudget(f"{name!r} already holds a reservation in {path}")
        total = sum(entry["worst_minutes"] for entry in lines) + worst_minutes
        if total > budget_minutes:
            raise OverBudget(
                f"reserving {worst_minutes:.0f} worst-case minutes for {name!r} would bring {path} to "
                f"{total:.0f}, above its budget of {budget_minutes:.0f}"
            )
        entry = {"name": name, "worst_minutes": worst_minutes, "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        f.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return total
