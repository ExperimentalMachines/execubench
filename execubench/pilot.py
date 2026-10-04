"""What still blocks the P1 pilot from spending device minutes, read from its manifest.

`python -m execubench pilot check data/pilot/p1.yaml` prints each blocker and exits 1 while
there is any. Scheduling code for the pilot must call `blockers()` first and refuse on any.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPAT = ROOT / "data" / "runtime" / "compat-1.4.0-vs-1.5.1"


def worst_case_minutes(manifest: dict) -> int:
    return sum(
        len(e["devices"]) * len(e["models"]) * len(e["arms"]) * e["jobs_per_cell"] * e["job_timeout_min"]
        for e in manifest["experiments"].values()
    )


def workload_problems(name: str, w: dict, root: Path = ROOT) -> list[str]:
    out = []
    if w.get("prompts_file"):
        path = root / w["prompts_file"]
        if not path.exists():
            return [f"workload {name}: {w['prompts_file']} is missing"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != w.get("prompts_sha256"):
            out.append(f"workload {name}: {w['prompts_file']} does not match prompts_sha256")
        turns = sum(len(json.loads(line)["turns"]) for line in data.decode().splitlines() if line.strip())
        if not w.get("loop_until_killed") and w.get("requests") != turns * w.get("repeat_each", 1):
            out.append(f"workload {name}: requests {w.get('requests')} is not {turns} turns x repeat_each")
    if w.get("kind") == "speed" and not (w.get("source_text") and w.get("source_text_sha256")):
        out.append(f"workload {name}: the speed track's source text is not pinned")
    return out


def blockers(manifest: dict, root: Path = ROOT) -> list[str]:
    from .harness import run

    out = []
    if not (manifest.get("runtime") or {}).get("apk_sha256"):
        out.append("runtime.apk_sha256 is null: no ExecuServe benchmark build is pinned")
    if not run.ORCHESTRATION_IMPLEMENTED:
        out.append("the harness orchestration is not implemented (execubench/harness/run.py)")
    for name, w in manifest["workloads"].items():
        out += workload_problems(name, w, root)
    for key, m in manifest["models"].items():
        stem = Path(m["file"]).stem
        compare = COMPAT / f"{stem}.compare.json"
        if not compare.exists():
            out.append(f"model {key}: no host compatibility report ({compare.relative_to(ROOT)})")
    worst = worst_case_minutes(manifest)
    if worst > manifest["max_device_minutes"]:
        out.append(f"worst case {worst} device minutes exceeds max_device_minutes {manifest['max_device_minutes']}")
    if not manifest.get("ledger"):
        out.append("no spend ledger named: the pilot-wide ceiling cannot be enforced across runs")
    return out
