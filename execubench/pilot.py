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


def _execution_problems(report: dict) -> list[str]:
    """Evidence that the runner actually ran every prompt: the fixed output cap, and runner stats
    with a positive prompt, at least one generated step and ordered timestamps. Empty text can be
    a real answer; missing execution evidence cannot."""
    from . import host

    out = []
    for key, kind in (("executorch", str), ("pte", dict), ("tokenizer", dict), ("results", list)):
        if not isinstance(report.get(key), kind):
            out.append(f"{key} is missing or not a {kind.__name__}")
    for key in ("pte", "tokenizer"):
        if isinstance(report.get(key), dict) and not isinstance(report[key].get("sha256"), str):
            out.append(f"{key}.sha256 is missing")
    if out:
        return out
    if report.get("max_new_tokens") != host.MAX_NEW_TOKENS:
        out.append(f"max_new_tokens {report.get('max_new_tokens')!r}, not {host.MAX_NEW_TOKENS}")
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        return out + ["no results"]
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            out.append(f"prompt {i}: not an object")
            continue
        stats, pieces = row.get("stats"), row.get("pieces")
        if not isinstance(stats, dict) or not isinstance(pieces, list) or not pieces:
            out.append(f"prompt {i}: no pieces or runner stats")
            continue
        if not all(isinstance(p, str) for p in pieces):
            out.append(f"prompt {i}: pieces are not strings")
            continue
        keys = ("prompt_tokens", "generated_tokens", "inference_start_ms", "prompt_eval_end_ms", "inference_end_ms")
        # Integers only (booleans and floats are not token counts or the runner's millisecond clock).
        if not all(type(stats.get(k)) is int for k in keys):
            out.append(f"prompt {i}: runner stats are missing or not integers")
            continue
        if not (
            stats["prompt_tokens"] > 0
            and 1 <= stats["generated_tokens"] <= host.MAX_NEW_TOKENS
            and 0 <= stats["inference_start_ms"] <= stats["prompt_eval_end_ms"] <= stats["inference_end_ms"]
        ):
            out.append(f"prompt {i}: runner stats do not show a completed generation")
    return out


def compat_problems(model: dict, runtime: str, folder: Path = COMPAT) -> list[str]:
    """The model's host compatibility evidence, checked rather than assumed: both host reports
    exist, are for this exact file and tokenizer, ran the export runtime and the benchmark runtime
    on the fixed prompts, and the committed comparison is what `host.compare` derives from them.
    Disagreeing outputs are acceptable evidence (they are the finding); missing or mismatched
    evidence is not."""
    from . import host

    stem = Path(model["file"]).stem
    paths = {
        "export": folder / f"{stem}.executorch-{model['export_executorch']}.json",
        "runtime": folder / f"{stem}.executorch-{runtime}.json",
        "compare": folder / f"{stem}.compare.json",
    }
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        return [f"host compatibility evidence missing: {missing}"]
    try:
        a, b, c = (json.loads(paths[k].read_text()) for k in ("export", "runtime", "compare"))
    except json.JSONDecodeError as error:
        return [f"host compatibility evidence is not JSON: {error}"]
    if not all(isinstance(x, dict) for x in (a, b, c)):
        return ["host compatibility evidence is not a set of JSON objects"]
    out = []
    for name, report in (("export", a), ("runtime", b)):
        out += [f"{name} report: {p}" for p in _execution_problems(report)]
    if out:
        return out  # structure first: nothing below runs on a malformed report
    for name, report, version in (("export", a, model["export_executorch"]), ("runtime", b, runtime)):
        if report.get("executorch") != version:
            out.append(f"{name} report ran executorch {report.get('executorch')}, not {version}")
        if report.get("pte", {}).get("sha256") != model["sha256"]:
            out.append(f"{name} report is for another .pte than the pinned sha256")
        if report.get("tokenizer", {}).get("sha256") != model["tokenizer_sha256"]:
            out.append(f"{name} report used another tokenizer than the pinned sha256")
        if [r.get("prompt") for r in report.get("results", [])] != list(host.PROMPTS):
            out.append(f"{name} report did not run the fixed host prompts")
    if not out:
        try:
            if c != host.compare(a, b):
                out.append("the committed comparison is not what host.compare derives from the two reports")
        except (KeyError, TypeError, ValueError) as error:
            out.append(f"the two reports cannot be compared: {error}")
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
        out += [f"model {key}: {p}" for p in compat_problems(m, manifest["runtime"]["executorch"])]
    worst = worst_case_minutes(manifest)
    if worst > manifest["max_device_minutes"]:
        out.append(f"worst case {worst} device minutes exceeds max_device_minutes {manifest['max_device_minutes']}")
    if not manifest.get("ledger"):
        out.append("no spend ledger named: the pilot-wide ceiling cannot be enforced across runs")
    return out
