"""Validate generated files against the JSON Schemas in schemas/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"


def validator(name: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    docs = {p.name: json.loads(p.read_text()) for p in SCHEMAS.glob("*.schema.json")}
    resources = []
    for file_name, doc in docs.items():
        res = Resource.from_contents(doc)
        # Reachable both by $id and by the relative file name used in "$ref".
        resources += [(doc["$id"], res), (file_name, res)]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(docs[name], registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER)


def errors_for(name: str, instance: dict) -> list[str]:
    v = validator(name)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(instance)]


def _device_ids(root: Path) -> set[str]:
    devices = root / "data" / "devices" / "devices.json"
    return {rec.get("id") for rec in json.loads(devices.read_text())["devices"]} if devices.exists() else set()


def validate_runs(path: Path, root: Path = ROOT) -> list[str]:
    """A pulled harness run as collected (restricted storage, outside this repository): every
    job folder's records with the raw-record rules, and the pull manifest must say complete.

    Accepts the pull layout (`<run>/<device>/artifacts/job.json`) or any folder holding job.json
    files. Finding no job at all is an error, never an empty success.
    """
    from . import semantics

    out = []
    manifest = path / "pull-manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if not m.get("complete"):
            out.append(f"{manifest}: the pull is not complete")
        if m.get("kind") != "harness":
            out.append(f"{manifest}: pulled as {m.get('kind')!r}, not as a harness run")
    job_dirs = sorted({job.parent for job in path.rglob("job.json")})
    if not job_dirs:
        return out + [f"{path}: no job.json found; nothing was validated"]
    device_ids = _device_ids(root)
    for job_dir in job_dirs:
        out += semantics.run_folder(job_dir, errors_for, device_ids, raw=True)
    return out


def validate_repo(root: Path = ROOT) -> list[str]:
    """Every file that has a schema, checked, plus the semantic checks: device records,
    schema examples, every collected job under data/runs/ (job.json, requests.jsonl,
    samples.jsonl, with line numbers) and every summary under data/results/."""
    from . import semantics

    out = []
    device_ids = _device_ids(root)
    devices = root / "data" / "devices" / "devices.json"
    if devices.exists():
        for rec in json.loads(devices.read_text())["devices"]:
            out += [f"devices.json {rec.get('id')}: {e}" for e in errors_for("device.schema.json", rec)]
    for example in sorted((SCHEMAS / "examples").glob("*.json")):
        schema = example.name.split(".")[0] + ".schema.json"
        instance = json.loads(example.read_text())
        out += [f"{example.name}: {e}" for e in errors_for(schema, instance)]
        check = {"request": semantics.request, "summary": semantics.summary, "sample": semantics.sample}.get(
            schema.split(".")[0]
        )
        if check:
            out += [f"{example.name}: {e}" for e in check(instance)]
    runs = root / "data" / "runs"
    job_dirs = {job.parent for job in runs.rglob("job.json")}
    for job_dir in sorted(job_dirs):
        out += semantics.run_folder(job_dir, errors_for, device_ids)
    for orphan in sorted({p.parent for p in runs.rglob("*.jsonl")} - job_dirs):
        out.append(f"{orphan}: records without a job.json")
    for path in sorted((root / "data" / "results").rglob("*.json")):
        rec = json.loads(path.read_text())
        out += [f"{path}: {e}" for e in errors_for("summary.schema.json", rec) + semantics.summary(rec)]
    return out
