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


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reconcile_pull(path: Path) -> list[str]:
    """The pull manifest must exist, say complete and harness, and match the folder exactly:
    every job it lists present with every file at its recorded sha256 and size, nothing else."""
    manifest = path / "pull-manifest.json"
    if not manifest.exists():
        return [f"{manifest}: missing; a pulled run is validated against its manifest (or use --job-folder)"]
    m = json.loads(manifest.read_text())
    out = []
    if not m.get("complete"):
        out.append(f"{manifest}: the pull is not complete")
    if m.get("kind") != "harness":
        out.append(f"{manifest}: pulled as {m.get('kind')!r}, not as a harness run")
    jobs = m.get("jobs") or {}
    if not jobs:
        out.append(f"{manifest}: lists no jobs")
    for name, job in sorted(jobs.items()):
        artifacts = path / name / "artifacts"
        listed = job.get("files") or {}
        present = {p.relative_to(artifacts).as_posix() for p in artifacts.rglob("*") if p.is_file()}
        for rel in sorted(present - set(listed)):
            out.append(f"{artifacts / rel}: not in the pull manifest")
        for rel, meta in sorted(listed.items()):
            f = artifacts / rel
            if not f.is_file():
                out.append(f"{f}: listed in the pull manifest but missing")
            elif f.stat().st_size != meta.get("bytes") or _sha256(f) != meta.get("sha256"):
                out.append(f"{f}: differs from the sha256 or size in the pull manifest")
    for job_json in sorted(path.rglob("job.json")):
        rel = job_json.relative_to(path).parts
        if len(rel) != 3 or rel[0] not in jobs or rel[1] != "artifacts":
            out.append(f"{job_json}: not a job of the pull manifest")
    return out


def validate_runs(path: Path, root: Path = ROOT, standalone: bool = False) -> list[str]:
    """A pulled harness run as collected (restricted storage, outside this repository).

    The pull manifest is required and reconciled with the folder (`_reconcile_pull`), then every
    job folder's records are checked with the raw-record rules. `standalone=True` skips the
    manifest, for inspecting a job folder that did not come from a pull (`--job-folder`); it never
    stands in for a pulled run. Finding no job at all is an error, never an empty success.
    """
    from . import semantics

    out = [] if standalone else _reconcile_pull(path)
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
