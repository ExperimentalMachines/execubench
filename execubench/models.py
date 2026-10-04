"""The model inventory: every XNNPACK `.pte` we can benchmark, pinned to a Hub revision.

A file enters the inventory with three hashes that must agree: the Hub's LFS sha256, the
sha256 its export report recorded when execupack built it, and (on the device, at run time)
the sha256 of the bytes actually pushed. A disagreement anywhere drops the file from a run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ORGS = ("experimentalmachines",)
WINDOW = re.compile(r"-(\d+)k\.pte$")


def scan(token: str | None = None) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    out = []
    for org in ORGS:
        for m in api.list_models(author=org):
            info = api.model_info(m.id, files_metadata=True)
            ptes = [s for s in info.siblings if s.rfilename.endswith(".pte")]
            xnn = [s for s in ptes if s.rfilename.startswith("xnnpack/")]
            if not xnn:
                continue
            reports = {}
            for s in info.siblings:
                if s.rfilename.startswith("xnnpack/export-report-") and s.rfilename.endswith(".json"):
                    path = hf_hub_download(m.id, s.rfilename, revision=info.sha, token=token)
                    reports[s.rfilename] = json.loads(Path(path).read_text())
            by_file: dict[str, tuple] = {}
            conflicts: set[str] = set()
            for name, r in sorted(reports.items()):
                for entry in r.get("files", []):
                    previous = by_file.get(entry["path"])
                    # Two reports naming one file must agree on everything that defines it (hash,
                    # source, toolchain, tokenizer, recipe); otherwise neither is trusted.
                    if previous and provenance(previous[1], entry["path"]) != provenance(r, entry["path"]):
                        conflicts.add(entry["path"])
                    by_file.setdefault(entry["path"], (name, r))
            lfs = {x.rfilename: x.lfs.sha256 for x in info.siblings if x.lfs}
            plain = {x.rfilename for x in info.siblings}
            # Small tokenizers are stored in git, not LFS, so the Hub gives no sha256: hash them.
            for name in {r.get("tokenizer") for r in reports.values()} - set(lfs) - {None}:
                if name in plain:
                    path = hf_hub_download(m.id, name, revision=info.sha, token=token)
                    lfs[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for s in sorted(xnn, key=lambda s: s.rfilename):
                report_name, report = by_file.get(s.rfilename, (None, None))
                recorded = next(
                    (f["sha256"] for f in (report or {}).get("files", []) if f["path"] == s.rfilename), None
                )
                window = WINDOW.search(s.rfilename)
                out.append(
                    {
                        "repo": m.id,
                        "revision": info.sha,
                        "file": s.rfilename,
                        "bytes": s.size,
                        "sha256": s.lfs.sha256 if s.lfs else None,
                        "report": report_name,
                        "report_sha256": recorded,
                        "hashes_agree": bool(recorded) and s.lfs is not None and recorded == s.lfs.sha256,
                        "report_conflict": s.rfilename in conflicts,
                        "window": int(window.group(1)) * 1024 if window else None,
                        "qmode": (report or {}).get("recipe", {}).get("qmode"),
                        "executorch": (report or {}).get("toolchain", {}).get("executorch"),
                        "source_model": (report or {}).get("source", {}).get("id"),
                        "source_revision": (report or {}).get("source", {}).get("sha"),
                        "family": (report or {}).get("source", {}).get("family"),
                        "params": (report or {}).get("source", {}).get("total_params"),
                        # The report names the tokenizer relative to the repo root.
                        "tokenizer": (report or {}).get("tokenizer")
                        if (report or {}).get("tokenizer") in plain
                        else None,
                        "tokenizer_sha256": lfs.get((report or {}).get("tokenizer")),
                        "recipe": {
                            k: (report or {}).get("recipe", {}).get(k)
                            for k in (
                                "group_size",
                                "embedding_quantize",
                                "int4_codes",
                                "kv_cache_dtype",
                                "prefill_chunk",
                            )
                        },
                        # execupack's sizing model, per window: an estimate of .pte + fp32 KV cache
                        # + overhead, not a measurement (docs/PLAN.md section 3.2 quotes it).
                        "resident_estimate_bytes": {
                            str(t["context"]): t["device_resident_bytes"]
                            for t in (report or {}).get("window", {}).get("table", [])
                        },
                    }
                )
    return out


def provenance(report: dict, path: str) -> tuple:
    """What a report says defines one file; two reports naming the file must give the same."""
    entry = next((e for e in report.get("files", []) if e.get("path") == path), {})
    return (
        entry.get("sha256"),
        json.dumps(report.get("source", {}), sort_keys=True),
        json.dumps(report.get("toolchain", {}), sort_keys=True),
        report.get("tokenizer"),
        json.dumps(report.get("recipe", {}), sort_keys=True),
    )


HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


def problems(inv: list[dict], accepted: list[dict] | None = None) -> list[str]:
    """Every reason an inventory must not be accepted. The hashes are compared here, not read
    from the `hashes_agree` flag, so a hand-edited flag cannot hide a mismatch.

    With `accepted` (the inventory in use), a file it lists that the scan lost is a problem: a
    shrinking inventory must be accepted on purpose (`models scan --allow-removed`), never by
    accident. Every repo must also keep its 8k file, the window every track runs at.
    """
    if not inv:
        return ["the scan found no files"]
    out = []
    if accepted is not None:
        now = {(f.get("repo"), f.get("file")) for f in inv}
        for f in accepted:
            if (f.get("repo"), f.get("file")) not in now:
                out.append(f"{f.get('repo')}/{f.get('file')}: in the accepted inventory, missing from the scan")
    for repo in sorted({f.get("repo") for f in inv} - {f.get("repo") for f in inv if f.get("window") == 8192}):
        out.append(f"{repo}: no 8k file")
    seen: set[tuple] = set()
    for f in inv:
        where = f"{f.get('repo')}/{f.get('file')}"
        if (f.get("repo"), f.get("file")) in seen:
            out.append(f"{where}: listed twice")
        seen.add((f.get("repo"), f.get("file")))
        if f.get("report_conflict"):
            out.append(f"{where}: export reports disagree about this file")
        if not isinstance(f.get("source_revision"), str) or not HEX40.fullmatch(f["source_revision"]):
            out.append(f"{where}: source_revision missing or malformed")
        for key, pattern in (
            ("sha256", HEX64),
            ("report_sha256", HEX64),
            ("tokenizer_sha256", HEX64),
            ("revision", HEX40),
        ):
            if not isinstance(f.get(key), str) or not pattern.fullmatch(f[key]):
                out.append(f"{where}: {key} missing or malformed")
        if f.get("sha256") != f.get("report_sha256"):
            out.append(f"{where}: Hub sha256 differs from the export report's")
        if bool(f.get("hashes_agree")) != (f.get("sha256") == f.get("report_sha256") and f.get("sha256") is not None):
            out.append(f"{where}: hashes_agree flag does not match the hashes")
        for key in ("tokenizer", "source_model", "source_revision", "executorch", "window"):
            if not f.get(key):
                out.append(f"{where}: {key} missing")
    return out
