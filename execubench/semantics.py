"""Checks JSON Schema cannot express: formulas, count relationships, hashes, percentile rules
and cross-record links. `execubench validate` runs them on every collected run and summary.

Each check returns readable error strings; an empty list means the record is consistent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# docs/METRICS.md, "Percentiles and intervals": minimum n for each percentile to be published.
PERCENTILE_MIN_N = {"p25": 5, "p50": 5, "p75": 5, "p95": 20, "p99": 100}
DECODE_MIN_MS, DECODE_MIN_STEPS, PREFILL_MIN_MS = 250, 16, 50
RATE_TOLERANCE = 0.01  # relative; rates are recomputed from integer milliseconds


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RATE_TOLERANCE * max(abs(a), abs(b), 1e-9)


def request(rec: dict) -> list[str]:
    out: list[str] = []
    if rec.get("status") != "ok":
        return out
    t, tok, item = rec["timings"], rec["tokens"], rec["item"]
    run = t.get("runner", {})
    if tok["sampled"] != tok["decode_steps"] + 1:
        out.append(f"sampled {tok['sampled']} != decode_steps {tok['decode_steps']} + 1")
    if run:
        if tok["prefill"] != run["prompt_tokens"]:
            out.append("tokens.prefill differs from runner prompt_tokens")
        if tok["decode_steps"] != run["generated_tokens"]:
            out.append("tokens.decode_steps differs from runner generated_tokens")
        if t["ttft_ms"] != run["first_token_ms"] - run["inference_start_ms"]:
            out.append("ttft_ms is not first_token_ms - inference_start_ms")
        if t["decode_ms"] != run["inference_end_ms"] - run["prompt_eval_end_ms"]:
            out.append("decode_ms is not inference_end_ms - prompt_eval_end_ms")
        prefill_ms = run["prompt_eval_end_ms"] - run["inference_start_ms"]
        if prefill_ms < PREFILL_MIN_MS:
            if t.get("prefill_tps") is not None:
                out.append(f"prefill_tps must be null under {PREFILL_MIN_MS} ms")
        elif t.get("prefill_tps") is None or not _close(t["prefill_tps"], run["prompt_tokens"] / (prefill_ms / 1000)):
            out.append("prefill_tps does not match prompt_tokens / prefill interval")
    if t["decode_ms"] < DECODE_MIN_MS or tok["decode_steps"] < DECODE_MIN_STEPS:
        if t.get("decode_tps") is not None:
            out.append("decode_tps must be null for short decodes")
    elif t.get("decode_tps") is None or not _close(t["decode_tps"], tok["decode_steps"] / (t["decode_ms"] / 1000)):
        out.append("decode_tps does not match decode_steps / decode_ms")
    text = item.get("prompt_text")
    if text is not None and hashlib.sha256(text.encode()).hexdigest() != item.get("prompt_sha256"):
        out.append("prompt_sha256 is not the sha256 of prompt_text")
    host = rec.get("host", {})
    if host and host["done_ns"] < host["sent_ns"]:
        out.append("host done_ns precedes sent_ns")
    mem = rec["memory"]
    if mem.get("rss_mean_mib") is not None and mem.get("rss_sampled_peak_mib") is not None:
        if mem["rss_mean_mib"] > mem["rss_sampled_peak_mib"]:
            out.append("rss_mean_mib exceeds rss_sampled_peak_mib")
    return out


def percentiles(name: str, p: dict) -> list[str]:
    out = []
    values = []
    for key, minimum in PERCENTILE_MIN_N.items():
        v = p[key]
        if p["n"] < minimum and v is not None:
            out.append(f"{name}.{key} published with n={p['n']} < {minimum}")
        if v is not None:
            values.append((key, v))
    for (ka, a), (kb, b) in zip(values, values[1:], strict=False):
        if a > b:
            out.append(f"{name}: {ka} {a} > {kb} {b}")
    if values and p.get("min") is not None and values[0][1] < p["min"]:
        out.append(f"{name}: {values[0][0]} below min")
    if values and p.get("max") is not None and values[-1][1] > p["max"]:
        out.append(f"{name}: {values[-1][0]} above max")
    return out


def summary(rec: dict) -> list[str]:
    out = []
    q = rec.get("quality") or {}
    if q.get("attempted") is not None and q.get("correct") is not None:
        if q["correct"] > q["attempted"]:
            out.append("quality.correct exceeds attempted")
        elif (
            q.get("accuracy") is not None
            and q["attempted"]
            and not _close(q["accuracy"], q["correct"] / q["attempted"])
        ):
            out.append("quality.accuracy is not correct / attempted")
    split = q.get("answer_in_context")
    if split:
        for part in ("in_context", "not_in_context"):
            if split[f"{part}_correct"] > split[f"{part}_attempted"]:
                out.append(f"answer_in_context.{part}_correct exceeds attempted")
    for group in ("timing", "memory", "thermal"):
        for name, value in (rec.get(group) or {}).items():
            if isinstance(value, dict) and "method" in value:
                out += percentiles(f"{group}.{name}", value)
    if len(rec.get("device_units", [])) > len(rec.get("jobs", [])):
        out.append("more device units than jobs")
    return out


def _jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            rows.append((number, json.loads(line)))
    return rows


def run_folder(folder: Path, schema_errors, device_ids: set[str]) -> list[str]:
    """One pulled job folder holding job.json, requests.jsonl and samples.jsonl."""
    out = []
    job_path = folder / "job.json"
    job = json.loads(job_path.read_text())
    out += [f"{job_path}: {e}" for e in schema_errors("run.schema.json", job)]
    if device_ids and job.get("device_id") not in device_ids:
        out.append(f"{job_path}: device_id {job.get('device_id')!r} not in devices.json")
    seqs: set[int] = set()
    req_path = folder / "requests.jsonl"
    if req_path.exists():
        for number, rec in _jsonl(req_path):
            where = f"{req_path}:{number}"
            out += [f"{where}: {e}" for e in schema_errors("request.schema.json", rec)]
            out += [f"{where}: {e}" for e in request(rec)]
            if rec.get("job_id") != job.get("job_id"):
                out.append(f"{where}: job_id differs from job.json")
            if rec.get("seq") in seqs:
                out.append(f"{where}: duplicate seq {rec.get('seq')}")
            seqs.add(rec.get("seq"))
    if seqs and seqs != set(range(len(seqs))):
        out.append(f"{req_path}: seq values are not 0..{len(seqs) - 1}")
    sample_path = folder / "samples.jsonl"
    if sample_path.exists():
        for number, rec in _jsonl(sample_path):
            where = f"{sample_path}:{number}"
            out += [f"{where}: {e}" for e in schema_errors("sample.schema.json", rec)]
            if rec.get("seq") is not None and rec["seq"] not in seqs:
                out.append(f"{where}: seq {rec['seq']} has no request")
    return out
