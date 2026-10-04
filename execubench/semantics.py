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
# docs/METRICS.md, clock cross-check: runner and native monotonic durations agree within this.
CLOCK_TOLERANCE_MS, CLOCK_TOLERANCE_REL = 5.0, 0.01
NATIVE_KEYS = ("call_start_ns", "first_sampled_token_ns", "first_callback_ns", "call_end_ns")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RATE_TOLERANCE * max(abs(a), abs(b), 1e-9)


def clock_ok(runner_ms: float, mono_ms: float) -> bool:
    return abs(runner_ms - mono_ms) <= CLOCK_TOLERANCE_MS + CLOCK_TOLERANCE_REL * max(runner_ms, mono_ms)


def request(rec: dict, raw: bool = False) -> list[str]:
    """Semantic checks of one request; a record too malformed to check says so instead of raising.

    `raw=True` is for records as collected (restricted storage): they must carry the prompt text,
    because only the publication exporter may remove it.
    """
    try:
        return _request(rec, raw)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError) as error:
        return [f"malformed record, semantic checks not possible: {type(error).__name__} {error}"]


def _timing(t: dict, host: dict) -> list[str]:
    """Every derived duration and flag recomputed from the raw coordinates it came from."""
    out = []
    native = t.get("native") or {}
    if all(k in native for k in NATIVE_KEYS):
        points = [native[k] for k in NATIVE_KEYS]
        if not (points[0] <= points[1] <= points[3] and points[1] <= points[2] <= points[3]):
            out.append("native timestamps out of order (start <= first token <= first callback, <= end)")
        mono = t.get("monotonic") or {}
        derived = {
            "start_to_first_token_ms": (points[1] - points[0]) / 1e6,
            "first_token_to_end_ms": (points[3] - points[1]) / 1e6,
            "total_ms": (points[3] - points[0]) / 1e6,
        }
        for key, value in derived.items():
            if key not in mono or abs(mono[key] - value) > 1e-6:
                out.append(f"monotonic.{key} is not derived from the native timestamps")
        if t.get("ttft_ms") is not None and t.get("decode_ms") is not None:
            agree = clock_ok(t["ttft_ms"], derived["start_to_first_token_ms"]) and clock_ok(
                t["decode_ms"], derived["first_token_to_end_ms"]
            )
            if t.get("clock_disagreement") is not (not agree):
                out.append("clock_disagreement does not match the runner and native durations")
    if host:
        if t.get("e2e_ms") != round((host["done_ns"] - host["sent_ns"]) / 1e6):
            out.append("e2e_ms is not done_ns - sent_ns")
        first = host.get("first_byte_ns")
        want = None if first is None else round((first - host["sent_ns"]) / 1e6)
        if t.get("client_ttft_ms") != want:
            out.append("client_ttft_ms is not first_byte_ns - sent_ns")
    return out


def _request(rec: dict, raw: bool = False) -> list[str]:
    out: list[str] = []
    item = rec.get("item") or {}
    # The restricted marker is a publication marker: no raw record may carry it, whatever its
    # status. A successful raw record also needs its prompt (a failed one may have none, if it
    # failed before the prompt was rendered).
    if raw and item.get("replay") == "restricted":
        out.append("a raw record cannot be replay restricted; only the publication exporter sets it")
    if raw and rec.get("status") == "ok" and not item.get("prompt_text"):
        out.append("a raw record must carry its prompt_text; only the publication exporter removes it")
    if item.get("replay") == "restricted" and item.get("prompt_text") is not None:
        out.append("replay restricted but prompt_text is still present")
    state = rec.get("state") or {}
    if state.get("thermal_status_max") is not None and state.get("thermal_event") is not (
        state["thermal_status_max"] > 0
    ):
        out.append("thermal_event does not match thermal_status_max > 0")
    if rec.get("status") != "ok":
        return out
    if rec.get("error"):
        out.append("status ok with an error message")
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
    if host:
        points = [host["sent_ns"], host.get("first_byte_ns"), host["done_ns"]]
        points = [x for x in points if x is not None]
        if points != sorted(points):
            out.append("host timestamps out of order (sent <= first byte <= done)")
    if not all(k in (t.get("native") or {}) for k in NATIVE_KEYS):
        out.append("a successful record needs all four native timestamps")
    out += _timing(t, host)
    if item.get("max_tokens") is not None and tok["sampled"] > item["max_tokens"]:
        out.append(f"sampled {tok['sampled']} tokens exceeds the cap max_tokens {item['max_tokens']}")
    mem = rec["memory"]
    if mem.get("rss_mean_mib") is not None and mem.get("rss_sampled_peak_mib") is not None:
        if mem["rss_mean_mib"] > mem["rss_sampled_peak_mib"]:
            out.append("rss_mean_mib exceeds rss_sampled_peak_mib")
    return out


def percentiles(name: str, p: dict, minimum_value: float | None = 0.0) -> list[str]:
    out = []
    values = []
    if p["n"] == 0 and any(p.get(k) is not None for k in ("mean", "min", "max", *PERCENTILE_MIN_N)):
        out.append(f"{name}: values published with n=0")
    lo, mean, hi = p.get("min"), p.get("mean"), p.get("max")
    if lo is not None and hi is not None and lo > hi:
        out.append(f"{name}: min {lo} > max {hi}")
    if mean is not None and ((lo is not None and mean < lo) or (hi is not None and mean > hi)):
        out.append(f"{name}: mean {mean} outside [min, max]")
    for key in ("mean", "min", "max", *PERCENTILE_MIN_N):
        v = p.get(key)
        if v is not None and minimum_value is not None and v < minimum_value:
            out.append(f"{name}.{key} {v} is negative")
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
    try:
        return _summary(rec)
    except (KeyError, TypeError, ZeroDivisionError) as error:
        return [f"malformed summary, semantic checks not possible: {type(error).__name__} {error}"]


# (rate, numerator, denominator) in quality.counts; docs/METRICS.md "Quality" defines each.
RATE_COUNTS = (
    ("tool_call_valid_rate", "tool_call_valid", "tool_call_rows"),
    ("tool_call_correct_rate", "tool_call_correct", "tool_call_rows"),
    ("tool_decision_vs_label_rate", "tool_decision_matches_label", "tool_decision_rows"),
    ("tool_decision_vs_own_rate", "tool_decision_matches_own", "tool_decision_own_rows"),
)


def _summary(rec: dict) -> list[str]:
    out = []
    q = rec.get("quality") or {}
    if q.get("attempted") == 0 and q.get("accuracy") is not None:
        out.append("quality.accuracy must be null when nothing was attempted")
    # Every rate needs its own positive denominator and must equal the ratio of its counts:
    # without them it cannot be checked or bounded, and over zero rows it means nothing.
    if q.get("accuracy") is not None and not (isinstance(q.get("attempted"), int) and q["attempted"] > 0):
        out.append("quality.accuracy published without a positive quality.attempted")
    counts = q.get("counts") or {}
    for rate, numerator, denominator in RATE_COUNTS:
        n, d = counts.get(numerator), counts.get(denominator)
        if n is not None and d is not None and n > d:
            out.append(f"quality.counts.{numerator} exceeds {denominator}")
        if q.get(rate) is None:
            continue
        if not (isinstance(d, int) and d > 0 and isinstance(n, int)):
            out.append(f"quality.{rate} published without a positive quality.counts.{denominator}")
        elif not _close(q[rate], n / d):
            out.append(f"quality.{rate} is not counts.{numerator} / counts.{denominator}")
    if q.get("retrieval_gain") is not None:
        d = counts.get("paired_rows")
        a, b = counts.get("search_tool_correct_paired"), counts.get("closed_book_correct_paired")
        if not (isinstance(d, int) and d > 0 and isinstance(a, int) and isinstance(b, int)):
            out.append("quality.retrieval_gain published without paired counts")
        elif max(a, b) > d:
            out.append("quality.counts paired correct answers exceed paired_rows")
        elif not abs(q["retrieval_gain"] - (a - b) / d) <= 1e-9 + RATE_TOLERANCE * abs((a - b) / d):
            out.append(
                "quality.retrieval_gain is not (search_tool_correct_paired - closed_book_correct_paired) / paired_rows"
            )
    if q.get("accuracy") is not None and q.get("correct") is None:
        out.append("quality.accuracy published without quality.correct")
    if q.get("ci95") is not None:
        lo, hi = q["ci95"]
        if lo > hi:
            out.append("quality.ci95 lower bound above upper bound")
        elif q.get("accuracy") is not None and not lo <= q["accuracy"] <= hi:
            out.append("quality.accuracy outside its ci95")
    for part in ("format_failures", "truncated"):
        if q.get(part) is not None and q.get("attempted") is not None and q[part] > q["attempted"]:
            out.append(f"quality.{part} exceeds attempted")
    if q.get("eligible") is not None and q.get("attempted") is not None and q["attempted"] > q["eligible"]:
        out.append("quality.attempted exceeds eligible")
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
        # Search-tool rows without a call are in neither subset, so the subsets sum to at most the total.
        if (
            q.get("attempted") is not None
            and split["in_context_attempted"] + split["not_in_context_attempted"] > (q["attempted"])
        ):
            out.append("answer_in_context subsets exceed quality.attempted")
        if (
            q.get("correct") is not None
            and split["in_context_correct"] + split["not_in_context_correct"] > (q["correct"])
        ):
            out.append("answer_in_context correct subsets exceed quality.correct")
    agreement = rec.get("agreement") or {}
    for key in ("identical_vs_reference", "identical_vs_host"):
        if agreement.get(key) is not None and agreement.get("compared") is not None:
            if agreement[key] > agreement["compared"]:
                out.append(f"agreement.{key} exceeds compared")
    interval = rec.get("interval") or {}
    if interval.get("lo") is not None and interval.get("hi") is not None and interval["lo"] > interval["hi"]:
        out.append("interval.lo above interval.hi")
    if interval.get("per_job_medians") is not None and len(interval["per_job_medians"]) != len(rec.get("jobs", [])):
        out.append("interval.per_job_medians does not have one value per job")
    excluded = sum((rec.get("excluded") or {}).values())
    for name, value in (rec.get("timing") or {}).items():
        if isinstance(value, dict) and value.get("n", 0) + excluded > rec.get("n", 0):
            out.append(f"timing.{name}.n plus excluded requests exceeds the cell's n")
    for group in ("timing", "memory", "thermal"):
        for name, value in (rec.get(group) or {}).items():
            if isinstance(value, dict) and "method" in value:
                # Durations, rates and memory cannot be negative; temperatures can.
                out += percentiles(f"{group}.{name}", value, None if group == "thermal" else 0.0)
    if len(rec.get("device_units", [])) > len(rec.get("jobs", [])):
        out.append("more device units than jobs")
    return out


def _jsonl(path: Path) -> list[tuple[int, object]]:
    """(line number, value) per non-empty line; a line that is not JSON becomes an error string
    wrapped so it can never pass as a record."""
    rows: list[tuple[int, object]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append((number, json.loads(line)))
            except json.JSONDecodeError as error:
                rows.append((number, _Invalid(str(error))))
    return rows


class _Invalid(str):
    pass


def sample(rec: dict) -> list[str]:
    out = []
    t0, t, t1 = rec.get("t_start_ns"), rec.get("t_ns"), rec.get("t_end_ns")
    if None not in (t0, t, t1) and (t0 > t1 or t != (t0 + t1) // 2):
        out.append("t_ns is not the midpoint of [t_start_ns, t_end_ns]")
    if rec.get("read_error") and any(
        rec.get(k) is not None for k in ("vm_rss_kib", "cpu_cur_khz", "thermal_status", "temps_c")
    ):
        out.append("a failed read carries values")
    return out


def job(rec: dict) -> list[str]:
    """A finalized job's hashes must be the hashes of what it records."""
    from .harness.records import config_sha256, protocol_sha256

    out = []
    if rec.get("finalized"):
        if rec.get("config_sha256") and rec["config_sha256"] != config_sha256(rec):
            out.append("config_sha256 is not the hash of the recorded configuration")
        proto = rec.get("protocol") or {}
        if proto.get("protocol_sha256") and proto["protocol_sha256"] != protocol_sha256(proto):
            out.append("protocol_sha256 is not the hash of the recorded protocol")
    return out


def _records(path: Path, schema: str, schema_errors, out: list[str]):
    """Yield (where, record) for each line that is a JSON object passing its schema; every other
    line is reported and skipped, so semantic checks never run on a structurally invalid record."""
    for number, rec in _jsonl(path):
        where = f"{path}:{number}"
        if isinstance(rec, _Invalid):
            out.append(f"{where}: not JSON: {rec}")
            continue
        if not isinstance(rec, dict):
            out.append(f"{where}: not a JSON object")
            continue
        errors = schema_errors(schema, rec)
        out += [f"{where}: {e}" for e in errors]
        if not errors:
            yield where, rec


def run_folder(folder: Path, schema_errors, device_ids: set[str], raw: bool = False) -> list[str]:
    """One job folder holding job.json, requests.jsonl and samples.jsonl. `raw=True` for records
    as collected (they must keep their prompts; see `request`)."""
    out: list[str] = []
    job_path = folder / "job.json"
    try:
        job_ = json.loads(job_path.read_text())
    except json.JSONDecodeError as error:
        return [f"{job_path}: not JSON: {error}"]
    if not isinstance(job_, dict):
        return [f"{job_path}: not a JSON object"]
    job_errors = schema_errors("run.schema.json", job_)
    out += [f"{job_path}: {e}" for e in job_errors + (job(job_) if not job_errors else [])]
    complete = job_.get("outcome") == "complete"
    if complete:
        for needed in ("requests.jsonl", "samples.jsonl"):
            if not (folder / needed).exists():
                out.append(f"{folder / needed}: missing for a complete job")
    if device_ids and job_.get("device_id") not in device_ids:
        out.append(f"{job_path}: device_id {job_.get('device_id')!r} not in devices.json")
    seqs: set[int] = set()
    req_path = folder / "requests.jsonl"
    if req_path.exists():
        for where, rec in _records(req_path, "request.schema.json", schema_errors, out):
            out += [f"{where}: {e}" for e in request(rec, raw=raw)]
            if rec.get("job_id") != job_.get("job_id"):
                out.append(f"{where}: job_id differs from job.json")
            if rec["seq"] in seqs:
                out.append(f"{where}: duplicate seq {rec['seq']}")
            seqs.add(rec["seq"])
    if seqs and seqs != set(range(len(seqs))):
        out.append(f"{req_path}: seq values are not 0..{len(seqs) - 1}")
    planned = job_.get("planned_requests")
    if complete and (not seqs or (isinstance(planned, int) and len(seqs) != planned)):
        out.append(f"{req_path}: a complete job recorded {len(seqs)} requests, planned {planned}")
    elif isinstance(planned, int) and len(seqs) > planned:
        out.append(f"{req_path}: {len(seqs)} requests, more than the {planned} planned")
    sample_path = folder / "samples.jsonl"
    if sample_path.exists():
        for where, rec in _records(sample_path, "sample.schema.json", schema_errors, out):
            out += [f"{where}: {e}" for e in sample(rec)]
            if rec.get("job_id") != job_.get("job_id"):
                out.append(f"{where}: job_id differs from job.json")
            if rec.get("seq") is not None and rec["seq"] not in seqs:
                out.append(f"{where}: seq {rec['seq']} has no request")
    return out
