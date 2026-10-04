"""One Device Farm job: identify the phone, stage the model, check the server contract, run the
job's requests with the sampler going, and write job.json, requests.jsonl and samples.jsonl.

    python -m execubench.harness.run --manifest data/pilot/p1.yaml --model <key> --out "$DEVICEFARM_LOG_DIR"

The P1 pilot uses the manifest's fixed smoke prompts; the frozen dataset tracks plug into the
same `request_record` once suite v1 exists (phase P2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from ..semantics import DECODE_MIN_MS, DECODE_MIN_STEPS, PREFILL_MIN_MS, clock_ok
from .records import config_sha256, write_json_atomic

# Features the ExecuServe benchmark build must advertise (docs/EXECUSERVE-CONTRACT.md).
REQUIRED_FEATURES = {
    "cache_off",
    "native_max_new_tokens",
    "capacity_guard",
    "raw_runner_stats",
    "native_monotonic",
    "prompt_echo",
}
# False until staging, start, cooldown, warm-up, the request loop and finalisation are wired
# (docs/PLAN.md 6.1); execubench.pilot refuses to schedule while it is.
ORCHESTRATION_IMPLEMENTED = False


class ContractMissing(RuntimeError):
    pass


def require_contract(base_url: str, key: str, apk_sha256: str | None) -> dict:
    """The server's benchmark capabilities, or a refusal: no device minutes on a build that
    cannot report what METRICS needs, or on a build nobody pinned."""
    if not apk_sha256:
        raise ContractMissing("no pinned ExecuServe APK sha256 (runtime.apk_sha256 in the manifest)")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/execuserve/capabilities", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        caps = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as error:  # noqa: BLE001 - any failure means the contract is not there
        raise ContractMissing(f"no benchmark capabilities endpoint: {error}") from error
    bench = caps.get("benchmark") or {}
    missing = REQUIRED_FEATURES - set(bench.get("features", []))
    if bench.get("contract") != 1 or missing:
        raise ContractMissing(f"ExecuServe build lacks benchmark contract 1 features: {sorted(missing)}")
    if bench.get("apk_sha256") != apk_sha256:
        raise ContractMissing(f"ExecuServe build {bench.get('apk_sha256')} is not the pinned {apk_sha256}")
    return caps


def request_record(job_id: str, seq: int, track: str, item: dict, reply, memory: dict, state: dict) -> dict:
    """A request record per docs/METRICS.md, computed from the runner's raw stats and the
    native monotonic stage times in ExecuServe's `x_execuserve` extension."""
    ext = reply.extension
    run = ext["runner_stats"]
    mono = ext["monotonic"]
    prefill_ms = run["prompt_eval_end_ms"] - run["inference_start_ms"]
    ttft_ms = run["first_token_ms"] - run["inference_start_ms"]
    decode_ms = run["inference_end_ms"] - run["prompt_eval_end_ms"]
    steps = run["generated_tokens"]
    mono_ttft = (mono["first_sampled_token_ns"] - mono["call_start_ns"]) / 1e6
    mono_decode = (mono["call_end_ns"] - mono["first_sampled_token_ns"]) / 1e6
    prompt = ext["prompt_text"]
    return {
        "job_id": job_id,
        "seq": seq,
        "track": track,
        "item": {**item, "prompt_text": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        "timings": {
            "runner": run,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "e2e_ms": round((reply.done_ns - reply.sent_ns) / 1e6),
            "client_ttft_ms": round((reply.first_byte_ns - reply.sent_ns) / 1e6) if reply.first_byte_ns else None,
            "monotonic": {
                "start_to_first_token_ms": mono_ttft,
                "first_token_to_end_ms": mono_decode,
                "total_ms": (mono["call_end_ns"] - mono["call_start_ns"]) / 1e6,
            },
            "native": {
                k: mono[k]
                for k in ("call_start_ns", "first_sampled_token_ns", "first_callback_ns", "call_end_ns")
                if k in mono
            },
            "clock_disagreement": not (clock_ok(ttft_ms, mono_ttft) and clock_ok(decode_ms, mono_decode)),
            "prefill_tps": None if prefill_ms < PREFILL_MIN_MS else run["prompt_tokens"] / (prefill_ms / 1000),
            "decode_tps": None if decode_ms < DECODE_MIN_MS or steps < DECODE_MIN_STEPS else steps / (decode_ms / 1000),
        },
        "tokens": {
            "prefill": run["prompt_tokens"],
            "cached": ext.get("cached_tokens", 0),
            "decode_steps": steps,
            "sampled": steps + 1,
        },
        "memory": memory,
        "state": state,
        "host": {"sent_ns": reply.sent_ns, "first_byte_ns": reply.first_byte_ns, "done_ns": reply.done_ns},
        "output": {"text": reply.text, "tool_calls": reply.tool_calls, "finish_reason": reply.finish_reason or "stop"},
        "status": "ok",
        "error": None,
    }


class BadSample(ValueError):
    """A sample that cannot be placed in time, or that contradicts another."""


def memory_summary(samples: list[dict], sent_ns: int, done_ns: int, job_id: str) -> dict:
    """Sampled peak and time-weighted mean of VmRSS over one request of job `job_id`.

    Only this job's successful memory reads that started and ended inside [sent_ns, done_ns]
    count (a read that straddles a boundary belongs to neither request). The mean treats VmRSS
    as constant from each read to the next, and the last read as holding until done_ns; the part
    of the request before the first read is not covered, and `coverage` says what fraction was.
    A memory sample without a valid read interval, from another job, or contradicting another
    read at the same instant raises BadSample: it is never placed by guesswork.
    """
    window = []
    for s in samples:
        if s.get("kind") != "memory" or s.get("read_error") or s.get("vm_rss_kib") is None:
            continue
        if s.get("job_id") != job_id:
            raise BadSample(f"memory sample from job {s.get('job_id')!r}, not {job_id!r}")
        try:
            t0, t, t1 = s["t_start_ns"], s["t_ns"], s["t_end_ns"]
        except KeyError as error:
            raise BadSample(f"memory sample without its read interval: {error}") from error
        if not t0 <= t <= t1:
            raise BadSample(f"memory sample interval out of order: {t0} <= {t} <= {t1} fails")
        if t0 >= sent_ns and t1 <= done_ns:
            window.append(s)
    window.sort(key=lambda s: s["t_ns"])
    # Two reads with one timestamp carry no duration between them; agreeing ones collapse.
    fields = ("vm_rss_kib", "rss_anon_kib", "rss_file_kib", "vm_hwm_kib")
    deduped: dict[int, dict] = {}
    for s in window:
        seen = deduped.get(s["t_ns"])
        if seen is not None and any(seen.get(k) != s.get(k) for k in fields):
            raise BadSample(f"two memory reads at {s['t_ns']} disagree")
        deduped[s["t_ns"]] = s
    window = [deduped[t] for t in sorted(deduped)]
    if not window or done_ns <= sent_ns:
        return {"rss_sampled_peak_mib": None, "rss_mean_mib": None, "samples": 0, "null_reason": "no_samples_in_window"}
    ends = [s["t_ns"] for s in window[1:]] + [done_ns]
    spans = [max(0, end - s["t_ns"]) for s, end in zip(window, ends, strict=True)]
    covered = sum(spans)
    rss = [s["vm_rss_kib"] / 1024 for s in window]
    mean = sum(r * w for r, w in zip(rss, spans, strict=True)) / covered if covered else rss[-1]
    peak = max(window, key=lambda s: s["vm_rss_kib"])
    return {
        "rss_sampled_peak_mib": max(rss),
        "rss_mean_mib": mean,
        "rss_anon_mib": peak["rss_anon_kib"] / 1024 if peak.get("rss_anon_kib") is not None else None,
        "rss_file_mib": peak["rss_file_kib"] / 1024 if peak.get("rss_file_kib") is not None else None,
        "samples": len(window),
        "coverage": covered / (done_ns - sent_ns),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="execubench.harness.run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True, help="Key of the model in the manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = p.parse_args(argv)

    import yaml

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    key = os.environ.get("EXECUSERVE_KEY", "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": os.environ.get("DEVICEFARM_JOB_ID", "local"),
        "model": manifest["models"][args.model],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finalized": False,
    }
    write_json_atomic(out / "job.json", job)
    try:
        require_contract(args.base_url, key, (manifest.get("runtime") or {}).get("apk_sha256"))
    except ContractMissing as error:
        job.update({"outcome": "failed", "failure": str(error), "finished_utc": datetime.now(UTC).isoformat()})
        write_json_atomic(out / "job.json", job)
        print(error, file=sys.stderr)
        return 3
    # Staging, server start, cooldown, warm-up, the request loop and finalisation follow
    # docs/PLAN.md 6.1 and are wired only once an ExecuServe build with contract 1 exists to
    # test them against. Until then a job that gets this far fails loudly, never reports success.
    job.update(
        {
            "config_sha256": config_sha256(job),
            "outcome": "failed",
            "failure": "harness orchestration not implemented (phase P1)",
            "finished_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json_atomic(out / "job.json", job)
    print(job["failure"], file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
