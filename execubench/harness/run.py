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

from ..semantics import DECODE_MIN_MS, DECODE_MIN_STEPS, PREFILL_MIN_MS
from .records import JsonlWriter, config_sha256, write_json_atomic

# Features the ExecuServe benchmark build must advertise (docs/EXECUSERVE-CONTRACT.md).
REQUIRED_FEATURES = {
    "cache_off",
    "native_max_new_tokens",
    "capacity_guard",
    "raw_runner_stats",
    "native_monotonic",
    "prompt_echo",
}
CLOCK_TOLERANCE_MS, CLOCK_TOLERANCE_REL = 5.0, 0.01


class ContractMissing(RuntimeError):
    pass


def require_contract(base_url: str, key: str) -> dict:
    """The server's benchmark capabilities, or a refusal: no device minutes on a build that
    cannot report what METRICS needs."""
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
    return caps


def _clock_ok(runner_ms: float, mono_ms: float) -> bool:
    return abs(runner_ms - mono_ms) <= CLOCK_TOLERANCE_MS + CLOCK_TOLERANCE_REL * max(runner_ms, mono_ms)


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
            "clock_disagreement": not (_clock_ok(ttft_ms, mono_ttft) and _clock_ok(decode_ms, mono_decode)),
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


def memory_summary(samples: list[dict], sent_ns: int, done_ns: int) -> dict:
    """Sampled peak and time-weighted mean of VmRSS between a request's host send and receive."""
    window = [
        s
        for s in samples
        if s.get("kind") == "memory_clock" and sent_ns <= s["t_ns"] <= done_ns and s.get("vm_rss_kib")
    ]
    if not window:
        return {"rss_sampled_peak_mib": None, "rss_mean_mib": None, "samples": 0, "null_reason": "no_samples_in_window"}
    rss = [s["vm_rss_kib"] / 1024 for s in window]
    if len(window) == 1:
        mean = rss[0]
    else:
        spans = [b["t_ns"] - a["t_ns"] for a, b in zip(window, window[1:], strict=False)]
        mean = sum(r * w for r, w in zip(rss, spans, strict=False)) / sum(spans)
    peak = max(window, key=lambda s: s["vm_rss_kib"])
    return {
        "rss_sampled_peak_mib": max(rss),
        "rss_mean_mib": mean,
        "rss_anon_mib": peak.get("rss_anon_kib", 0) / 1024 if peak.get("rss_anon_kib") is not None else None,
        "rss_file_mib": peak.get("rss_file_kib", 0) / 1024 if peak.get("rss_file_kib") is not None else None,
        "samples": len(window),
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
        require_contract(args.base_url, key)
    except ContractMissing as error:
        job.update({"outcome": "failed", "failure": str(error), "finished_utc": datetime.now(UTC).isoformat()})
        write_json_atomic(out / "job.json", job)
        print(error, file=sys.stderr)
        return 3
    # Staging, server start, warm-up and the request loop follow the order in docs/PLAN.md 6.1;
    # they are wired once the ExecuServe benchmark build exists to test them against.
    job["config_sha256"] = config_sha256(job)
    write_json_atomic(out / "job.json", job)
    JsonlWriter(out / "requests.jsonl").close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
