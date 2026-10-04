"""The semantic checks catch the impossible records review found validating against the schemas."""

import copy
import json
from pathlib import Path

from execubench import schemas, semantics
from execubench.harness import records

EXAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "examples"
REQ = json.loads((EXAMPLES / "request.example.json").read_text())
SUMMARY = json.loads((EXAMPLES / "summary.example.json").read_text())


def test_examples_are_consistent():
    assert semantics.request(REQ) == []
    assert semantics.summary(SUMMARY) == []


def test_negative_and_inconsistent_requests_are_rejected():
    rec = copy.deepcopy(REQ)
    rec["timings"]["ttft_ms"] = -5
    assert schemas.errors_for("request.schema.json", rec)
    rec = copy.deepcopy(REQ)
    rec["tokens"]["sampled"] = 7
    assert any("sampled" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["output"]["finish_reason"] = "error"
    assert schemas.errors_for("request.schema.json", rec)
    rec = copy.deepcopy(REQ)
    rec["timings"]["decode_tps"] = 99.0
    assert any("decode_tps" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["item"]["prompt_text"] = "something else"
    assert any("prompt_sha256" in e for e in semantics.request(rec))


def test_impossible_summaries_are_rejected():
    s = copy.deepcopy(SUMMARY)
    s["n"] = -1
    assert schemas.errors_for("summary.schema.json", s)
    s = copy.deepcopy(SUMMARY)
    s["days"] = []
    assert schemas.errors_for("summary.schema.json", s)
    s = copy.deepcopy(SUMMARY)
    s["quality"] = {"accuracy": 0.5, "attempted": 2, "correct": 3}
    assert any("exceeds" in e for e in semantics.summary(s))
    s = copy.deepcopy(SUMMARY)
    s["timing"]["prefill_tps"].update({"n": 1, "p95": 1.0, "p99": 2.0})
    assert any("p95" in e for e in semantics.summary(s))
    s = copy.deepcopy(SUMMARY)
    s["timing"]["prefill_tps"].update({"p25": 120.0})
    assert any("p25" in e for e in semantics.summary(s))


SAMPLE = {"job_id": "example-00000", "seq": None, "t_ns": 5, "t_start_ns": 0, "t_end_ns": 10}


def _mem_raw(kib: int) -> str:
    return f"VmRSS:\t {kib} kB\n"


THERMAL_0 = "Thermal Status: 0\n"


def test_a_thermal_sample_needs_a_reading():
    # Otherwise valid: the only thing missing is the reading, so the thermal rule is what fails.
    good = {**SAMPLE, "kind": "thermal", "thermal_status": 0, "temps_c": None, "raw": THERMAL_0}
    assert schemas.errors_for("sample.schema.json", good) == []
    assert schemas.errors_for("sample.schema.json", {**SAMPLE, "kind": "thermal", "raw": THERMAL_0})
    # A failed read needs no reading, but must say why and carry no values.
    failed = {**SAMPLE, "kind": "thermal", "read_error": "unparseable", "raw": "x"}
    assert schemas.errors_for("sample.schema.json", failed) == [] and semantics.sample(failed) == []
    assert semantics.sample({**failed, "thermal_status": 1})


def test_sample_intervals_must_hold_their_midpoint():
    good = {**SAMPLE, "kind": "memory", "vm_rss_kib": 1, "raw": _mem_raw(1)}
    assert semantics.sample(good) == []
    assert semantics.sample({**good, "t_start_ns": 8, "t_end_ns": 2})  # reversed
    assert semantics.sample({**good, "t_start_ns": 0, "t_end_ns": 100, "t_ns": 0})  # inside, not the midpoint
    assert schemas.errors_for("sample.schema.json", {**good, "vm_rss_kib": None, "read_error": ""})


def test_timings_are_rederived_from_their_coordinates():
    rec = copy.deepcopy(REQ)
    rec["timings"]["e2e_ms"] = 1
    assert any("e2e_ms" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["timings"]["client_ttft_ms"] = 7
    assert any("client_ttft_ms" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["timings"]["monotonic"]["first_token_to_end_ms"] = 9000.0
    assert any("monotonic.first_token_to_end_ms" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["timings"]["native"]["call_end_ns"] = 9_000_000_000
    rec["timings"]["monotonic"].update({"first_token_to_end_ms": 7999.0, "total_ms": 9000.0})
    assert any("clock_disagreement" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    rec["timings"]["native"]["first_callback_ns"] = 1
    assert any("out of order" in e for e in semantics.request(rec))
    rec = copy.deepcopy(REQ)
    del rec["timings"]["native"]
    assert schemas.errors_for("request.schema.json", rec)
    assert any("native" in e for e in semantics.request(rec))


def test_thermal_flag_must_match_the_status():
    rec = copy.deepcopy(REQ)
    rec["state"]["thermal_status_max"] = 6
    assert any("thermal_event" in e for e in semantics.request(rec))


def test_restricted_replay_rules():
    rec = copy.deepcopy(REQ)
    rec["item"]["replay"] = "restricted"
    assert schemas.errors_for("request.schema.json", rec)  # still holds its prompt text
    rec["item"]["prompt_text"] = None
    assert schemas.errors_for("request.schema.json", rec) == []
    assert any("raw record" in e for e in semantics.request(rec, raw=True))
    assert semantics.request(REQ, raw=True) == []
    # Not on a failed raw record either.
    failed = {**copy.deepcopy(REQ), "status": "error", "error": "load failed"}
    failed["item"] = {**failed["item"], "replay": "restricted", "prompt_text": None}
    assert any("cannot be replay restricted" in e for e in semantics.request(failed, raw=True))


def test_impossible_statistics_are_rejected():
    s = copy.deepcopy(SUMMARY)
    s["quality"] = {"accuracy": 1}
    assert any("positive quality.attempted" in e for e in semantics.summary(s))
    s["quality"] = {"attempted": 0, "tool_call_valid_rate": 1}
    assert any("tool_call_valid_rate" in e for e in semantics.summary(s))
    # A positive overall count is not the metric's own population.
    s["quality"] = {"attempted": 1, "tool_decision_vs_own_rate": 1, "retrieval_gain": 1}
    errors = semantics.summary(s)
    assert any("tool_decision_vs_own_rate" in e for e in errors) and any("retrieval_gain" in e for e in errors)
    counts = {"paired_rows": 10, "search_tool_correct_paired": 7, "closed_book_correct_paired": 4}
    counts |= {"tool_decision_own_rows": 10, "tool_decision_matches_own": 6}
    counts |= {"tool_decision_rows": 10}
    s["quality"] = {"attempted": 10, "tool_decision_vs_own_rate": 0.6, "retrieval_gain": 0.3, "counts": counts}
    assert semantics.summary(s) == [] and schemas.errors_for("summary.schema.json", s) == []
    s["quality"]["attempted"] = 0  # populations larger than the cell itself
    assert any("exceeds quality.attempted" in e for e in semantics.summary(s))
    s["quality"]["attempted"] = 10
    s["quality"]["retrieval_gain"] = 0.5
    assert any("retrieval_gain is not" in e for e in semantics.summary(s))
    s = copy.deepcopy(SUMMARY)
    s["quality"] = {
        "answer_in_context": dict.fromkeys(
            ("in_context_attempted", "in_context_correct", "not_in_context_attempted", "not_in_context_correct"), -1
        )
    }
    assert schemas.errors_for("summary.schema.json", s)
    s = copy.deepcopy(SUMMARY)
    s["timing"]["prefill_tps"]["mean"] = 10**9
    assert any("mean" in e for e in semantics.summary(s))
    s = copy.deepcopy(SUMMARY)
    s["quality"] = {"attempted": 10, "correct": 5, "accuracy": 0.5, "ci95": [0.6, 0.4]}
    assert any("ci95" in e for e in semantics.summary(s))


def test_run_folders_are_checked_line_by_line(tmp_path):
    job = json.loads((EXAMPLES / "run.example.json").read_text())
    (tmp_path / "job.json").write_text(json.dumps(job))
    good = copy.deepcopy(REQ)
    dup = copy.deepcopy(REQ)
    other = copy.deepcopy(REQ)
    other["seq"] = 1
    other["job_id"] = "someone-else"
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in (good, dup, other)) + "\n")
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({**SAMPLE, "job_id": "x", "kind": "memory", "seq": 9, "vm_rss_kib": 1, "raw": _mem_raw(1)}) + "\n"
    )
    errors = semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]})
    joined = "\n".join(errors)
    assert "requests.jsonl:2: duplicate seq 0" in joined
    assert "requests.jsonl:3: job_id differs" in joined
    assert "samples.jsonl:1: seq 9 has no request" in joined
    assert "samples.jsonl:1: job_id differs" in joined
    assert not semantics.run_folder(tmp_path, schemas.errors_for, set()) == []


def _samples_for_req(job_id: str = "example-00000") -> str:
    """Samples that support REQ's memory and thermal figures: 20 memory reads 255 ms apart over its
    5.1 s host interval, alternating 800 and 1,000 MiB (peak 1,000, mean 900), and one thermal read,
    each with the raw output its values parse from."""
    rows = []
    for k in range(20):
        kib, t = (800 if k % 2 else 1000) * 1024, k * 255_000_000
        point = {"t_ns": t, "t_start_ns": t, "t_end_ns": t}
        rows.append({"job_id": job_id, "seq": None, **point, "kind": "memory", "vm_rss_kib": kib, "raw": _mem_raw(kib)})
    point = {"t_ns": 10**9, "t_start_ns": 10**9, "t_end_ns": 10**9}
    rows.append({"job_id": job_id, "seq": None, **point, "kind": "thermal", "thermal_status": 0, "raw": THERMAL_0})
    return "".join(json.dumps(r) + "\n" for r in rows)


def _job_folder(tmp_path, requests: str, samples: str | None = None, **overrides):
    job = {**json.loads((EXAMPLES / "run.example.json").read_text()), **overrides}
    (tmp_path / "job.json").write_text(json.dumps(job))
    (tmp_path / "requests.jsonl").write_text(requests)
    (tmp_path / "samples.jsonl").write_text(_samples_for_req(job["job_id"]) if samples is None else samples)
    return job


def test_a_complete_job_needs_its_planned_requests(tmp_path):
    staging = {"sha256_verified_on_host": True, "sha256_verified_on_device": True}
    done = {"outcome": "complete", "staging": staging, "finalized": True, "finished_utc": "2026-10-04T00:00:00Z"}
    base = json.loads((EXAMPLES / "run.example.json").read_text())
    protocol = {**base["protocol"]}
    protocol["protocol_sha256"] = records.protocol_sha256(protocol)
    done["protocol"] = protocol
    done["config_sha256"] = records.config_sha256({**base, **done})
    job = _job_folder(tmp_path, "", **done)
    errors = semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]})
    assert [e for e in errors if "planned" in e] == errors and errors
    _job_folder(tmp_path, json.dumps(REQ) + "\n", **done)
    assert semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}) == []
    # A partial job may stop short, but never record more than it planned.
    _job_folder(tmp_path, json.dumps(REQ) + "\n" + json.dumps({**REQ, "seq": 1}) + "\n")
    assert any("more than the 1 planned" in e for e in semantics.run_folder(tmp_path, schemas.errors_for, set()))


def test_sample_values_must_match_their_raw_output():
    good = {**SAMPLE, "kind": "memory", "vm_rss_kib": 1024, "raw": _mem_raw(1024)}
    assert semantics.sample(good) == []
    assert any("raw output gives" in e for e in semantics.sample({**good, "vm_rss_kib": 1024000}))
    assert schemas.errors_for("sample.schema.json", {**good, "raw": None})  # a success needs its raw output


def test_memory_fields_are_checked_one_by_one(tmp_path):
    job = _job_folder(tmp_path, json.dumps(REQ) + "\n", "")
    # No samples: a mean, a count or a coverage without a peak still needs samples behind it.
    claimed = copy.deepcopy(REQ)
    claimed["memory"] = {"rss_sampled_peak_mib": None, "rss_mean_mib": 900.0, "samples": 20, "coverage": 1.0}
    claimed["memory"]["null_reason"] = "proc_status_unreadable"  # passes the schema; the samples refute it
    _job_folder(tmp_path, json.dumps(claimed) + "\n", "")
    joined = "\n".join(semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}))
    assert "memory.rss_mean_mib is 900.0" in joined and "memory.samples 20 but 0" in joined
    assert "memory.coverage is 1.0" in joined
    # Samples that support a mean: a null mean is refused too.
    nulled = copy.deepcopy(REQ)
    nulled["memory"].update({"rss_mean_mib": None, "null_reason": "mean not computed"})
    _job_folder(tmp_path, json.dumps(nulled) + "\n")
    assert any("rss_mean_mib is None" in e for e in semantics.run_folder(tmp_path, schemas.errors_for, set()))


def test_memory_and_thermal_figures_must_come_from_the_samples(tmp_path):
    job = _job_folder(tmp_path, json.dumps(REQ) + "\n")
    assert semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}) == []
    _job_folder(tmp_path, json.dumps(REQ) + "\n", "")  # the same request, no samples behind it
    joined = "\n".join(semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}))
    assert "memory.samples 20 but 0" in joined and "no thermal sample" in joined
    inflated = copy.deepcopy(REQ)
    inflated["memory"]["rss_sampled_peak_mib"] = 2000.0
    _job_folder(tmp_path, json.dumps(inflated) + "\n")
    assert any("rss_sampled_peak_mib" in e for e in semantics.run_folder(tmp_path, schemas.errors_for, set()))


def test_structurally_invalid_lines_are_reported_not_crashed_on(tmp_path):
    bad_seq = {**REQ, "seq": [1]}
    job = _job_folder(tmp_path, "[]\nnot json\n" + json.dumps(bad_seq) + "\n" + json.dumps(REQ) + "\n", "7\n")
    joined = "\n".join(semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}))
    assert "requests.jsonl:1: not a JSON object" in joined
    assert "requests.jsonl:2: not JSON" in joined
    assert "requests.jsonl:3: seq" in joined
    assert "samples.jsonl:1: not a JSON object" in joined


def _manifest(run, complete=True, kind="harness", **overrides):
    import hashlib

    files = {}
    for f in sorted((run / "dev" / "artifacts").rglob("*")):
        if f.is_file():
            files[f.relative_to(run / "dev" / "artifacts").as_posix()] = {
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "bytes": f.stat().st_size,
            }
    job = {"job": "00000", "files": files, "state": "complete", "status": "COMPLETED", "missing": []}
    run_arn = "arn:aws:devicefarm:us-west-2:<account>:run:p/r"
    (run / "devicefarm-run.json").write_text(json.dumps({"arn": run_arn, "status": "COMPLETED", "totalJobs": 1}))
    job_arn = run_arn.replace(":run:", ":job:") + "/00000"
    (run / "dev" / "devicefarm-job.json").write_text(json.dumps({"arn": job_arn, "status": "COMPLETED"}))
    m = {"kind": kind, "complete": complete, "run_status": "COMPLETED", "failure": None, "jobs_expected": 1}
    m["run_arn"] = run_arn
    m.update({"jobs_listed": 1, "jobs": {"dev": job}, **overrides})
    (run / "pull-manifest.json").write_text(json.dumps(m))


def _complete_job():
    staging = {"sha256_verified_on_host": True, "sha256_verified_on_device": True}
    done = {"outcome": "complete", "staging": staging, "finalized": True, "finished_utc": "2026-10-04T00:00:00Z"}
    base = json.loads((EXAMPLES / "run.example.json").read_text())
    protocol = {**base["protocol"]}
    protocol["protocol_sha256"] = records.protocol_sha256(protocol)
    done["protocol"] = protocol
    done["config_sha256"] = records.config_sha256({**base, **done})
    return done


def test_validate_runs_checks_a_pulled_run_against_its_manifest(tmp_path):
    run = tmp_path / "run"
    folder = run / "dev" / "artifacts"
    (folder / "probe").mkdir(parents=True)
    for f in ("_host.txt", "getprop.txt"):
        (folder / "probe" / f).write_text("x\n")
    _job_folder(folder, json.dumps(REQ) + "\n", **_complete_job())
    # Without a manifest a pull is refused; standalone inspection is a separate, explicit mode.
    assert any("missing" in e for e in schemas.validate_runs(run))
    assert schemas.validate_runs(run, standalone=True) == []
    _manifest(run)
    assert schemas.validate_runs(run) == []
    # Each claim of the manifest is checked, not only its summary flag.
    for bad, needle in (
        ({"run_status": "RUNNING"}, "run status"),
        ({"failure": {"type": "X"}}, "records a failure"),
        ({"jobs_expected": 3}, "job counts"),
        ({"jobs_expected": None}, "job counts"),
        ({"jobs": {"dev": {"files": {}, "state": "incomplete"}}}, "not complete"),
        ({"jobs": {"dev": {"files": {}, "state": "complete", "status": "COMPLETED"}, "gone": {}}}, "folder missing"),
    ):
        _manifest(run, **bad)
        assert any(needle in e for e in schemas.validate_runs(run)), needle
    # Device Farm's own run record must agree with the manifest.
    _manifest(run)
    run_record = json.loads((run / "devicefarm-run.json").read_text())
    (run / "devicefarm-run.json").write_text(json.dumps({**run_record, "totalJobs": 2}))
    assert any("manifest has 1" in e for e in schemas.validate_runs(run))
    # Jobs must belong to the manifest's run, and two folders cannot hold one job.
    _manifest(run)
    (run / "dev" / "devicefarm-job.json").write_text(
        json.dumps({"arn": "arn:aws:devicefarm:us-west-2:<account>:job:p/other/00000", "status": "COMPLETED"})
    )
    assert any("not a job of run" in e for e in schemas.validate_runs(run))
    _manifest(run)
    m = json.loads((run / "pull-manifest.json").read_text())
    (run / "twin" / "artifacts").mkdir(parents=True, exist_ok=True)
    (run / "twin" / "devicefarm-job.json").write_text((run / "dev" / "devicefarm-job.json").read_text())
    m["jobs"]["twin"] = {**m["jobs"]["dev"], "files": {}}
    m.update({"jobs_expected": 2, "jobs_listed": 2})
    (run / "pull-manifest.json").write_text(json.dumps(m))
    (run / "devicefarm-run.json").write_text(json.dumps({**run_record, "totalJobs": 2}))
    assert any("the same job as another folder" in e for e in schemas.validate_runs(run))
    import shutil

    shutil.rmtree(run / "twin")
    # A partial job belongs in standalone inspection, not in a complete pulled run.
    _job_folder(folder, json.dumps(REQ) + "\n")
    _manifest(run)
    assert any("finalized, complete jobs" in e for e in schemas.validate_runs(run))
    _job_folder(folder, json.dumps(REQ) + "\n", **_complete_job())
    _manifest(run)
    (folder / "requests.jsonl").write_text(json.dumps({**REQ, "seq": 0}) + "\n\n")
    assert any("differs from the sha256" in e for e in schemas.validate_runs(run))
    (folder / "extra.txt").write_text("x")
    assert any("not in the pull manifest" in e for e in schemas.validate_runs(run))
    _manifest(run, complete=False, kind="probe")
    errors = schemas.validate_runs(run)
    assert any("not complete" in e for e in errors) and any("not as a harness run" in e for e in errors)
    assert schemas.validate_runs(tmp_path / "empty", standalone=True) != []
