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


def test_a_thermal_sample_needs_a_reading():
    # Otherwise valid: the only thing missing is the reading, so the thermal rule is what fails.
    good = {**SAMPLE, "kind": "thermal", "thermal_status": 0}
    assert schemas.errors_for("sample.schema.json", good) == []
    assert schemas.errors_for("sample.schema.json", {**SAMPLE, "kind": "thermal"})
    # A failed read needs no reading, but must say why and carry no values.
    failed = {**SAMPLE, "kind": "thermal", "read_error": "unparseable", "raw": "x"}
    assert schemas.errors_for("sample.schema.json", failed) == [] and semantics.sample(failed) == []
    assert semantics.sample({**failed, "thermal_status": 1})


def test_sample_intervals_must_hold_their_midpoint():
    assert semantics.sample({**SAMPLE, "kind": "memory", "vm_rss_kib": 1, "t_start_ns": 8, "t_end_ns": 2})


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


def test_impossible_statistics_are_rejected():
    s = copy.deepcopy(SUMMARY)
    s["quality"] = {"accuracy": 1}
    assert any("without quality.attempted" in e for e in semantics.summary(s))
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
        json.dumps({**SAMPLE, "job_id": "x", "kind": "memory", "seq": 9, "vm_rss_kib": 1}) + "\n"
    )
    errors = semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]})
    joined = "\n".join(errors)
    assert "requests.jsonl:2: duplicate seq 0" in joined
    assert "requests.jsonl:3: job_id differs" in joined
    assert "samples.jsonl:1: seq 9 has no request" in joined
    assert "samples.jsonl:1: job_id differs" in joined
    assert not semantics.run_folder(tmp_path, schemas.errors_for, set()) == []


def _job_folder(tmp_path, requests: str, samples: str = "", **overrides):
    job = {**json.loads((EXAMPLES / "run.example.json").read_text()), **overrides}
    (tmp_path / "job.json").write_text(json.dumps(job))
    (tmp_path / "requests.jsonl").write_text(requests)
    (tmp_path / "samples.jsonl").write_text(samples)
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


def test_structurally_invalid_lines_are_reported_not_crashed_on(tmp_path):
    bad_seq = {**REQ, "seq": [1]}
    job = _job_folder(tmp_path, "[]\nnot json\n" + json.dumps(bad_seq) + "\n" + json.dumps(REQ) + "\n", "7\n")
    joined = "\n".join(semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]}))
    assert "requests.jsonl:1: not a JSON object" in joined
    assert "requests.jsonl:2: not JSON" in joined
    assert "requests.jsonl:3: seq" in joined
    assert "samples.jsonl:1: not a JSON object" in joined


def test_validate_runs_checks_a_pulled_run(tmp_path):
    folder = tmp_path / "run" / "dev" / "artifacts"
    folder.mkdir(parents=True)
    job = _job_folder(folder, json.dumps(REQ) + "\n")
    assert schemas.validate_runs(tmp_path / "run") == [], job
    (tmp_path / "run" / "pull-manifest.json").write_text(json.dumps({"kind": "probe", "complete": False}))
    errors = schemas.validate_runs(tmp_path / "run")
    assert any("not complete" in e for e in errors) and any("not as a harness run" in e for e in errors)
    assert schemas.validate_runs(tmp_path / "empty") != []
