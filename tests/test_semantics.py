"""The semantic checks catch the impossible records review found validating against the schemas."""

import copy
import json
from pathlib import Path

from execubench import schemas, semantics

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


def test_a_thermal_sample_needs_a_reading():
    assert schemas.errors_for("sample.schema.json", {"job_id": "j", "t_ns": 0, "kind": "thermal", "seq": None})


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
        json.dumps({"job_id": "x", "t_ns": 1, "kind": "memory_clock", "seq": 9, "vm_rss_kib": 1}) + "\n"
    )
    errors = semantics.run_folder(tmp_path, schemas.errors_for, {job["device_id"]})
    joined = "\n".join(errors)
    assert "requests.jsonl:2: duplicate seq 0" in joined
    assert "requests.jsonl:3: job_id differs" in joined
    assert "samples.jsonl:1: seq 9 has no request" in joined
    assert not semantics.run_folder(tmp_path, schemas.errors_for, set()) == []
