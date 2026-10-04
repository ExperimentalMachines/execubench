"""Harness pieces against local stubs: no phone, no network."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from execubench import schemas, semantics
from execubench.harness import client, records, run, sampler, staging
from execubench.harness.adb import Adb

STATS = {
    "inference_start_ms": 1000,
    "prompt_eval_end_ms": 1400,
    "first_token_ms": 1400,
    "inference_end_ms": 5400,
    "prompt_tokens": 200,
    "generated_tokens": 100,
}
EXT = {
    "runner_stats": STATS,
    "monotonic": {"call_start_ns": 0, "first_sampled_token_ns": 400_000_000, "call_end_ns": 4_400_000_000},
    "prompt_text": "<|user|>hi<|assistant|>",
    "cached_tokens": 0,
}


class Handler(BaseHTTPRequestHandler):
    features = sorted(run.REQUIRED_FEATURES)

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"benchmark": {"contract": 1, "features": self.features}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in (
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 200}, "x_execuserve": EXT},
        ):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_client_streams_and_times(server):
    reply = client.chat(server, "k", {"model": "m", "messages": []})
    assert reply.text == "Hello" and reply.finish_reason == "stop"
    assert reply.sent_ns <= reply.first_byte_ns <= reply.done_ns
    assert reply.extension["runner_stats"] == STATS


def test_a_record_built_from_a_reply_passes_schema_and_semantics(server):
    reply = client.chat(server, "k", {"model": "m", "messages": []})
    samples = [
        {"kind": "memory_clock", "t_ns": reply.sent_ns, "vm_rss_kib": 1024 * 900},
        {"kind": "memory_clock", "t_ns": reply.done_ns, "vm_rss_kib": 1024 * 1000},
    ]
    rec = run.request_record(
        "job",
        0,
        "quality",
        {"dataset": "gsm8k", "row_id": "1", "max_tokens": 64},
        reply,
        run.memory_summary(samples, reply.sent_ns, reply.done_ns),
        {
            "battery_temp_c_start": 30.0,
            "battery_temp_c_end": 31.0,
            "thermal_status_max": 0,
            "thermal_event": False,
            "clock_drop": None,
        },
    )
    assert schemas.errors_for("request.schema.json", rec) == []
    assert semantics.request(rec) == []
    assert rec["timings"]["clock_disagreement"] is False


def test_a_server_without_the_contract_is_refused(server, monkeypatch):
    run.require_contract(server, "k")
    monkeypatch.setattr(Handler, "features", ["cache_off"])
    with pytest.raises(run.ContractMissing, match="native_max_new_tokens"):
        run.require_contract(server, "k")


def test_sampler_parsers():
    assert sampler.parse_status("VmRSS:\t  1234 kB\nVmHWM:\t 2000 kB\n") == {"vm_rss_kib": 1234, "vm_hwm_kib": 2000}
    assert sampler.parse_freqs("cpu0 1800000\ncpu7 4473600\n") == {"0": 1800000, "7": 4473600}
    dump = (
        "Thermal Status: 1\nCurrent temperatures from HAL:\n"
        "\tTemperature{mValue=41.5, mType=3, mName=SKIN, mStatus=0}\n"
    )
    assert sampler.parse_thermal(dump) == {"thermal_status": 1, "temps_c": {"SKIN": 41.5}}


def test_memory_summary_without_samples_says_why():
    assert run.memory_summary([], 0, 10)["null_reason"] == "no_samples_in_window"


def test_records_survive_and_hash(tmp_path):
    w = records.JsonlWriter(tmp_path / "r.jsonl")
    w.write({"a": 1})
    w.close()
    assert json.loads((tmp_path / "r.jsonl").read_text()) == {"a": 1}
    job = {"model": {"sha256": "0" * 64}, "protocol": {"x": 1, "protocol_sha256": "y"}}
    assert records.config_sha256(job) == records.config_sha256({**job, "protocol": {"x": 1}})


class FakeAdb(Adb):
    def __init__(self, phone_sha):
        super().__init__("serial")
        object.__setattr__(self, "phone_sha", phone_sha)

    def run(self, *args, timeout_s=None, check=True):
        if args[0] == "shell":
            return f"{self.phone_sha}  {args[1].split()[-1]}\n"
        return ""


def test_push_checks_the_bytes_on_the_phone(tmp_path):
    f = tmp_path / "m.pte"
    f.write_bytes(b"model")
    good = staging.sha256_file(f)
    assert staging.push(FakeAdb(good), f, "/sdcard/x/m.pte", good) >= 0
    with pytest.raises(staging.StagingError):
        staging.push(FakeAdb("0" * 64), f, "/sdcard/x/m.pte", good)


def test_main_refuses_without_contract(tmp_path, monkeypatch, server):
    monkeypatch.setattr(Handler, "features", [])
    manifest = tmp_path / "m.yaml"
    manifest.write_text("models:\n  a: {repo: r, revision: '0', file: f, sha256: '0'}\n")
    assert (
        run.main(["--manifest", str(manifest), "--model", "a", "--out", str(tmp_path / "out"), "--base-url", server])
        == 3
    )
    job = json.loads((Path(tmp_path) / "out" / "job.json").read_text())
    assert job["outcome"] == "failed" and "contract" in job["failure"]


def test_contract_doc_and_code_list_the_same_required_features():
    import re

    doc = (Path(__file__).resolve().parent.parent / "docs" / "EXECUSERVE-CONTRACT.md").read_text()
    table = doc[doc.index("## Features") : doc.index("The first six are required")]
    listed = re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.M)
    assert set(listed[:6]) == run.REQUIRED_FEATURES
    assert set(listed[6:]) == {"effective_threads", "status_memory"}
