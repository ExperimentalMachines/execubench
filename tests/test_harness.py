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
    "monotonic": {
        "call_start_ns": 0,
        "first_sampled_token_ns": 400_000_000,
        "first_callback_ns": 401_000_000,
        "call_end_ns": 4_400_000_000,
    },
    "prompt_text": "<|user|>hi<|assistant|>",
    "cached_tokens": 0,
}


APK = "a" * 64


class Handler(BaseHTTPRequestHandler):
    features = sorted(run.REQUIRED_FEATURES)

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"benchmark": {"contract": 1, "features": self.features, "apk_sha256": APK}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
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
    samples = [_mem(reply.sent_ns, 900), _mem(reply.done_ns, 1000)]
    rec = run.request_record(
        "job",
        0,
        "quality",
        {"dataset": "gsm8k", "row_id": "1", "max_tokens": 128},
        reply,
        run.memory_summary(samples, reply.sent_ns, reply.done_ns, "job"),
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


def _mem(t, mib, job="job", **extra):
    sample = {"job_id": job, "kind": "memory", "t_ns": t, "t_start_ns": t, "t_end_ns": t, "vm_rss_kib": mib * 1024}
    return {**sample, **extra}


def test_a_server_without_the_contract_is_refused(server, monkeypatch):
    run.require_contract(server, "k", APK)
    with pytest.raises(run.ContractMissing, match="no pinned"):
        run.require_contract(server, "k", None)
    monkeypatch.setattr(Handler, "features", ["cache_off"])
    with pytest.raises(run.ContractMissing, match="native_max_new_tokens"):
        run.require_contract(server, "k", APK)


def test_sampler_parsers():
    assert sampler.parse_status("VmRSS:\t  1234 kB\nVmHWM:\t 2000 kB\n") == {"vm_rss_kib": 1234, "vm_hwm_kib": 2000}
    assert sampler.parse_freqs("cpu0 1800000\ncpu7 4473600\n") == {"0": 1800000, "7": 4473600}
    dump = (
        "Thermal Status: 1\nCurrent temperatures from HAL:\n"
        "\tTemperature{mValue=41.5, mType=3, mName=SKIN, mStatus=0}\n"
    )
    parsed = sampler.parse_thermal(dump)
    assert parsed["thermal_status"] == 1 and parsed["temps_c"] == {"SKIN": 41.5}


class ScriptedAdb:
    """Answers each command from a table; a missing entry is an adb failure."""

    def __init__(self, answers):
        self.answers = answers

    def shell(self, command, timeout_s=None):
        from execubench.harness.adb import AdbError

        for key, value in self.answers.items():
            if key in command:
                return value
        raise AdbError(f"error: device 'SERIAL12345' not found ({command})", "exit_1")


def test_every_read_writes_a_sample_with_its_raw_output(tmp_path):
    status = "VmRSS:\t  1234 kB\n"
    adb = ScriptedAdb({"/status": status, "scaling_cur_freq": "garbage", "thermalservice": "Thermal Status: 2\n"})
    writer = records.JsonlWriter(tmp_path / "s.jsonl")
    s = sampler.Sampler(adb, 1, writer, "job")
    s._read("memory", "cat /proc/1/status", sampler.parse_status, 5)
    s._read("clock", sampler.FREQ_CMD, lambda t: {"cpu_cur_khz": sampler.parse_freqs(t) or None}, 5)
    s._read("thermal", "dumpsys thermalservice", sampler.parse_thermal, 5)
    s.adb = ScriptedAdb({})
    s._read("memory", "cat /proc/1/status", sampler.parse_status, 5)
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows[0]["vm_rss_kib"] == 1234 and rows[0]["raw"] == status and rows[0]["read_error"] is None
    assert rows[1]["read_error"] == "unparseable" and rows[1]["raw"] == "garbage" and "cpu_cur_khz" not in rows[1]
    assert rows[2]["thermal_status"] == 2 and rows[2]["raw"].startswith("Thermal Status")
    assert rows[3]["read_error"] == "adb_exit_1" and rows[3]["raw"] is None
    assert "SERIAL12345" not in (tmp_path / "s.jsonl").read_text()
    assert s.missed == 2
    for row in rows:
        assert schemas.errors_for("sample.schema.json", row) == [], row
        assert semantics.sample(row) == []


def test_memory_summary_without_samples_says_why():
    assert run.memory_summary([], 0, 10, "job")["null_reason"] == "no_samples_in_window"


def test_memory_summary_refuses_samples_it_cannot_place():
    with pytest.raises(run.BadSample, match="other"):
        run.memory_summary([_mem(5, 1, job="other")], 0, 10, "job")
    with pytest.raises(run.BadSample, match="out of order"):
        run.memory_summary([_mem(5, 1, t_start_ns=8, t_end_ns=2)], 0, 10, "job")
    with pytest.raises(run.BadSample, match="interval"):
        run.memory_summary([{"job_id": "job", "kind": "memory", "t_ns": 5, "vm_rss_kib": 1}], 0, 10, "job")
    with pytest.raises(run.BadSample, match="disagree"):
        run.memory_summary([_mem(5, 1), _mem(5, 2)], 0, 10, "job")
    with pytest.raises(run.BadSample, match="disagree"):
        run.memory_summary([_mem(5, 1, rss_anon_kib=10), _mem(5, 1, rss_anon_kib=20)], 0, 10, "job")
    failed = {**_mem(5, 1), "vm_rss_kib": None, "read_error": "unparseable"}
    assert run.memory_summary([failed], 0, 10, "job")["samples"] == 0


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


MANIFEST = f"runtime: {{apk_sha256: '{APK}'}}\nmodels:\n  a: {{repo: r, revision: '0', file: f, sha256: '0'}}\n"


def test_main_refuses_without_contract(tmp_path, monkeypatch, server):
    monkeypatch.setattr(Handler, "features", [])
    manifest = tmp_path / "m.yaml"
    manifest.write_text(MANIFEST)
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


def test_main_does_not_claim_success_without_orchestration(tmp_path, server):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(MANIFEST)
    args = ["--manifest", str(manifest), "--model", "a", "--out", str(tmp_path / "out"), "--base-url", server]
    assert run.main(args) == 4
    assert json.loads((tmp_path / "out" / "job.json").read_text())["outcome"] == "failed"


def test_memory_mean_is_a_step_function_with_coverage():
    samples = [
        _mem(100, 1),
        _mem(100, 1),
        _mem(300, 3),
        _mem(5, 9999, t_start_ns=-10, t_end_ns=20),  # straddles the start
    ]
    m = run.memory_summary(samples, 0, 400, "job")
    assert m["samples"] == 2 and m["rss_mean_mib"] == (1 * 200 + 3 * 100) / 300
    assert m["coverage"] == 300 / 400 and m["rss_sampled_peak_mib"] == 3


class StreamHandler(BaseHTTPRequestHandler):
    events: list = []
    delay = 0.0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))  # an unread body makes close() reset
        self.send_response(200)
        self.end_headers()
        import time

        for e in self.events:
            self.wfile.write(e.encode())
            self.wfile.flush()
            time.sleep(self.delay)


@pytest.fixture
def stream_server():
    httpd = HTTPServer(("127.0.0.1", 0), StreamHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _sse(*events):
    return [f"data: {json.dumps(e) if isinstance(e, dict) else e}\n\n" for e in events]


def test_a_cut_stream_is_not_a_reply(stream_server, monkeypatch):
    monkeypatch.setattr(StreamHandler, "events", _sse({"choices": [{"delta": {"content": "Hi"}}]}))
    with pytest.raises(client.IncompleteStream) as caught:
        client.chat(stream_server, "k", {})
    assert caught.value.partial.text == "Hi"


def test_a_failed_stream_keeps_its_partial_tool_calls_and_end_time(stream_server, monkeypatch):
    frag = {"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q":'}}
    for tail in ([], _sse({"error": {"message": "boom"}})):
        events = _sse({"choices": [{"delta": {"tool_calls": [frag]}}]}) + tail
        monkeypatch.setattr(StreamHandler, "events", events)
        with pytest.raises(client.IncompleteStream) as caught:
            client.chat(stream_server, "k", {})
        partial = caught.value.partial
        assert partial.tool_calls and partial.tool_calls[0]["id"] == "c1"
        assert partial.done_ns >= partial.sent_ns > 0


def test_an_error_event_is_not_a_reply(stream_server, monkeypatch):
    monkeypatch.setattr(StreamHandler, "events", _sse({"error": {"message": "overflow"}}, "[DONE]"))
    with pytest.raises(client.IncompleteStream, match="overflow"):
        client.chat(stream_server, "k", {})


def test_streamed_tool_calls_are_assembled(stream_server, monkeypatch):
    frag = [
        {"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q":'}},
        {"index": 0, "function": {"arguments": ' "x"}'}},
    ]
    events = _sse(
        {"choices": [{"delta": {"tool_calls": [frag[0]]}}]},
        {"choices": [{"delta": {"tool_calls": [frag[1]]}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    )
    monkeypatch.setattr(StreamHandler, "events", events)
    reply = client.chat(stream_server, "k", {})
    want = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
    assert reply.tool_calls == [want]


def test_an_unpinned_build_is_refused(server):
    with pytest.raises(run.ContractMissing, match="not the pinned"):
        run.require_contract(server, "k", "f" * 64)


class TrickleHandler(BaseHTTPRequestHandler):
    """Headers, then one byte every 50 ms and never a newline."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        import time

        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.end_headers()
        try:
            for _ in range(200):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            pass


def test_a_trickle_without_newlines_stops_at_the_deadline():
    import time
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), TrickleHandler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(client.IncompleteStream, match="no completion within"):
            client.chat(f"http://127.0.0.1:{httpd.server_port}", "k", {}, timeout_s=0.3)
        elapsed = time.monotonic() - started  # the client only, not the server's shutdown
    finally:
        httpd.shutdown()
    assert elapsed < 1.0


def test_malformed_events_are_incomplete_streams_with_the_partial_reply(stream_server, monkeypatch):
    first = {"choices": [{"delta": {"content": "Hi"}}]}
    # Each malformed event is followed by a valid finish and [DONE], so only the bad event can fail it.
    finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    bad_events = (
        "[]",
        '{"choices": [null]}',
        '{"choices": [{"delta": []}]}',
        '{"choices": [{"delta": {"tool_calls": [1]}}]}',
        '{"choices": [{"delta": {"tool_calls": [{"index": [], "function": {"name": "f"}}]}}]}',
        '{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 5}}]}}]}',
        '{"choices": [{"delta": {}, "finish_reason": 3}]}',
        '{"choices": [{"delta": {}, "finish_reason": "banana"}]}',
        '{"choices": [{"delta": {}, "finish_reason": []}]}',
        '{"choices": [{"delta": {}, "finish_reason": {}}]}',
        '{"choices": {}}',
        '{"choices": false}',
        '{"choices": 0}',
    )
    monkeypatch.setattr(StreamHandler, "events", _sse(first, finish, "[DONE]"))
    assert client.chat(stream_server, "k", {}).text == "Hi"  # the clean stream passes
    for bad in bad_events:
        monkeypatch.setattr(StreamHandler, "events", _sse(first, bad, finish, "[DONE]"))
        with pytest.raises(client.IncompleteStream) as caught:
            client.chat(stream_server, "k", {})
        partial = caught.value.partial
        assert partial.text == "Hi" and partial.done_ns >= partial.sent_ns > 0, bad


def test_the_deadline_branch_records_an_end_time(stream_server, monkeypatch):
    monkeypatch.setattr(StreamHandler, "events", _sse({"choices": [{"delta": {"content": "x"}}]}) * 5)
    monkeypatch.setattr(StreamHandler, "delay", 0.2)
    with pytest.raises(client.IncompleteStream, match="no completion within") as caught:
        client.chat(stream_server, "k", {}, timeout_s=0.3)
    assert caught.value.partial.done_ns > caught.value.partial.sent_ns
