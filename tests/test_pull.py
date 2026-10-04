"""pull_run against a fake Device Farm: fail-closed redaction, staging and completeness."""

import io
import json
import zipfile

import pytest

from execubench import devicefarm

ACCOUNT = "123456789012"
RUN_ARN = f"arn:aws:devicefarm:us-west-2:{ACCOUNT}:run:p/r"
SERIAL = "SERIAL12345"


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr("Host_Machine_Files/$DEVICEFARM_LOG_DIR/" + name, data)
    return buf.getvalue()


PROBE = {
    "probe/_host.txt": f"device_name={SERIAL}\nrun_arn={RUN_ARN}\n".encode(),
    "probe/getprop.txt": f"[ro.serialno]: [{SERIAL}]\n[ro.ril.oem.imei1]: [356789012345678]\n".encode(),
}


class FakeDF:
    def __init__(self, archives_per_job, status="COMPLETED", total_jobs=None, run_name=None):
        self.archives = archives_per_job
        self.status = status
        self.total = len(archives_per_job) if total_jobs is None else total_jobs
        self.run_name = run_name or f"probe on {SERIAL}"

    def get_run(self, arn):
        return {"run": {"arn": RUN_ARN, "status": self.status, "name": self.run_name, "totalJobs": self.total}}

    def list_jobs(self, arn, **_):
        return {
            "jobs": [
                {
                    "arn": f"{RUN_ARN}/{i:05d}".replace(":run:", ":job:"),
                    "status": "COMPLETED",
                    "result": "PASSED",
                    "message": f"unit {SERIAL}",
                    "device": {"modelId": f"M{i}", "os": "16"},
                }
                for i in range(len(self.archives))
            ]
        }

    def list_artifacts(self, arn, type, **_):
        i = int(arn.rsplit("/", 1)[-1])
        return {
            "artifacts": [
                {"type": "CUSTOMER_ARTIFACT", "arn": f"art-{i}-{k}", "url": f"fake://{i}/{k}"}
                for k in range(len(self.archives[i]))
            ]
        }


@pytest.fixture
def fake_http(monkeypatch):
    def install(df):
        def get(url, dest, max_bytes, timeout=60, deadline=None):
            i, k = map(int, url.removeprefix("fake://").split("/"))
            data = df.archives[i][k]
            dest.write_bytes(data)
            return len(data)

        monkeypatch.setattr(devicefarm, "http_get", get)

    return install


def test_a_clean_pull_redacts_dumps_and_metadata(tmp_path, fake_http):
    df = FakeDF([[_zip(PROBE)]])
    fake_http(df)
    dest = tmp_path / "run"
    manifest = devicefarm.pull_run(df, RUN_ARN, dest)
    assert manifest["complete"] and dest.exists() and not (tmp_path / "run.partial").exists()
    everything = b"".join(p.read_bytes() for p in dest.rglob("*") if p.is_file())
    for secret in (SERIAL.encode(), b"356789012345678", ACCOUNT.encode()):
        assert secret not in everything
    job = json.loads(next(dest.rglob("devicefarm-job.json")).read_text())
    assert "unit-" in job["message"]
    assert manifest["jobs"]["M0-android16"]["files"]["probe/getprop.txt"]["sha256"]


def test_an_identifier_in_a_harness_record_refuses_the_pull(tmp_path, fake_http):
    files = {**PROBE, "requests.jsonl": f'{{"note": "{SERIAL}"}}\n'.encode()}
    df = FakeDF([[_zip(files)]])
    fake_http(df)
    with pytest.raises(devicefarm.LeakFound):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")
    assert not (tmp_path / "run").exists()
    assert (tmp_path / "run.partial").exists()


def test_records_are_never_rewritten(tmp_path, fake_http):
    files = {**PROBE, "requests.jsonl": b'{"prompt_text": "hello"}\n'}
    df = FakeDF([[_zip(files)]])
    fake_http(df)
    dest = tmp_path / "run"
    devicefarm.pull_run(df, RUN_ARN, dest)
    assert (dest / "M0-android16" / "artifacts" / "requests.jsonl").read_bytes() == files["requests.jsonl"]


def test_a_job_without_artifacts_is_incomplete(tmp_path, fake_http):
    df = FakeDF([[_zip(PROBE)], []])
    fake_http(df)
    with pytest.raises(RuntimeError, match="incomplete"):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")
    manifest = devicefarm.pull_run(df, RUN_ARN, tmp_path / "run", allow_incomplete=True, restart=True)
    assert not manifest["complete"]
    assert manifest["jobs"]["M1-android16"]["state"] == "incomplete"
    assert manifest["jobs"]["M1-android16"]["missing"] == ["probe/_host.txt", "probe/getprop.txt"]
    assert list(tmp_path.glob("run.failed-*"))  # the first attempt was moved aside, not deleted


def test_a_running_run_is_refused(tmp_path, fake_http):
    df = FakeDF([[_zip(PROBE)]], status="RUNNING")
    fake_http(df)
    with pytest.raises(RuntimeError, match="RUNNING"):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")


def test_existing_destination_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        devicefarm.pull_run(FakeDF([]), RUN_ARN, tmp_path / "run")


def test_archive_bounds(tmp_path, monkeypatch):
    path = tmp_path / "a.zip"
    path.write_bytes(_zip({"probe/big.txt": b"x" * 4096}))
    monkeypatch.setattr(devicefarm, "MAX_MEMBER_BYTES", 1024)
    with pytest.raises(ValueError, match="size bounds"):
        devicefarm.extract_archive(path, tmp_path / "x1")
    monkeypatch.setattr(devicefarm, "MAX_MEMBER_BYTES", 1 << 20)
    # Member and byte budgets are per job: two archives share one budget.
    budget = devicefarm.JobBudget(members=1, bytes=1 << 20)
    devicefarm.extract_archive(path, tmp_path / "x2", budget)
    with pytest.raises(ValueError, match="members"):
        devicefarm.extract_archive(path, tmp_path / "x3", budget)
    budget = devicefarm.JobBudget(members=10, bytes=6000)
    devicefarm.extract_archive(path, tmp_path / "x4", budget)
    with pytest.raises(ValueError, match="size bounds"):
        devicefarm.extract_archive(path, tmp_path / "x5", budget)


def test_staged_members_are_scanned_across_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(devicefarm, "SCAN_CHUNK", 7)
    staged = tmp_path / "requests.jsonl"
    staged.write_bytes(b"x" * 5 + SERIAL.encode() + b"y" * 20)
    with pytest.raises(devicefarm.LeakFound):
        devicefarm.scrub({**PROBE, "requests.jsonl": staged})


def test_spend_guard():
    assert devicefarm.spend_guard(2, 15, 30) == 30
    with pytest.raises(RuntimeError, match="ceiling"):
        devicefarm.spend_guard(19, 15, 100)


def test_a_bundle_without_identity_is_refused(tmp_path, fake_http):
    df = FakeDF([[_zip({"requests.jsonl": b'{"serial": "SERIAL12345"}\n'})]])
    fake_http(df)
    with pytest.raises(devicefarm.LeakFound, match="no unit identity"):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")
    manifest = json.loads((tmp_path / "run.partial" / "pull-manifest.json").read_text())
    assert "no unit identity" in manifest["failure"]["message"] and not manifest["complete"]
    assert SERIAL not in json.dumps(manifest)


def test_multiline_identifiers_cannot_hide_in_json_metadata(tmp_path, fake_http):
    probe = {**PROBE, "probe/getprop.txt": PROBE["probe/getprop.txt"] + b"[ro.boot.chipid]: [CHIP123\nCHIP456]\n"}
    df = FakeDF([[_zip(probe)]], run_name="run CHIP123\nCHIP456")
    fake_http(df)
    dest = tmp_path / "run"
    devicefarm.pull_run(df, RUN_ARN, dest)
    assert "CHIP123" not in (dest / "devicefarm-run.json").read_text()


def test_a_missing_job_listing_is_incomplete(tmp_path, fake_http):
    df = FakeDF([[_zip(PROBE)]], total_jobs=3)
    fake_http(df)
    with pytest.raises(RuntimeError, match="1 of 3 jobs"):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")


def test_schedule_once_finds_a_run_created_by_an_ambiguous_call():
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    kw = {"projectArn": "p", "name": "probe-x", "devicePoolArn": "pool", "appArn": "app"}

    class Lister:
        """Lists `before` until the schedule call has been made, then `after`."""

        def __init__(self, before, after):
            self.before, self.after, self.called = before, after, False

        def schedule_run(self, **_):
            self.called = True
            raise TimeoutError("read timed out")

        def list_runs(self, arn, **_):
            return {"runs": self.after if self.called else self.before}

    mine = {"name": "probe-x", "arn": "run-1", "created": now, "devicePoolArn": "pool", "appUpload": "app"}
    df = Lister([], [mine])
    assert devicefarm.schedule_once(df, df, **kw)["arn"] == "run-1"

    df = Lister([], [])
    with pytest.raises(TimeoutError):
        devicefarm.schedule_once(df, df, **kw)

    # A name already in use is refused before anything is sent.
    df = Lister([{**mine, "created": now - datetime.timedelta(minutes=5)}], [])
    with pytest.raises(RuntimeError, match="already exists"):
        devicefarm.schedule_once(df, df, **kw)
    assert not df.called

    # Another pool's run with the same name is not ours; two candidates are never guessed between.
    df = Lister([], [{**mine, "devicePoolArn": "other"}])
    with pytest.raises(TimeoutError):
        devicefarm.schedule_once(df, df, **kw)
    df = Lister([], [mine, {**mine, "arn": "run-2"}])
    with pytest.raises(RuntimeError, match="2 runs match"):
        devicefarm.schedule_once(df, df, **kw)
    assert devicefarm.unique_run_name("probe") != devicefarm.unique_run_name("probe")


def _counting_server(handler_body):
    import http.server
    import threading

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            hits.append(1)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            handler_body(self)

        do_GET = do_POST  # noqa: N815

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, hits


def test_the_no_retry_client_sends_a_failing_call_once():
    """Botocore's max_attempts counts retries; the no-retry client must send exactly one call."""
    import boto3
    import botocore.config

    def fail(handler):
        handler.send_response(500)
        handler.send_header("Content-Length", "2")
        handler.end_headers()
        handler.wfile.write(b"{}")

    server, hits = _counting_server(fail)
    try:
        df = boto3.client(
            "devicefarm",
            region_name="us-west-2",
            endpoint_url=f"http://127.0.0.1:{server.server_port}",
            aws_access_key_id="x",
            aws_secret_access_key="x",
            config=botocore.config.Config(retries=devicefarm.retry_config(False)),
        )
        with pytest.raises(Exception):  # noqa: B017 - any error; the count is the point
            df.get_account_settings()
    finally:
        server.shutdown()
    assert len(hits) == 1


def test_a_trickling_download_stops_at_the_deadline(tmp_path, monkeypatch):
    import time

    def trickle(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", "100000")
        handler.end_headers()
        try:
            for _ in range(100):
                handler.wfile.write(b"x")
                handler.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass

    server, _ = _counting_server(trickle)
    monkeypatch.setattr(devicefarm, "DOWNLOAD_DEADLINE_S", 1)
    monkeypatch.setattr(devicefarm, "_backoff", lambda *a: (_ for _ in ()).throw(TimeoutError("no retry")))
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            devicefarm.http_get(f"http://127.0.0.1:{server.server_port}/a", tmp_path / "a", 10**6, timeout=5)
    finally:
        server.shutdown()
    assert time.monotonic() - started < 4


ESCAPED = (
    b'{"x": "SERIAL\\u00312345"}',
    b'{"x": "serial12345"}',
    b"GET /a?u=SERIAL%312345",
    b'{"x": "SERIAL%\\u003312345"}',  # a percent-escape inside a JSON string
    b"u=SERIAL%5Cu00312345",  # a JSON escape inside a percent-escaped value
)


def test_escaped_and_case_folded_identifiers_refuse_the_pull():
    for record in ESCAPED:
        with pytest.raises(devicefarm.LeakFound):
            devicefarm.scrub({**PROBE, "requests.jsonl": record})


def test_probe_dumps_are_rescanned_after_redaction():
    out, _ = devicefarm.scrub({**PROBE, "probe/extra.txt": b"x serial12345 y"})
    assert b"serial12345" not in out["probe/extra.txt"].lower()
    with pytest.raises(devicefarm.LeakFound, match="after redaction"):
        devicefarm.scrub({**PROBE, "probe/extra.txt": b'{"x": "SERIAL\\u00312345"}'})


def test_composed_escapes_are_found_across_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(devicefarm, "SCAN_CHUNK", 5)
    staged = tmp_path / "r.jsonl"
    staged.write_bytes(b"x" * 3 + ESCAPED[3] + b"y" * 40)
    with pytest.raises(devicefarm.LeakFound):
        devicefarm.scrub({**PROBE, "requests.jsonl": staged})


def test_directory_entries_spend_the_member_budget(tmp_path):
    path = tmp_path / "dirs.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(5):
            z.writestr(f"Host_Machine_Files/$DEVICEFARM_LOG_DIR/d{i}/", b"")
    with pytest.raises(ValueError, match="members"):
        devicefarm.extract_archive(path, tmp_path / "x", devicefarm.JobBudget(members=3))


def test_nonfinite_ceilings_are_refused():
    for bad in (float("nan"), float("inf"), 0, -5):
        with pytest.raises(ValueError):
            devicefarm.spend_guard(1, 15, bad)


def test_a_refusal_never_writes_the_identifier_into_the_manifest(tmp_path, fake_http):
    df = FakeDF([[_zip({**PROBE, f"{SERIAL}.log": b"hello"})]])
    fake_http(df)
    with pytest.raises(devicefarm.LeakFound):
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")
    text = (tmp_path / "run.partial" / "pull-manifest.json").read_text()
    assert SERIAL not in text and "file #" in text


def test_archive_errors_are_recorded_without_their_text(tmp_path, fake_http):
    df = FakeDF([[b"not a zip " + SERIAL.encode()]])
    fake_http(df)
    with pytest.raises(Exception):  # noqa: B017
        devicefarm.pull_run(df, RUN_ARN, tmp_path / "run")
    failure = json.loads((tmp_path / "run.partial" / "pull-manifest.json").read_text())["failure"]
    assert failure["stage"] == "download" and failure["message"] is None


def test_expired_urls_are_refreshed_by_artifact_arn(tmp_path, monkeypatch):
    calls = []
    data = _zip(PROBE)

    def get(url, dest, max_bytes, timeout=60, deadline=None):
        calls.append(url)
        if url == "fake://old":
            raise devicefarm.ExpiredURL("403")
        dest.write_bytes(data)
        return len(data)

    class DF:
        listings = 0

        def list_artifacts(self, arn, type, **_):
            DF.listings += 1
            mine = {
                "type": "CUSTOMER_ARTIFACT",
                "arn": "mine",
                "url": "fake://old" if DF.listings == 1 else "fake://new",
            }
            other = {"type": "CUSTOMER_ARTIFACT", "arn": "other", "url": "fake://other"}
            # The fresh listing returns the artifacts in another order; the ARN decides.
            return {"artifacts": [mine] if DF.listings == 1 else [other, mine]}

    monkeypatch.setattr(devicefarm, "http_get", get)
    files, _ = devicefarm._job_files(DF(), {"arn": "job"}, tmp_path)
    assert calls == ["fake://old", "fake://new"] and "probe/getprop.txt" in files


def test_non_regular_members_are_refused(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        info = zipfile.ZipInfo("Host_Machine_Files/$DEVICEFARM_LOG_DIR/probe/fifo")
        info.external_attr = 0o010644 << 16  # a FIFO
        z.writestr(info, b"")
    with pytest.raises(ValueError, match="non-regular"):
        devicefarm.extract_archive(path, tmp_path / "x")


def test_pool_rules_are_exactly_the_manifest_devices():
    rules = devicefarm.pool_rules(["arn:b", "arn:a"])
    assert rules == [{"attribute": "ARN", "operator": "IN", "value": '["arn:a", "arn:b"]'}]
    with pytest.raises(ValueError):
        devicefarm.pool_rules(["arn:a", "arn:a"])
