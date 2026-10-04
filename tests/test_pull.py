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
        def get(url, dest, max_bytes, timeout=60):
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
        devicefarm.read_archive(path)
    monkeypatch.setattr(devicefarm, "MAX_MEMBER_BYTES", 1 << 20)
    monkeypatch.setattr(devicefarm, "MAX_MEMBERS", 0)
    with pytest.raises(ValueError, match="members"):
        devicefarm.read_archive(path)


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
    assert "no unit identity" in manifest["failure"] and not manifest["complete"]


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

    class Flaky:
        def schedule_run(self, **kw):
            raise TimeoutError("read timed out")

    class Lister:
        def list_runs(self, arn, **_):
            now = datetime.datetime.now(datetime.UTC)
            return {"runs": [{"name": "probe-x", "arn": "run-1", "created": now}]}

    assert devicefarm.schedule_once(Flaky(), Lister(), projectArn="p", name="probe-x")["arn"] == "run-1"

    class Empty:
        def list_runs(self, arn, **_):
            return {"runs": []}

    with pytest.raises(TimeoutError):
        devicefarm.schedule_once(Flaky(), Empty(), projectArn="p", name="probe-x")


def test_expired_urls_are_refreshed_by_artifact_arn(tmp_path, monkeypatch):
    calls = []
    data = _zip(PROBE)

    def get(url, dest, max_bytes, timeout=60):
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
        devicefarm.read_archive(path)


def test_pool_rules_are_exactly_the_manifest_devices():
    rules = devicefarm.pool_rules(["arn:b", "arn:a"])
    assert rules == [{"attribute": "ARN", "operator": "IN", "value": '["arn:a", "arn:b"]'}]
    with pytest.raises(ValueError):
        devicefarm.pool_rules(["arn:a", "arn:a"])
