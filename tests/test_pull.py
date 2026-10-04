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
    def __init__(self, archives_per_job, status="COMPLETED"):
        self.archives = archives_per_job
        self.status = status

    def get_run(self, arn):
        return {"run": {"arn": RUN_ARN, "status": self.status, "name": f"probe on {SERIAL}"}}

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
            "artifacts": [{"type": "CUSTOMER_ARTIFACT", "url": f"fake://{i}/{k}"} for k in range(len(self.archives[i]))]
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
    assert manifest["jobs"]["M1-android16"]["state"] == "no_artifacts"
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
