"""AWS Device Farm plumbing: uploads, runs, and pulling every job's customer artifacts.

Device Farm lives in us-west-2 only. A run fans out to one job per device; each job's
`$DEVICEFARM_LOG_DIR` comes back as a "Customer Artifacts" zip, which is the only output we
trust. Device Farm's own device catalogue (`list-devices`) is recorded but never used as a
spec source: its `memory` field is storage, and its `cpu.clock` is one cluster's clock, not
the phone's fastest core (see docs/DEVICES.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

REGION = "us-west-2"


def client():
    import boto3

    return boto3.client("devicefarm", region_name=REGION)


def upload(df, project_arn: str, path: Path, kind: str, name: str | None = None) -> str:
    """Upload one file and wait until Device Farm has processed it; returns the upload ARN."""
    up = df.create_upload(
        projectArn=project_arn, name=name or path.name, type=kind, contentType="application/octet-stream"
    )["upload"]
    req = urllib.request.Request(up["url"], data=path.read_bytes(), method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    urllib.request.urlopen(req).read()
    while True:
        u = df.get_upload(arn=up["arn"])["upload"]
        if u["status"] == "SUCCEEDED":
            return u["arn"]
        if u["status"] == "FAILED":
            raise RuntimeError(f"upload {path.name} failed: {u.get('metadata') or u.get('message')}")
        time.sleep(3)


def wait(df, run_arn: str, poll: int = 30) -> dict:
    while True:
        run = df.get_run(arn=run_arn)["run"]
        if run["status"] == "COMPLETED":
            return run
        time.sleep(poll)


def jobs(df, run_arn: str) -> list[dict]:
    out, token = [], None
    while True:
        kw = {"arn": run_arn, **({"nextToken": token} if token else {})}
        page = df.list_jobs(**kw)
        out += page["jobs"]
        token = page.get("nextToken")
        if not token:
            return out


def customer_artifacts(df, job_arn: str) -> list[dict]:
    out, token = [], None
    while True:
        kw = {"arn": job_arn, "type": "FILE", **({"nextToken": token} if token else {})}
        page = df.list_artifacts(**kw)
        out += [a for a in page["artifacts"] if a["type"] == "CUSTOMER_ARTIFACT"]
        token = page.get("nextToken")
        if not token:
            return out


UNIT_PREFIX = "execubench-unit-v1:"
PROGRESS = re.compile(rb"^\[\s*\d+%\] .*\n", flags=re.M)


def unit_hash(serial: str) -> str:
    """A stable short id for one physical unit, so serials are never published verbatim.

    It is not a secret: anyone holding a serial can recompute it. It exists so that results
    can say "same unit" or "different unit" without carrying the serial itself.
    """
    return hashlib.sha256((UNIT_PREFIX + serial).encode()).hexdigest()[:16]


# getprop keys whose values identify one physical unit or SIM. Their values are redacted in
# getprop and wherever else they appear. Per-build ids (ro.build.uuid) and per-part ids
# (connectivity chip ids, ro.boot.cdt_hwid, which repeats across units) are kept.
IDENTIFIER_KEYS = re.compile(
    rb"^(?:"
    rb".*serial.*"  # ro.serialno, ro.boot.ap_serial, ro.boot.ddr_serial, ril.serialnumber, vendor.gsm.serial
    rb"|.*\.imei\d*|.*\.meid|.*iccid.*"  # phone and SIM identities
    rb"|.*mac(addr)?"  # ro.ril.oem.btmac, ro.vendor.oem.wifimac
    rb"|.*uniqueno|.*\.fp\.uid|.*fuseid.*"  # hardware unique numbers, fingerprint module, camera fuses
    rb"|.*\.psno|.*\.sno"  # product serial numbers
    rb"|.*cpuid|.*soc_id"  # ro.boot.cpuid, sys.boot.cpuid, vendor.modem.soc_id (one value per unit)
    rb"|ro\.boot\.chipid|ro\.boot\.board_id|ro\.boot\.vbmeta\.device|ro\.quick_start\.device_id"
    rb")$",
    flags=re.IGNORECASE,
)
# A property value can span lines (Redmi Note 10's ro.boot.chipid ends in a newline), so a value
# runs to the "]" that closes it before the next "[key]: [" line or the end of the dump.
GETPROP_LINE = re.compile(rb"^\[([^\]\n]+)\]: \[(.*?)\]\s*(?=^\[[^\]\n]+\]: \[|\Z)", flags=re.M | re.S)


def scrub(files: dict[str, bytes], account: str | None = None) -> tuple[dict[str, bytes], str | None]:
    """Make pulled artifacts publishable without changing what they measure.

    - The unit's serial becomes `unit-<hash>` everywhere, so "same unit" stays answerable.
    - Every other identifier in IDENTIFIER_KEYS (IMEIs, SIM ICCIDs, AP, DDR and chip serials)
      becomes `<redacted>` in getprop and in every other file where its value appears.
    - The AWS account number becomes `<account>`.
    - adb's per-percent push progress lines are dropped (thousands of them, no information
      beyond the final summary line).
    """
    serial = None
    secrets: set[bytes] = set()
    for name, blob in files.items():
        if name.endswith("_host.txt"):
            m = re.search(rb"^device_name=(\S+)$", blob, flags=re.M)
            if m:
                serial = serial or m.group(1).decode()
        if name.endswith("getprop.txt"):
            for key, value in GETPROP_LINE.findall(blob):
                if key in (b"ro.serialno", b"ro.boot.serialno") and value:
                    serial = serial or value.decode()
                # Values shorter than six bytes ("0", "1", "") are flags, not identifiers,
                # and replacing them everywhere would corrupt unrelated text.
                if IDENTIFIER_KEYS.match(key) and len(value.strip()) >= 6:
                    secrets.add(value)
                    # Replace the trimmed value too, in case another file prints it on one line.
                    secrets.add(value.strip())
    out = {}
    for name, blob in files.items():
        if serial:
            blob = blob.replace(serial.encode(), b"unit-" + unit_hash(serial).encode())
        for value in sorted(secrets, key=len, reverse=True):
            if serial and value == serial.encode():
                continue
            blob = blob.replace(value, b"<redacted>")
        if account:
            blob = blob.replace(account.encode(), b"<account>")
        if name.endswith("push.txt"):
            blob = PROGRESS.sub(b"", blob)
        out[name] = blob
    return out, unit_hash(serial) if serial else None


def _safe_member(name: str) -> str:
    """The artifact-relative path of a zip member, refusing anything that escapes the folder."""
    # The zip nests everything under a folder literally named
    # "Host_Machine_Files/$DEVICEFARM_LOG_DIR", which breaks shell globbing.
    rel = name.split("$DEVICEFARM_LOG_DIR/", 1)[-1]
    parts = Path(rel).parts
    if not rel or Path(rel).is_absolute() or ".." in parts:
        raise ValueError(f"refusing artifact path {name!r}")
    return rel


def slug(device: dict) -> str:
    """A stable folder name per device and OS: the catalogue's model id plus the Android version."""
    model_id = device["modelId"].strip("{}").split(",")[0].replace("/", "-").replace(" ", "_")
    return f"{model_id}-android{device['os']}"


def _write_new(path: Path, data: bytes) -> None:
    """Create a file that must not exist yet: a pull never replaces evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def pull_run(df, run_arn: str, dest: Path) -> list[Path]:
    """Unpack every job's customer artifacts under dest/<device slug>/.

    Beside the artifacts: `devicefarm-job.json` (Device Farm's record of the job: catalogue
    entry, status, result, device minutes) and, once per run, `dest/devicefarm-run.json`.
    `dest` must be new or empty, and is checked before anything is written; every file is
    created exclusively, and two archives of one job may not supply the same path. So a pull
    can fail, but it can never replace evidence from another pull.
    """
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty; pull each run into a fresh folder")
    # ARNs keep their run and job ids, which is all a maintainer needs to find a run again;
    # the account number is masked so pulled data can be published as is.
    account = run_arn.split(":")[4] or None

    def mask(text: str) -> bytes:
        return (text.replace(account, "<account>") if account else text).encode()

    run = df.get_run(arn=run_arn)["run"]
    all_jobs = jobs(df, run_arn)
    slugs = [slug(job["device"]) for job in all_jobs]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"two jobs in {run_arn} map to one folder: {sorted(slugs)}")

    dest.mkdir(parents=True, exist_ok=True)
    _write_new(dest / "devicefarm-run.json", mask(json.dumps(run, indent=1, default=str, sort_keys=True) + "\n"))
    written = []
    for job, name in zip(all_jobs, slugs, strict=True):
        folder = dest / name
        _write_new(folder / "devicefarm-job.json", mask(json.dumps(job, indent=1, default=str, sort_keys=True) + "\n"))
        files: dict[str, bytes] = {}
        for art in customer_artifacts(df, job["arn"]):
            blob = urllib.request.urlopen(art["url"]).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for m in z.infolist():
                    if m.is_dir():
                        continue
                    rel = _safe_member(m.filename)
                    if rel in files:
                        raise ValueError(f"{job['arn']}: two artifact entries for {rel}")
                    files[rel] = z.read(m)
        files, unit = scrub(files, account)
        for rel, data in files.items():
            _write_new(folder / "artifacts" / rel, data)
        if unit:
            _write_new(folder / "unit.txt", (unit + "\n").encode())
        written.append(folder)
    return written
