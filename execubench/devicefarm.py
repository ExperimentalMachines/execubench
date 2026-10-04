"""AWS Device Farm plumbing: uploads, runs, and pulling every job's customer artifacts.

Device Farm lives in us-west-2 only. A run fans out to one job per device; each job's
`$DEVICEFARM_LOG_DIR` comes back as a "Customer Artifacts" zip, which is the only output we
trust. Device Farm's own device catalogue (`list-devices`) is recorded but never used as a
spec source: its `memory` field is storage, and its `cpu.clock` is one cluster's clock, not
the phone's fastest core (see docs/DEVICES.md).

Pulls are fail-closed. Only the probe's own text dumps are ever rewritten (identifiers
redacted); any other file that contains an identifier stops the pull instead of being
silently edited, because harness records carry content hashes that an edit would falsify.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .devices import getprop_pairs

REGION = "us-west-2"

# Bounds. Device Farm drops every customer artifact of a job above 1 GB, so a larger archive
# is either not Device Farm's or broken; members and totals are capped against zip bombs.
MAX_ARCHIVE_BYTES = 1_100_000_000
MAX_MEMBERS = 20_000
MAX_MEMBER_BYTES = 512 * 2**20
MAX_EXPANDED_BYTES = 2 * 2**30
HTTP_TIMEOUT_S = 60
HTTP_ATTEMPTS = 4
DOWNLOAD_DEADLINE_S = 30 * 60  # total, not per socket read: a trickling response still ends
MAX_JOB_ARCHIVE_BYTES = 2 * MAX_ARCHIVE_BYTES
MAX_JOB_EXPANDED_BYTES = 2 * 2**30
UPLOAD_DEADLINE_S = 30 * 60
RUN_DEADLINE_S = 8 * 3600


class LeakFound(RuntimeError):
    """An identifier appeared somewhere the scrubber is not allowed to rewrite."""


class ExpiredURL(RuntimeError):
    """A presigned URL answered 403: list the artifacts again for a fresh one."""


def client(retries: bool = True):
    """A Device Farm client. `retries=False` is for calls that are not idempotent
    (`schedule_run` has no client token, so a blind retry can start a second run)."""
    import boto3
    from botocore.config import Config

    # Adaptive retries back off on Device Farm's throttling (10 TPS for most calls, 1 TPS for
    # writes); explicit timeouts stop a dead connection from hanging a pull.
    config = Config(
        region_name=REGION,
        retries={"max_attempts": 10, "mode": "adaptive"} if retries else {"max_attempts": 1, "mode": "standard"},
        connect_timeout=10,
        read_timeout=60,
    )
    return boto3.client("devicefarm", config=config)


def _backoff(attempt: int) -> None:
    time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random() / 2))


def _transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500 or error.code == 429
    return isinstance(error, urllib.error.URLError | TimeoutError | ConnectionError)


def http_get(url: str, dest: Path, max_bytes: int, timeout: int = HTTP_TIMEOUT_S) -> int:
    """Stream `url` into `dest`, refusing more than `max_bytes`; retries transient failures.

    Raises ExpiredURL on 403, so the caller can fetch a fresh presigned URL.
    """
    for attempt in range(HTTP_ATTEMPTS):
        try:
            total = 0
            deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
                while chunk := resp.read(1 << 20):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"download exceeds {max_bytes} bytes")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"download not finished after {DOWNLOAD_DEADLINE_S} s")
                    out.write(chunk)
            return total
        except urllib.error.HTTPError as error:
            if error.code == 403:
                raise ExpiredURL(str(error)) from error
            if not _transient(error) or attempt == HTTP_ATTEMPTS - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == HTTP_ATTEMPTS - 1:
                raise
        _backoff(attempt)
    raise AssertionError("unreachable")


def http_put(url: str, path: Path, timeout: int = HTTP_TIMEOUT_S) -> None:
    """PUT a file to a presigned URL, streamed rather than read into memory."""
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with open(path, "rb") as body:
                req = urllib.request.Request(url, data=body, method="PUT")
                req.add_header("Content-Type", "application/octet-stream")
                req.add_header("Content-Length", str(path.stat().st_size))
                urllib.request.urlopen(req, timeout=timeout).read()
            return
        except Exception as error:  # noqa: BLE001 - classified just below
            if not _transient(error) or attempt == HTTP_ATTEMPTS - 1:
                raise
        _backoff(attempt)


def upload(df, project_arn: str, path: Path, kind: str, name: str | None = None) -> str:
    """Upload one file and wait, up to a deadline, until Device Farm has processed it."""
    up = df.create_upload(
        projectArn=project_arn, name=name or path.name, type=kind, contentType="application/octet-stream"
    )["upload"]
    http_put(up["url"], path)
    deadline = time.monotonic() + UPLOAD_DEADLINE_S
    while time.monotonic() < deadline:
        u = df.get_upload(arn=up["arn"])["upload"]
        if u["status"] == "SUCCEEDED":
            return u["arn"]
        if u["status"] == "FAILED":
            raise RuntimeError(f"upload {path.name} failed: {u.get('metadata') or u.get('message')}")
        time.sleep(3)
    raise TimeoutError(f"upload {path.name} still processing after {UPLOAD_DEADLINE_S} s")


def wait(df, run_arn: str, poll: int = 30, deadline_s: int = RUN_DEADLINE_S) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        run = df.get_run(arn=run_arn)["run"]
        if run["status"] == "COMPLETED":
            return run
        time.sleep(poll)
    raise TimeoutError(f"{run_arn} not completed after {deadline_s} s")


def _pages(call, key: str, **kw) -> list[dict]:
    out, token = [], None
    while True:
        page = call(**kw, **({"nextToken": token} if token else {}))
        out += page[key]
        token = page.get("nextToken")
        if not token:
            return out


def jobs(df, run_arn: str) -> list[dict]:
    return _pages(df.list_jobs, "jobs", arn=run_arn)


def customer_artifacts(df, job_arn: str) -> list[dict]:
    return [
        a for a in _pages(df.list_artifacts, "artifacts", arn=job_arn, type="FILE") if a["type"] == "CUSTOMER_ARTIFACT"
    ]


def schedule_once(df_no_retry, df, **kwargs) -> dict:
    """Schedule a run exactly once. ScheduleRun has no idempotency token, so on an ambiguous
    failure (timeout, connection reset) the project's runs are searched for one with this name
    created in the last ten minutes before anything is retried; none found means it is safe to
    report the failure and let the operator decide."""
    started = time.time()
    try:
        return df_no_retry.schedule_run(**kwargs)["run"]
    except Exception:
        for run in _pages(df.list_runs, "runs", arn=kwargs["projectArn"]):
            created = run.get("created")
            ts = created.timestamp() if hasattr(created, "timestamp") else 0
            if run.get("name") == kwargs["name"] and ts >= started - 600:
                return run
        raise


def pool_rules(device_arns: list[str]) -> list[dict]:
    """A device pool that is exactly these devices: one ARN IN rule, nothing inferred."""
    if not device_arns or len(set(device_arns)) != len(device_arns):
        raise ValueError("a pool needs distinct device ARNs")
    return [{"attribute": "ARN", "operator": "IN", "value": json.dumps(sorted(device_arns))}]


def pool_devices(df, pool_arn: str, app_arn: str, test_type: str) -> list[dict]:
    """The devices a run on this pool would use: the spend guard counts these."""
    result = df.get_device_pool_compatibility(devicePoolArn=pool_arn, appArn=app_arn, testType=test_type)
    return [d["device"] for d in result.get("compatibleDevices", [])]


def spend_guard(devices: int, job_timeout_min: int, max_device_minutes: float) -> float:
    """Worst-case metered minutes of a run, refusing it above the operator's ceiling."""
    worst = float(devices * job_timeout_min)
    if worst > max_device_minutes:
        raise RuntimeError(
            f"{devices} devices x {job_timeout_min} min = {worst:.0f} worst-case device minutes, "
            f"above the --max-device-minutes ceiling of {max_device_minutes:.0f}"
        )
    return worst


UNIT_PREFIX = "execubench-unit-v1:"
PROGRESS = re.compile(rb"^\[\s*\d+%\] .*\n", flags=re.M)


def unit_hash(serial: str) -> str:
    """A stable short id for one physical unit, so serials are never published verbatim.

    It is not a secret: anyone holding a serial can recompute it. It exists so that results
    can say "same unit" or "different unit" without carrying the serial itself.
    """
    return hashlib.sha256((UNIT_PREFIX + serial).encode()).hexdigest()[:16]


# getprop keys whose values identify one physical unit or SIM. Per-build ids (ro.build.uuid)
# and per-part ids (connectivity chip ids, ro.boot.cdt_hwid, which repeats across units) are
# kept. Every key here held a per-unit value in a real probe dump (tests list them).
IDENTIFIER_KEYS = re.compile(
    rb"^(?:"
    rb".*serial.*"  # ro.serialno, ro.boot.ap_serial, ro.boot.ddr_serial, ril.serialnumber, vendor.gsm.serial
    rb"|.*\.imei\d*|.*\.meid|.*iccid.*"  # phone and SIM identities
    rb"|.*mac(addr)?"  # ro.ril.oem.btmac, ro.vendor.oem.wifimac
    rb"|.*uniqueno|.*\.fp\.uid|.*fuseid.*"  # hardware unique numbers, fingerprint module, camera fuses
    rb"|.*\.psno|.*\.sno"  # product serial numbers
    rb"|.*cpuid|.*soc_id"  # ro.boot.cpuid, sys.boot.cpuid, vendor.modem.soc_id (one value per unit)
    rb"|.*uuid_info|.*\.phone\.id"  # vendor.lge.ril.modem.uuid_info, ril.cdma.phone.id
    rb"|ro\.boot\.chipid|ro\.boot\.board_id|ro\.boot\.vbmeta\.device|ro\.quick_start\.device_id"
    rb")$",
    flags=re.IGNORECASE,
)


PRODUCT_KEYS = re.compile(r"^ro\.product\.(?:[a-z_]+\.)?(?:model|device|name|brand|manufacturer)$")


def _redactable(name: str) -> bool:
    """Only the probe's raw text dumps may be rewritten."""
    return name.startswith("probe/") and name.endswith(".txt")


@dataclass
class Identity:
    serial: str | None = None
    secrets: set[bytes] = field(default_factory=set)

    @property
    def unit(self) -> str | None:
        return unit_hash(self.serial) if self.serial else None

    def needles(self, account: str | None) -> list[tuple[bytes, bytes]]:
        """(identifier, replacement) pairs, longest first so no value is half-replaced."""
        pairs = [(s, b"<redacted>") for s in self.secrets if not (self.serial and s == self.serial.encode())]
        if self.serial:
            pairs.append((self.serial.encode(), b"unit-" + self.unit.encode()))
        if account:
            pairs.append((account.encode(), b"<account>"))
        return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def identify(files: dict[str, bytes]) -> Identity:
    ident = Identity()
    for name, blob in files.items():
        if name.endswith("probe/_host.txt"):
            m = re.search(rb"^device_name=(\S+)$", blob, flags=re.M)
            if m:
                ident.serial = ident.serial or m.group(1).decode()
        if name.endswith("probe/getprop.txt"):
            text = blob.decode("utf-8", "surrogateescape")
            pairs = getprop_pairs(text)
            # Some identifier-shaped keys hold the model name (ro.quick_start.device_id is
            # "SM-X710" on a Galaxy Tab S9, "SM-A346" on a Galaxy A34 whose model is SM-A346B): a
            # value contained in the phone's own product identity is shared by every unit of the
            # model, so it is not redacted.
            shared = {v.strip().encode("utf-8", "surrogateescape") for k, v in pairs if PRODUCT_KEYS.match(k)}
            for key, value in pairs:
                raw = value.encode("utf-8", "surrogateescape")
                if key in ("ro.serialno", "ro.boot.serialno") and raw.strip():
                    ident.serial = ident.serial or raw.strip().decode()
                # Values shorter than six bytes ("0", "1", "") are flags, not identifiers,
                # and replacing them everywhere would corrupt unrelated text.
                is_shared = any(raw.strip() in product for product in shared)
                if IDENTIFIER_KEYS.match(key.encode()) and len(raw.strip()) >= 6 and not is_shared:
                    ident.secrets.update({raw, raw.strip()})
    return ident


def scrub(files: dict[str, bytes], account: str | None = None) -> tuple[dict[str, bytes], str | None]:
    """Redact identifiers in the probe's text dumps; refuse any other file that holds one.

    In `probe/*.txt`: the unit's serial becomes `unit-<hash>`, every value of an
    IDENTIFIER_KEYS property becomes `<redacted>`, the account number becomes `<account>`, and
    adb's per-percent push progress lines are dropped. Any other file (harness records, logs,
    binaries) or file name that contains one of those values raises LeakFound: those files
    carry content hashes, and the harness must never write identifiers into them.
    """
    ident = identify(files)
    if ident.serial is None:
        # Without the probe's identity the leak check below has nothing to look for, so a
        # bundle without it is refused rather than passed through unchecked.
        raise LeakFound("no unit identity (probe/_host.txt or probe/getprop.txt): cannot check for identifiers")
    needles = ident.needles(account)
    out, leaks = {}, []
    for name, blob in files.items():
        if any(needle in name.encode() for needle, _ in needles):
            leaks.append(f"file name {name!r}")
        if _redactable(name):
            for needle, replacement in needles:
                blob = blob.replace(needle, replacement)
            if name.endswith("push.txt"):
                blob = PROGRESS.sub(b"", blob)
        elif any(needle in blob for needle, _ in needles):
            leaks.append(name)
        out[name] = blob
    if leaks:
        raise LeakFound(f"identifiers outside the probe dumps, pull refused: {sorted(leaks)}")
    return out, ident.unit


def sanitize(value, idents: list[Identity], account: str | None):
    """Redact identifiers in every string inside Device Farm's run and job records, before they
    are serialised (so an identifier holding a newline cannot hide behind JSON escaping)."""
    if isinstance(value, dict):
        return {k: sanitize(v, idents, account) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, idents, account) for v in value]
    if isinstance(value, str):
        blob = value.encode("utf-8", "surrogateescape")
        for ident in idents:
            for needle, replacement in ident.needles(account):
                blob = blob.replace(needle, replacement)
        if account:
            blob = blob.replace(account.encode(), b"<account>")
        return blob.decode("utf-8", "surrogateescape")
    return value


def dump_sanitized(record: dict, idents: list[Identity], account: str | None) -> bytes:
    """JSON of a sanitised record, then checked again in serialised form, raw and escaped."""
    text = json.dumps(sanitize(json.loads(json.dumps(record, default=str)), idents, account), indent=1, sort_keys=True)
    for ident in idents:
        for needle, _ in ident.needles(account):
            raw = needle.decode("utf-8", "surrogateescape")
            if raw in text or json.dumps(raw)[1:-1] in text:
                raise LeakFound("an identifier survived metadata sanitising")
    return (text + "\n").encode()


def _safe_member(name: str) -> str:
    """The artifact-relative path of a zip member, refusing anything that escapes the folder."""
    # The zip nests everything under a folder literally named
    # "Host_Machine_Files/$DEVICEFARM_LOG_DIR", which breaks shell globbing.
    rel = name.split("$DEVICEFARM_LOG_DIR/", 1)[-1]
    parts = Path(rel).parts
    if not rel or Path(rel).is_absolute() or ".." in parts:
        raise ValueError(f"refusing artifact path {name!r}")
    return rel


def read_archive(path: Path, budget: list[int] | None = None) -> dict[str, bytes]:
    """Every regular file in a customer-artifact zip, within the size and count bounds.

    `budget` is a one-element list of bytes still allowed for the whole job, shared across all
    of its archives and decremented here.
    """
    files: dict[str, bytes] = {}
    expanded = 0
    budget = budget if budget is not None else [MAX_JOB_EXPANDED_BYTES]
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) > MAX_MEMBERS:
            raise ValueError(f"{path.name}: {len(members)} members, more than {MAX_MEMBERS}")
        for m in members:
            kind = (m.external_attr >> 16) & 0o170000
            # Zips written without Unix modes carry 0; anything else must be a regular file.
            if kind not in (0, 0o100000):
                raise ValueError(f"{path.name}: non-regular member {m.filename!r} (mode {kind:o})")
            rel = _safe_member(m.filename)
            if rel in files:
                raise ValueError(f"{path.name}: two entries for {rel}")
            # Declared sizes can lie, so the cap is enforced on the bytes actually inflated.
            chunks, size = [], 0
            with z.open(m) as src:
                while chunk := src.read(1 << 20):
                    size += len(chunk)
                    expanded += len(chunk)
                    budget[0] -= len(chunk)
                    if size > MAX_MEMBER_BYTES or expanded > MAX_EXPANDED_BYTES or budget[0] < 0:
                        raise ValueError(f"{path.name}: {rel} expands past the size bounds")
                    chunks.append(chunk)
            files[rel] = b"".join(chunks)
    return files


def slug(device: dict) -> str:
    """A stable folder name per device and OS: the catalogue's model id plus the Android version."""
    model_id = device["modelId"].strip("{}").split(",")[0].replace("/", "-").replace(" ", "_")
    return f"{model_id}-android{device['os']}"


def _write_new(path: Path, data: bytes) -> None:
    """Create a file that must not exist yet: a pull never replaces evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def _job_files(df, job: dict, scratch: Path) -> tuple[dict[str, bytes], int]:
    """Download and read every customer-artifact archive of one job, within job-level bounds."""
    files: dict[str, bytes] = {}
    arts = customer_artifacts(df, job["arn"])
    downloaded, budget = 0, [MAX_JOB_EXPANDED_BYTES]
    for index, art in enumerate(arts):
        archive = scratch / f"{index}.zip"
        try:
            downloaded += http_get(art["url"], archive, MAX_ARCHIVE_BYTES)
        except ExpiredURL:
            # Presigned URLs expire: list again and take the same artifact, matched by its ARN.
            fresh = {a["arn"]: a for a in customer_artifacts(df, job["arn"])}
            if art["arn"] not in fresh:
                raise
            downloaded += http_get(fresh[art["arn"]]["url"], archive, MAX_ARCHIVE_BYTES)
        if downloaded > MAX_JOB_ARCHIVE_BYTES:
            raise ValueError(f"{job['arn']}: artifacts exceed {MAX_JOB_ARCHIVE_BYTES} bytes")
        for rel, data in read_archive(archive, budget).items():
            if rel in files:
                raise ValueError(f"{job['arn']}: two artifact entries for {rel}")
            files[rel] = data
        archive.unlink()
    return files, len(arts)


# Files every job of a run kind must deliver, or it is not a complete job.
EXPECTED_FILES = {
    "probe": ("probe/_host.txt", "probe/getprop.txt"),
    "harness": ("probe/_host.txt", "probe/getprop.txt", "job.json", "requests.jsonl", "samples.jsonl"),
}


def pull_run(
    df, run_arn: str, dest: Path, allow_incomplete: bool = False, restart: bool = False, kind: str = "probe"
) -> dict:
    """Pull a run into `dest`, atomically, and return its manifest.

    Everything is written to `<dest>.partial` first and renamed to `dest` only when every job
    has been read and checked. A failed pull leaves `<dest>.partial` with a
    `pull-manifest.json` that says what failed; `restart=True` moves it aside to
    `<dest>.failed-<time>` (never deleted) before trying again. `dest` itself must not exist.

    A job is complete when Device Farm finished it and its artifacts contain every file in
    EXPECTED_FILES[kind]; the run is complete when it is COMPLETED and its job listing matches
    the run's own job count. Anything less is refused unless `allow_incomplete`, in which case
    the manifest records it.
    """
    if dest.exists():
        raise FileExistsError(f"{dest} exists; pull each run into a new folder")
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        if not restart:
            raise FileExistsError(f"{partial} holds an unfinished pull; pass restart to move it aside")
        partial.rename(dest.with_name(f"{dest.name}.failed-{time.strftime('%Y%m%dT%H%M%S')}"))
    account = run_arn.split(":")[4] or None
    expected = EXPECTED_FILES[kind]

    run = df.get_run(arn=run_arn)["run"]
    if run["status"] != "COMPLETED" and not allow_incomplete:
        raise RuntimeError(f"{run_arn} is {run['status']}; wait for it or pass allow_incomplete")
    all_jobs = jobs(df, run_arn)
    slugs = [slug(job["device"]) for job in all_jobs]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"two jobs in {run_arn} map to one folder: {sorted(slugs)}")

    partial.mkdir(parents=True)
    manifest: dict = {
        "kind": kind,
        "run_status": run["status"],
        "jobs_expected": run.get("totalJobs"),
        "jobs_listed": len(all_jobs),
        "jobs": {},
        "complete": False,
        "failure": None,
    }
    idents: list[Identity] = []

    def save_manifest() -> None:
        target = partial / "pull-manifest.json"
        target.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    try:
        with tempfile.TemporaryDirectory(dir=partial.parent) as tmp:
            for job, name in zip(all_jobs, slugs, strict=True):
                files, archives = _job_files(df, job, Path(tmp))
                ident = identify(files) if files else Identity()
                if files:
                    files, unit = scrub(files, account)
                    idents.append(ident)
                else:
                    unit = None
                missing = [f for f in expected if f not in files]
                state = "complete" if not missing and job["status"] == "COMPLETED" else "incomplete"
                manifest["jobs"][name] = {
                    "job": job["arn"].rsplit("/", 1)[-1],
                    "status": job["status"],
                    "result": job.get("result"),
                    "archives": archives,
                    "missing": missing,
                    "state": state,
                    "files": {r: {"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)} for r, d in files.items()},
                }
                folder = partial / name
                for rel, data in files.items():
                    _write_new(folder / "artifacts" / rel, data)
                _write_new(folder / "devicefarm-job.json", dump_sanitized(job, [ident] if files else idents, account))
                if unit:
                    _write_new(folder / "unit.txt", (unit + "\n").encode())
        counts_match = run.get("totalJobs") in (None, len(all_jobs))
        manifest["complete"] = (
            run["status"] == "COMPLETED"
            and counts_match
            and bool(all_jobs)
            and all(j["state"] == "complete" for j in manifest["jobs"].values())
        )
        if not manifest["complete"] and not allow_incomplete:
            raise RuntimeError(
                f"incomplete pull: run {run['status']}, {len(all_jobs)} of {run.get('totalJobs')} jobs listed, "
                f"incomplete jobs {[k for k, v in manifest['jobs'].items() if v['state'] != 'complete']}"
            )
        manifest["run_arn"] = sanitize(run_arn, idents, account)
        _write_new(partial / "devicefarm-run.json", dump_sanitized(run, idents, account))
    except Exception as error:
        manifest["failure"] = sanitize(f"{type(error).__name__}: {error}", idents, account)
        save_manifest()
        raise
    save_manifest()
    partial.rename(dest)
    return manifest
