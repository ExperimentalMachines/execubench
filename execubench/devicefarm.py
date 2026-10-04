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
import http.client
import json
import math
import random
import re
import struct
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import netio
from .devices import getprop_pairs

REGION = "us-west-2"

# Bounds. Device Farm drops every customer artifact of a job above 1 GB, so a larger archive
# is either not Device Farm's or broken; members and totals are capped against zip bombs.
# Member and byte budgets are per job, shared by all of its archives, and members are staged on
# disk and scanned in chunks, so memory stays bounded by SCAN_CHUNK and the probe dumps.
MAX_ARCHIVE_BYTES = 1_100_000_000
MAX_JOB_ARCHIVE_BYTES = 2 * MAX_ARCHIVE_BYTES
MAX_JOB_MEMBERS = 20_000
MAX_MEMBER_BYTES = 512 * 2**20
MAX_JOB_EXPANDED_BYTES = 2 * 2**30
MAX_DUMP_BYTES = 64 * 2**20  # a probe dump is read whole for redaction; real ones are under 2 MiB
SCAN_CHUNK = 8 * 2**20
HTTP_TIMEOUT_S = 60
HTTP_ATTEMPTS = 4
# Total per operation across every attempt, backoff and URL refresh, enforced by a watchdog
# that cuts the connection off (execubench/netio.py).
DOWNLOAD_DEADLINE_S = 30 * 60
UPLOAD_DEADLINE_S = 30 * 60
RUN_DEADLINE_S = 8 * 3600


class LeakFound(RuntimeError):
    """An identifier appeared somewhere the scrubber is not allowed to rewrite."""


class ExpiredURL(RuntimeError):
    """A presigned URL answered 403: list the artifacts again for a fresh one."""


def retry_config(retries: bool) -> dict:
    """Botocore retry settings. `total_max_attempts` counts the first call; `max_attempts`
    counts only retries after it, so `max_attempts: 1` would still send a call twice."""
    return {"total_max_attempts": 10, "mode": "adaptive"} if retries else {"total_max_attempts": 1, "mode": "standard"}


def client(retries: bool = True):
    """A Device Farm client. `retries=False` is for calls that are not idempotent
    (`schedule_run` has no client token, so a blind retry can start a second run)."""
    import boto3
    from botocore.config import Config

    # Adaptive retries back off on Device Farm's throttling (10 TPS for most calls, 1 TPS for
    # writes); explicit timeouts stop a dead connection from hanging a pull.
    config = Config(region_name=REGION, retries=retry_config(retries), connect_timeout=10, read_timeout=60)
    return boto3.client("devicefarm", config=config)


_remaining = netio.remaining


def _backoff(attempt: int, deadline: float, what: str) -> None:
    pause = min(30.0, 2.0**attempt) * (0.5 + random.random() / 2)
    if pause >= _remaining(deadline, what):
        raise TimeoutError(f"{what}: no time left to retry")
    time.sleep(pause)


class HTTPStatus(RuntimeError):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def _transient(error: Exception) -> bool:
    if isinstance(error, HTTPStatus):
        return error.status >= 500 or error.status == 429
    # A passed deadline is final; retrying cannot help.
    return isinstance(error, OSError | http.client.HTTPException) and not isinstance(error, TimeoutError)


def _check_status(status: int) -> None:
    if status == 403:
        raise ExpiredURL("HTTP 403")
    if not 200 <= status < 300:
        raise HTTPStatus(status)


def http_get(url: str, dest: Path, max_bytes: int, timeout: int = HTTP_TIMEOUT_S, deadline: float | None = None) -> int:
    """Stream `url` into `dest`, refusing more than `max_bytes`; retries transient failures, all
    before `deadline` (default: DOWNLOAD_DEADLINE_S from now). A watchdog cuts off whatever is in
    progress at the deadline (execubench/netio.py), so a trickling server cannot extend it.

    Raises ExpiredURL on 403, so the caller can fetch a fresh presigned URL within the same deadline.
    """
    deadline = deadline if deadline is not None else time.monotonic() + DOWNLOAD_DEADLINE_S
    for attempt in range(HTTP_ATTEMPTS):
        try:
            total = 0
            with netio.request("GET", url, deadline, "download", sock_timeout=timeout) as resp:
                _check_status(resp.status)
                with open(dest, "wb") as out:
                    for chunk in netio.read_chunks(resp, deadline, "download"):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"download exceeds {max_bytes} bytes")
                        out.write(chunk)
            return total
        except Exception as error:  # noqa: BLE001 - classified just below
            if not _transient(error) or attempt == HTTP_ATTEMPTS - 1:
                raise
        _backoff(attempt, deadline, "download")
    raise AssertionError("unreachable")


def http_put(url: str, path: Path, timeout: int = HTTP_TIMEOUT_S) -> None:
    """PUT a file to a presigned URL, streamed, every attempt within UPLOAD_DEADLINE_S."""
    deadline = time.monotonic() + UPLOAD_DEADLINE_S
    headers = {"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)}
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with (
                open(path, "rb") as body,
                netio.request("PUT", url, deadline, "upload", body=body, headers=headers, sock_timeout=timeout) as resp,
            ):
                _check_status(resp.status)
                for _ in netio.read_chunks(resp, deadline, "upload"):
                    pass
            return
        except Exception as error:  # noqa: BLE001 - classified just below
            if not _transient(error) or attempt == HTTP_ATTEMPTS - 1:
                raise
        _backoff(attempt, deadline, "upload")


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


def unique_run_name(prefix: str) -> str:
    """A run name no earlier run can share, so a run can be found again by its name alone."""
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def _matches(run: dict, kwargs: dict, started: float) -> bool:
    """A listed run is the one this call scheduled: same name, pool and app, created after it."""
    created = run.get("created")
    ts = created.timestamp() if hasattr(created, "timestamp") else None
    return (
        run.get("name") == kwargs["name"]
        and run.get("devicePoolArn") in (None, kwargs.get("devicePoolArn"))
        and (run.get("appUpload") in (None, kwargs.get("appArn")))
        and ts is not None
        and ts >= started - 60
    )


def schedule_once(df_no_retry, df, **kwargs) -> dict:
    """Schedule a run exactly once.

    ScheduleRun has no idempotency token. So the name must be unused in the project before the
    call (refused otherwise), the call is sent once (`df_no_retry`, one total attempt), and on an
    ambiguous failure (timeout, connection reset) the project is searched for a run with that
    name, pool and app created since the call started. One match is the run; none means the
    failure stands; two or more is refused, never guessed.
    """
    project = kwargs["projectArn"]
    if any(r.get("name") == kwargs["name"] for r in _pages(df.list_runs, "runs", arn=project)):
        raise RuntimeError(f"a run named {kwargs['name']!r} already exists; use unique_run_name()")
    started = time.time()
    try:
        return df_no_retry.schedule_run(**kwargs)["run"]
    except Exception:
        found = [r for r in _pages(df.list_runs, "runs", arn=project) if _matches(r, kwargs, started)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(f"{len(found)} runs match the ambiguous schedule call; inspect them by hand") from None
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


def finite_positive(value: float, what: str) -> float:
    """Refuse NaN, infinities and non-positive amounts: a NaN ceiling makes every comparison false."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be a finite positive number, not {value!r}")
    return value


def spend_guard(devices: int, job_timeout_min: int, max_device_minutes: float) -> float:
    """Worst-case metered minutes of a run, refusing it above the operator's ceiling."""
    finite_positive(max_device_minutes, "the device-minute ceiling")
    worst = finite_positive(devices * job_timeout_min, "the run's worst case")
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


# A pulled file is either bytes in memory (probe dumps, tests) or a member staged on disk.
Blob = bytes | Path


def _read_small(src: Blob, cap: int = MAX_DUMP_BYTES) -> bytes:
    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"a probe dump is larger than {cap} bytes")
    return data


def _chunks(src: Blob, overlap: int) -> Iterator[bytes]:
    """The blob in SCAN_CHUNK windows that overlap by `overlap` bytes, so a needle split across a
    boundary is still seen whole."""
    if isinstance(src, bytes):
        yield src
        return
    tail = b""
    with open(src, "rb") as f:
        while block := f.read(SCAN_CHUNK):
            yield tail + block
            tail = (tail + block)[-overlap:] if overlap else b""


_JSON_ESCAPE = re.compile(rb'\\u([0-9a-fA-F]{4})|\\(["\\/bfnrt])')
_SIMPLE_ESCAPES = dict(zip(b'"\\/bfnrt', b'"\\/\b\f\n\r\t', strict=True))


def _json_unescape(blob: bytes) -> bytes:
    def one(m: re.Match) -> bytes:
        if m.group(1):
            return chr(int(m.group(1), 16)).encode("utf-8", "surrogatepass")
        return bytes([_SIMPLE_ESCAPES[m.group(2)[0]]])

    return _JSON_ESCAPE.sub(one, blob)


def _views(blob: bytes) -> list[bytes]:
    """The forms an identifier can take in a text record: as written, with JSON string escapes
    decoded (`SERIAL\\u00312345`), with URL percent escapes decoded, and both compositions (a
    percent-escape inside a JSON string, or the reverse), all case-folded. Decoding more than the
    writer meant can only add matches, so the check errs towards refusing."""
    unquote = urllib.parse.unquote_to_bytes
    json_once, url_once = _json_unescape(blob), unquote(blob)
    return [v.lower() for v in (blob, json_once, url_once, unquote(json_once), _json_unescape(url_once))]


# A needle byte can grow to 18 bytes when both encodings are applied (`%31` as `\u0025\u0033\u0031`).
_MAX_EXPANSION = 18


def _holds(src: Blob, needles: list[bytes]) -> bool:
    if not needles:
        return False
    folded = [n.lower() for n in needles]
    overlap = _MAX_EXPANSION * (max(map(len, folded)) + 1)
    return any(n in view for chunk in _chunks(src, overlap) for view in _views(chunk) for n in folded)


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


def identify(files: dict[str, Blob]) -> Identity:
    ident = Identity()
    for name, src in files.items():
        if name.endswith("probe/_host.txt"):
            m = re.search(rb"^device_name=(\S+)$", _read_small(src), flags=re.M)
            if m:
                ident.serial = ident.serial or m.group(1).decode()
        if name.endswith("probe/getprop.txt"):
            text = _read_small(src).decode("utf-8", "surrogateescape")
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


def scrub(
    files: dict[str, Blob], account: str | None = None, ident: Identity | None = None
) -> tuple[dict[str, Blob], str | None]:
    """Redact identifiers in the probe's text dumps; refuse any other file that holds one.

    In `probe/*.txt`: the unit's serial becomes `unit-<hash>`, every value of an
    IDENTIFIER_KEYS property becomes `<redacted>`, the account number becomes `<account>`, and
    adb's per-percent push progress lines are dropped. Any other file (harness records, logs,
    binaries) or file name that contains one of those values, as written or escaped
    (`_views`), raises LeakFound: those files carry content hashes, and the harness must never
    write identifiers into them. LeakFound names offending files by position, never by a name
    that may itself hold the identifier.
    """
    ident = ident or identify(files)
    if ident.serial is None:
        # Without the probe's identity the leak check below has nothing to look for, so a
        # bundle without it is refused rather than passed through unchecked.
        raise LeakFound("no unit identity (probe/_host.txt or probe/getprop.txt): cannot check for identifiers")
    pairs = ident.needles(account)
    needles = [n for n, _ in pairs]
    out: dict[str, Blob] = {}
    leaks: list[str] = []
    for index, (name, src) in enumerate(sorted(files.items())):
        bad_name = _holds(name.encode("utf-8", "surrogateescape"), needles)
        if bad_name:
            leaks.append(f"the name of file #{index}")
        if _redactable(name):
            blob = _read_small(src)
            for needle, replacement in pairs:
                blob = re.sub(re.escape(needle), lambda _, r=replacement: r, blob, flags=re.IGNORECASE)
            if name.endswith("push.txt"):
                blob = PROGRESS.sub(b"", blob)
            # Redaction replaces the literal forms (any case); an escaped form that survives it is
            # refused like anywhere else, never published.
            if _holds(blob, needles):
                leaks.append(f"file #{index} after redaction" if bad_name else f"{name!r} after redaction")
            out[name] = blob
        else:
            if _holds(src, needles):
                leaks.append(f"file #{index}" if bad_name else repr(name))
            out[name] = src
    if leaks:
        raise LeakFound(f"identifiers outside the probe dumps, pull refused: {leaks}")
    return out, ident.unit


def sanitize(value, idents: list[Identity], account: str | None):
    """Redact identifiers in every string inside Device Farm's run and job records, before they
    are serialised (so an identifier holding a newline cannot hide behind JSON escaping)."""
    if isinstance(value, dict):
        return {sanitize(k, idents, account): sanitize(v, idents, account) for k, v in value.items()}
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
    """JSON of a sanitised record, then checked again in serialised form, as written and escaped."""
    text = json.dumps(sanitize(json.loads(json.dumps(record, default=str)), idents, account), indent=1, sort_keys=True)
    needles = [n for ident in idents for n, _ in ident.needles(account)]
    if _holds(text.encode("utf-8", "surrogateescape"), needles):
        raise LeakFound("an identifier survived metadata sanitising")
    return (text + "\n").encode()


def _safe_member(name: str) -> str:
    """The artifact-relative path of a zip member, refusing anything that escapes the folder."""
    # The zip nests everything under a folder literally named
    # "Host_Machine_Files/$DEVICEFARM_LOG_DIR", which breaks shell globbing.
    rel = name.split("$DEVICEFARM_LOG_DIR/", 1)[-1]
    parts = Path(rel).parts
    if not rel or Path(rel).is_absolute() or ".." in parts:
        raise ValueError("refusing an artifact path that escapes the job folder")
    return rel


MAX_CENTRAL_DIRECTORY_BYTES = 16 * 2**20


def _central_directory_entries(path: Path) -> int:
    """Entries the zip declares, read from its end-of-central-directory record. Zip64 archives
    (over 65,535 entries or 4 GiB) are refused: no Device Farm job archive within our bounds needs
    one, and the classic record cannot describe them."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - (22 + 65535)))
        tail = f.read()
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or len(tail) - at < 22:
        raise ValueError(f"{path.name}: not a zip archive (no end-of-central-directory record)")
    entries, cd_bytes = struct.unpack("<HI", tail[at + 10 : at + 16])
    if entries == 0xFFFF or cd_bytes == 0xFFFFFFFF or b"PK\x06\x07" in tail[max(0, at - 20) : at]:
        raise ValueError(f"{path.name}: zip64 archives are refused")
    if cd_bytes > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError(f"{path.name}: central directory larger than {MAX_CENTRAL_DIRECTORY_BYTES} bytes")
    return entries


@dataclass
class JobBudget:
    """What one job's archives may still expand to, shared across all of them."""

    members: int = MAX_JOB_MEMBERS
    bytes: int = MAX_JOB_EXPANDED_BYTES


def extract_archive(path: Path, into: Path, budget: JobBudget | None = None) -> dict[str, Path]:
    """Stage every regular file of a customer-artifact zip on disk, within the bounds.

    Members are written under numbered names (a member's own name is untrusted), and the cap is
    enforced on the bytes actually inflated, because declared sizes can lie. Errors never quote a
    member name: it may hold an identifier and has not been checked yet.
    """
    budget = budget or JobBudget()
    # The entry count comes from the end-of-central-directory record before ZipFile parses (and
    # holds in memory) the whole directory; directory entries count like files.
    entries = _central_directory_entries(path)
    if entries > budget.members:
        raise ValueError(f"{path.name}: the job's archives hold more than {MAX_JOB_MEMBERS} members")
    into.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        budget.members -= len(infos)
        if budget.members < 0 or len(infos) != entries:
            raise ValueError(f"{path.name}: the job's archives hold more than {MAX_JOB_MEMBERS} members")
        for m in (m for m in infos if not m.is_dir()):
            kind = (m.external_attr >> 16) & 0o170000
            # Zips written without Unix modes carry 0; anything else must be a regular file.
            if kind not in (0, 0o100000):
                raise ValueError(f"{path.name}: a non-regular member (mode {kind:o})")
            rel = _safe_member(m.filename)
            if rel in files:
                raise ValueError(f"{path.name}: two entries for one path")
            target = into / f"{path.stem}-{len(files):05d}.bin"
            size = 0
            with z.open(m) as src, open(target, "xb") as out:
                while chunk := src.read(1 << 20):
                    size += len(chunk)
                    budget.bytes -= len(chunk)
                    if size > MAX_MEMBER_BYTES or budget.bytes < 0:
                        raise ValueError(f"{path.name}: a member expands past the size bounds")
                    out.write(chunk)
            files[rel] = target
    return files


def slug(device: dict) -> str:
    """A stable folder name per device and OS: the catalogue's model id plus the Android version."""
    model_id = device["modelId"].strip("{}").split(",")[0].replace("/", "-").replace(" ", "_")
    return f"{model_id}-android{device['os']}"


def _write_new(path: Path, src: Blob) -> tuple[str, int]:
    """Create a file that must not exist yet (a pull never replaces evidence); its sha256 and size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with open(path, "xb") as out:
        for chunk in _chunks(src, 0):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    return digest.hexdigest(), size


def _job_files(df, job: dict, scratch: Path) -> tuple[dict[str, Path], int]:
    """Download and stage every customer-artifact archive of one job, within job-level bounds."""
    files: dict[str, Path] = {}
    scratch.mkdir(parents=True, exist_ok=True)
    arts = customer_artifacts(df, job["arn"])
    downloaded, budget = 0, JobBudget()
    for index, art in enumerate(arts):
        archive = scratch / f"{index}.zip"
        # One deadline per archive, kept across a URL refresh.
        deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
        try:
            downloaded += http_get(art["url"], archive, MAX_ARCHIVE_BYTES, deadline=deadline)
        except ExpiredURL:
            # Presigned URLs expire: list again and take the same artifact, matched by its ARN.
            fresh = {a["arn"]: a for a in customer_artifacts(df, job["arn"])}
            if art["arn"] not in fresh:
                raise
            downloaded += http_get(fresh[art["arn"]]["url"], archive, MAX_ARCHIVE_BYTES, deadline=deadline)
        if downloaded > MAX_JOB_ARCHIVE_BYTES:
            raise ValueError(f"the job's artifacts exceed {MAX_JOB_ARCHIVE_BYTES} bytes")
        for rel, staged in extract_archive(archive, scratch / f"{index}.d", budget).items():
            if rel in files:
                raise ValueError("two artifact archives hold one path")
            files[rel] = staged
        archive.unlink()
    return files, len(arts)


# Files every job of a run kind must deliver, or it is not a complete job.
EXPECTED_FILES = {
    "probe": ("probe/_host.txt", "probe/getprop.txt"),
    "harness": ("probe/_host.txt", "probe/getprop.txt", "job.json", "requests.jsonl", "samples.jsonl"),
}

# Stages of a pull whose exception text cannot hold an unchecked identifier. Download and
# archive errors can quote URLs or member names from before the identity was known, so the
# manifest records only their type and stage.
_QUOTABLE_STAGES = {"listing", "scrub", "write", "reconcile"}


def pull_run(
    df, run_arn: str, dest: Path, allow_incomplete: bool = False, restart: bool = False, kind: str = "probe"
) -> dict:
    """Pull a run into `dest`, atomically, and return its manifest.

    Everything is written to `<dest>.partial` first and renamed to `dest` only when every job
    has been read and checked. A failed pull leaves `<dest>.partial` with a
    `pull-manifest.json` that says what failed (sanitised, and without any text that could not
    be checked); `restart=True` moves it aside to `<dest>.failed-<time>` (never deleted) before
    trying again. `dest` itself must not exist.

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
    stage, current = "listing", None

    def save_manifest() -> None:
        try:
            data = dump_sanitized(manifest, idents, account)
        except LeakFound:
            # Never write what could not be cleaned: keep only the bare outcome.
            data = dump_sanitized({"complete": False, "failure": {"stage": stage, "type": "LeakFound"}}, [], account)
        (partial / "pull-manifest.json").write_bytes(data)

    try:
        with tempfile.TemporaryDirectory(dir=partial.parent) as tmp:
            for number, (job, name) in enumerate(zip(all_jobs, slugs, strict=True)):
                current = name
                stage = "download"
                files, archives = _job_files(df, job, Path(tmp) / f"job{number}")
                stage = "scrub"
                ident = identify(files) if files else Identity()
                # Registered before scrubbing, so a refusal's diagnostics are cleaned with it too.
                idents.append(ident)
                unit = None
                if files:
                    files, unit = scrub(files, account, ident)
                stage = "write"
                folder = partial / name
                written = {rel: _write_new(folder / "artifacts" / rel, src) for rel, src in files.items()}
                missing = [f for f in expected if f not in files]
                manifest["jobs"][name] = {
                    "job": job["arn"].rsplit("/", 1)[-1],
                    "status": job["status"],
                    "result": job.get("result"),
                    "archives": archives,
                    "missing": missing,
                    "state": "complete" if not missing and job["status"] == "COMPLETED" else "incomplete",
                    "files": {rel: {"sha256": h, "bytes": n} for rel, (h, n) in written.items()},
                }
                _write_new(folder / "devicefarm-job.json", dump_sanitized(job, idents, account))
                if unit:
                    _write_new(folder / "unit.txt", (unit + "\n").encode())
        stage, current = "reconcile", None
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
        manifest["run_arn"] = run_arn
        _write_new(partial / "devicefarm-run.json", dump_sanitized(run, idents, account))
    except Exception as error:
        manifest["failure"] = {
            "stage": stage,
            "job": current,
            "type": type(error).__name__,
            "message": str(error)[:2000] if stage in _QUOTABLE_STAGES else None,
        }
        save_manifest()
        raise
    save_manifest()
    partial.rename(dest)
    return manifest
