"""Put a model file on the phone and prove the bytes are the pinned ones, on host and phone."""

from __future__ import annotations

import hashlib
import shlex
import time
from pathlib import Path

from .adb import Adb


class StagingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, revision: str, filename: str, expected_sha256: str, cache: Path) -> tuple[Path, float]:
    """Download one file at a pinned revision and check its sha256 before anything uses it."""
    from huggingface_hub import hf_hub_download

    t0 = time.monotonic()
    path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=cache))
    if sha256_file(path) != expected_sha256:
        raise StagingError(f"{repo}/{filename}@{revision[:8]}: sha256 does not match the manifest")
    return path, time.monotonic() - t0


def push(adb: Adb, local: Path, remote: str, expected_sha256: str) -> float:
    """adb push into a folder the app created (adb-created folders are unreadable by the app),
    then hash the copy on the phone."""
    t0 = time.monotonic()
    adb.run("push", str(local), remote, timeout_s=1800)
    seconds = time.monotonic() - t0
    out = adb.shell(f"sha256sum {shlex.quote(remote)}", timeout_s=600).split()
    if not out or out[0] != expected_sha256:
        raise StagingError(f"{remote}: sha256 on the phone does not match the manifest")
    return seconds
