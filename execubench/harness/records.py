"""Crash-safe JSONL records: every line is flushed and fsynced before the next request starts,
so a job killed at Device Farm's time limit keeps everything it reached."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            self._f.close()


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def protocol_sha256(protocol: dict) -> str:
    """Canonical hash of a job's protocol block, without its own hash field."""
    body = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_sha256(job: dict) -> str:
    """Canonical hash of everything that defines a configuration (docs/METRICS.md, cell key)."""
    keys = ("model", "tokenizer_sha256", "runtime", "generation", "protocol", "device_id")
    canonical = {k: job.get(k) for k in keys}
    protocol = dict(canonical.get("protocol") or {})
    protocol.pop("protocol_sha256", None)
    canonical["protocol"] = protocol
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
