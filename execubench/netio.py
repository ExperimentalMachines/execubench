"""HTTP with a hard deadline: connection, request, headers and body all end by a fixed time.

A socket timeout bounds one blocking call, not an operation: a server that sends a byte just
before each timeout keeps a read alive indefinitely. Here a watchdog timer shuts the socket down
at the deadline, so whatever call is blocked returns at once, and the caller gets TimeoutError.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager


def remaining(deadline: float, what: str) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError(f"{what}: deadline passed")
    return left


class _Watchdog:
    def __init__(self, conn: http.client.HTTPConnection, deadline: float):
        self.fired = False
        self._conn = conn
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        sock = self._conn.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self) -> None:
        self._timer.cancel()


@contextmanager
def request(
    method: str,
    url: str,
    deadline: float,
    what: str,
    body=None,
    headers: dict | None = None,
    sock_timeout: float = 60.0,
) -> Iterator[http.client.HTTPResponse]:
    """An HTTP(S) response whose every byte must arrive before `deadline` (time.monotonic()).

    Anything still in progress at the deadline is cut off and raises TimeoutError, whatever error
    the cut-off socket produced.
    """
    parts = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = cls(parts.hostname, parts.port, timeout=min(sock_timeout, remaining(deadline, what)))
    watchdog = None
    try:
        conn.connect()
        watchdog = _Watchdog(conn, deadline)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn.request(method, target, body=body, headers=headers or {})
        resp = conn.getresponse()
        yield resp
        if watchdog.fired:
            raise TimeoutError(f"{what}: deadline passed")
    except (OSError, http.client.HTTPException) as error:
        if (watchdog and watchdog.fired) or time.monotonic() >= deadline:
            raise TimeoutError(f"{what}: deadline passed") from error
        raise
    finally:
        if watchdog:
            watchdog.cancel()
        conn.close()


def read_chunks(resp: http.client.HTTPResponse, deadline: float, what: str, size: int = 1 << 20) -> Iterator[bytes]:
    """The body as it arrives (read1 returns after one socket read), checked against the deadline.
    A body shorter than its Content-Length raises IncompleteRead instead of ending quietly."""
    expected = resp.length
    got = 0
    while chunk := resp.read1(size):
        remaining(deadline, what)
        got += len(chunk)
        yield chunk
    if expected is not None and got < expected:
        raise http.client.IncompleteRead(b"", expected - got)
