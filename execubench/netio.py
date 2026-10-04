"""HTTP with a hard deadline: resolution, connection, TLS, request, headers and body all end by
a fixed time.

A socket timeout bounds one blocking call, not an operation: a server that sends a byte just
before each timeout keeps a read alive indefinitely, and name resolution has no timeout at all.
Here a watchdog is armed before anything else. Resolution runs in a worker thread that is waited
for only until the deadline; every socket is handed to the watchdog the moment it is created, so
at the deadline the watchdog shuts the transport down and whatever call is blocked on it (TCP
connect, TLS handshake, a chunk-size line, a body read) returns at once. The caller then gets
TimeoutError. The watchdog holds the transport socket itself, because http.client drops its own
reference when a response will close the connection.

Proxies come from the environment as urllib would find them (`getproxies`, `proxy_bypass`):
HTTP through the proxy with an absolute URL, HTTPS through a CONNECT tunnel, with the
destination's certificate verified, all under the same deadline.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager


def remaining(deadline: float, what: str) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError(f"{what}: deadline passed")
    return left


class _Watchdog:
    """Shuts every socket it was given down at the deadline."""

    def __init__(self, deadline: float):
        self.fired = False
        self._lock = threading.Lock()
        self._socks: list[socket.socket] = []
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def attach(self, sock: socket.socket) -> None:
        with self._lock:
            self._socks.append(sock)
            fired = self.fired
        if fired:
            self._shutdown(sock)

    @staticmethod
    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _fire(self) -> None:
        with self._lock:
            self.fired = True
            socks = list(self._socks)
        for sock in socks:
            self._shutdown(sock)

    def cancel(self) -> None:
        self._timer.cancel()
        with self._lock:
            self._socks.clear()


def _resolve(host: str, port: int, deadline: float, what: str) -> list[tuple]:
    """getaddrinfo in a daemon thread, waited for only until the deadline (it has no timeout)."""
    box: dict = {}

    def work() -> None:
        try:
            box["addrs"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            box["error"] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(remaining(deadline, what))
    if worker.is_alive():
        raise TimeoutError(f"{what}: name resolution did not finish by the deadline")
    if "error" in box:
        raise box["error"]
    return box["addrs"]


def _connector(watchdog: _Watchdog, deadline: float, what: str):
    """A replacement for http.client's socket.create_connection: bounded resolution, then each
    address tried with the time left, every socket attached to the watchdog before connecting."""

    def create_connection(address, timeout=None, source_address=None, *args, **kwargs):
        host, port = address[0], address[1]
        last: OSError | None = None
        for family, kind, proto, _, addr in _resolve(host, port, deadline, what):
            sock = socket.socket(family, kind, proto)
            watchdog.attach(sock)
            try:
                sock.settimeout(min(timeout or 60.0, remaining(deadline, what)))
                if source_address:
                    sock.bind(source_address)
                sock.connect(addr)
                return sock
            except OSError as error:
                sock.close()
                last = error
                remaining(deadline, what)
        raise last or OSError(f"{what}: no address for {host}")

    return create_connection


def _proxy_for(parts: urllib.parse.SplitResult) -> urllib.parse.SplitResult | None:
    if urllib.request.proxy_bypass(parts.hostname or ""):
        return None
    proxy = urllib.request.getproxies().get(parts.scheme)
    return urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}") if proxy else None


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
    the cut-off socket produced; a response consumed after the deadline is not a success either.
    """
    parts = urllib.parse.urlsplit(url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    proxy = _proxy_for(parts)
    timeout = min(sock_timeout, remaining(deadline, what))
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            proxy.hostname if proxy else parts.hostname,
            (proxy.port or 80) if proxy else port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        if proxy:
            conn.set_tunnel(parts.hostname, port)
    else:
        conn = http.client.HTTPConnection(
            proxy.hostname if proxy else parts.hostname, (proxy.port or 80) if proxy else port, timeout=timeout
        )
        if proxy:
            target = url
    watchdog = _Watchdog(deadline)
    conn._create_connection = _connector(watchdog, deadline, what)  # noqa: SLF001 - http.client's hook
    resp = None
    try:
        conn.request(method, target, body=body, headers=headers or {})
        resp = conn.getresponse()
        yield resp
        if watchdog.fired or time.monotonic() >= deadline:
            raise TimeoutError(f"{what}: deadline passed")
    except (OSError, http.client.HTTPException) as error:
        if watchdog.fired or time.monotonic() >= deadline:
            raise TimeoutError(f"{what}: deadline passed") from error
        raise
    finally:
        watchdog.cancel()
        # The response holds its own reference to the socket; closing only the connection would
        # leave the descriptor open, and the server would never see the connection end.
        if resp is not None:
            resp.close()
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
