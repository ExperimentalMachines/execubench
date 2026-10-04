"""execubench/netio.py: the deadline holds at every stage, against local servers only."""

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from execubench import netio


def _serve(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class SlowChunkHeader(BaseHTTPRequestHandler):
    """A chunked, connection-close response whose first chunk-size line never finishes."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for _ in range(200):
                self.wfile.write(b"1")
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            pass


def _timed(fn):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        fn()
    return time.monotonic() - started


def test_a_trickling_chunk_header_on_a_closing_connection_stops_at_the_deadline():
    httpd = _serve(SlowChunkHeader)
    try:

        def go():
            deadline = time.monotonic() + 0.3
            with netio.request("GET", f"http://127.0.0.1:{httpd.server_port}/", deadline, "t", sock_timeout=5) as r:
                for _ in netio.read_chunks(r, deadline, "t"):
                    pass

        elapsed = _timed(go)
    finally:
        httpd.shutdown()
    assert elapsed < 1.0


def test_slow_name_resolution_stops_at_the_deadline(monkeypatch):
    real = socket.getaddrinfo

    def slow(*args, **kwargs):
        time.sleep(2)
        return real(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow)

    def go():
        with netio.request("GET", "http://127.0.0.1:9/", time.monotonic() + 0.2, "t"):
            pass

    assert _timed(go) < 0.8


def test_an_environment_proxy_is_used(monkeypatch):
    seen = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    httpd = _serve(Proxy)
    for name in ("no_proxy", "NO_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{httpd.server_port}")
    try:
        deadline = time.monotonic() + 5
        with netio.request("GET", "http://destination.invalid/a?b=1", deadline, "t") as r:
            body = b"".join(netio.read_chunks(r, deadline, "t"))
    finally:
        httpd.shutdown()
    assert body == b"ok" and seen == ["http://destination.invalid/a?b=1"]


def test_a_short_body_is_not_a_success():
    class Short(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.write(b"abc")

    httpd = _serve(Short)
    try:
        deadline = time.monotonic() + 5
        with pytest.raises(Exception, match="IncompleteRead|incomplete"):  # noqa: B017
            with netio.request("GET", f"http://127.0.0.1:{httpd.server_port}/", deadline, "t") as r:
                list(netio.read_chunks(r, deadline, "t"))
    finally:
        httpd.shutdown()
