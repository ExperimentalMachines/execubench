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


@pytest.fixture(scope="module")
def tls_files(tmp_path_factory):
    """A throwaway self-signed certificate for 127.0.0.1 (needs the openssl command)."""
    import shutil
    import subprocess

    if not shutil.which("openssl"):
        pytest.skip("openssl not installed")
    d = tmp_path_factory.mktemp("tls")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(d / "key.pem"),
            "-out",
            str(d / "cert.pem"),
        ],
        check=True,
        capture_output=True,
    )
    return d / "cert.pem", d / "key.pem"


def _serve_tls(handler, cert, key):
    import ssl

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_https_trickle_on_a_closing_connection_stops_at_the_deadline(tls_files):
    cert, key = tls_files
    httpd = _serve_tls(SlowChunkHeader, cert, key)
    try:

        def go():
            deadline = time.monotonic() + 0.3
            url = f"https://127.0.0.1:{httpd.server_port}/"
            with netio.request("GET", url, deadline, "t", sock_timeout=5, cafile=str(cert)) as r:
                for _ in netio.read_chunks(r, deadline, "t"):
                    pass

        elapsed = _timed(go)
    finally:
        httpd.shutdown()
    assert elapsed < 1.0


def test_a_stalled_tls_handshake_stops_at_the_deadline(tls_files):
    cert, _ = tls_files
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    held = []
    threading.Thread(target=lambda: held.append(server.accept()), daemon=True).start()  # accepts, never speaks TLS
    try:

        def go():
            url = f"https://127.0.0.1:{server.getsockname()[1]}/"
            with netio.request("GET", url, time.monotonic() + 0.3, "t", sock_timeout=5, cafile=str(cert)):
                pass

        elapsed = _timed(go)
    finally:
        server.close()
    assert elapsed < 1.0


def test_https_verifies_the_certificate(tls_files):
    import ssl

    cert, key = tls_files

    class Ok(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    httpd = _serve_tls(Ok, cert, key)
    url = f"https://127.0.0.1:{httpd.server_port}/"
    try:
        deadline = time.monotonic() + 5
        with netio.request("GET", url, deadline, "t", cafile=str(cert)) as r:
            assert b"".join(netio.read_chunks(r, deadline, "t")) == b"ok"
        with pytest.raises(ssl.SSLError), netio.request("GET", url, time.monotonic() + 5, "t"):
            pass  # the default trust store does not know the self-signed certificate
    finally:
        httpd.shutdown()


def test_proxy_credentials_are_sent(monkeypatch):
    seen = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.headers.get("Proxy-Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = _serve(Proxy)
    for name in ("no_proxy", "NO_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", f"http://us%40er:p%3Ass@127.0.0.1:{httpd.server_port}")
    try:
        with netio.request("GET", "http://destination.invalid/", time.monotonic() + 5, "t"):
            pass
    finally:
        httpd.shutdown()
    import base64

    assert seen == ["Basic " + base64.b64encode(b"us@er:p:ss").decode()]
