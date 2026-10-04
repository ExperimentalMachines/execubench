"""A streaming OpenAI chat-completions client with host monotonic timestamps.

Host timing (`host.sent_ns`, `first_byte_ns`, `done_ns`) is taken here; runner timing comes back
in ExecuServe's `x_execuserve` extension (docs/EXECUSERVE-CONTRACT.md) and is kept verbatim.
A stream counts as complete only if it ends with `[DONE]` after a finish reason; anything else
raises IncompleteStream with whatever text arrived, so a cut stream is never scored as a reply.
"""

from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass, field

from .. import netio


class IncompleteStream(RuntimeError):
    def __init__(self, message: str, partial: Reply):
        super().__init__(message)
        self.partial = partial


@dataclass
class Reply:
    text: str = ""
    finish_reason: str | None = None
    tool_calls: list = field(default_factory=list)
    extension: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    sent_ns: int = 0
    first_byte_ns: int | None = None
    done_ns: int = 0


def _merge_tool_call(calls: dict[int, dict], fragment: dict) -> None:
    """OpenAI streams a tool call in pieces keyed by index: id and name once, arguments in parts."""
    call = calls.setdefault(
        fragment.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
    )
    call["id"] = fragment.get("id") or call["id"]
    fn = fragment.get("function") or {}
    call["function"]["name"] += fn.get("name") or ""
    call["function"]["arguments"] += fn.get("arguments") or ""


def chat(base_url: str, key: str, body: dict, timeout_s: float = 900) -> Reply:
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "Content-Length": str(len(data))}
    reply = Reply(sent_ns=time.monotonic_ns())
    calls: dict[int, dict] = {}
    deadline = time.monotonic() + timeout_s
    try:
        _read_stream(base_url.rstrip("/") + "/v1/chat/completions", data, headers, reply, calls, deadline)
    except IncompleteStream:
        raise
    except TimeoutError as error:  # the watchdog cut the connection at the deadline
        raise IncompleteStream(f"no completion within {timeout_s} s", reply) from error
    except (OSError, ValueError, http.client.HTTPException) as error:  # reset, refused or a malformed event
        raise IncompleteStream(f"stream failed: {type(error).__name__} {error}", reply) from error
    finally:
        # Every exit, clean or not, leaves the partial reply whole: its end time and the tool
        # calls assembled so far (an IncompleteStream carries this same object).
        reply.done_ns = reply.done_ns or time.monotonic_ns()
        reply.tool_calls = [calls[i] for i in sorted(calls)]
    if reply.finish_reason is None:
        raise IncompleteStream("stream ended without a finish reason", reply)
    return reply


def _lines(chunks):
    buf = b""
    for chunk in chunks:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line
    if buf:
        yield buf


def _read_stream(url: str, data: bytes, headers: dict, reply: Reply, calls: dict, deadline: float) -> None:
    """Read SSE lines until [DONE], everything (connect, headers, body) before `deadline`: a
    watchdog cuts the connection off then, however slowly the bytes trickle (execubench/netio.py)."""
    done = False
    with netio.request("POST", url, deadline, "stream", body=data, headers=headers, sock_timeout=120) as resp:
        if resp.status != 200:
            raise ValueError(f"HTTP {resp.status}")
        for raw in _lines(netio.read_chunks(resp, deadline, "stream", 1 << 16)):
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                reply.done_ns = time.monotonic_ns()
                raise IncompleteStream(f"server error event: {event['error']}", reply)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    if reply.first_byte_ns is None:
                        reply.first_byte_ns = time.monotonic_ns()
                    reply.text += delta["content"]
                for fragment in delta.get("tool_calls") or []:
                    if reply.first_byte_ns is None:
                        reply.first_byte_ns = time.monotonic_ns()
                    _merge_tool_call(calls, fragment)
                reply.finish_reason = choice.get("finish_reason") or reply.finish_reason
            reply.usage = event.get("usage") or reply.usage
            reply.extension = event.get("x_execuserve") or reply.extension
    if not done:
        raise ValueError("stream ended before [DONE]")
