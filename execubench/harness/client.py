"""A streaming OpenAI chat-completions client with host monotonic timestamps.

Host timing (`host.sent_ns`, `first_byte_ns`, `done_ns`) is taken here; runner timing comes back
in ExecuServe's `x_execuserve` extension (docs/EXECUSERVE-CONTRACT.md) and is kept verbatim.
A stream counts as complete only if it ends with `[DONE]` after a finish reason; anything else
raises IncompleteStream with whatever text arrived, so a cut stream is never scored as a reply.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


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
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    reply = Reply(sent_ns=time.monotonic_ns())
    calls: dict[int, dict] = {}
    deadline = time.monotonic() + timeout_s
    try:
        _read_stream(req, reply, calls, deadline, timeout_s)
    except (OSError, ValueError) as error:  # connection reset, timeout or a malformed event
        reply.done_ns = time.monotonic_ns()
        raise IncompleteStream(f"stream failed: {type(error).__name__} {error}", reply) from error
    reply.done_ns = time.monotonic_ns()
    reply.tool_calls = [calls[i] for i in sorted(calls)]
    if reply.finish_reason is None:
        raise IncompleteStream("stream ended without a finish reason", reply)
    return reply


def _read_stream(req, reply: Reply, calls: dict, deadline: float, timeout_s: float) -> None:
    done = False
    with urllib.request.urlopen(req, timeout=min(timeout_s, 120)) as resp:
        for raw in resp:
            if time.monotonic() > deadline:
                raise IncompleteStream(f"no completion within {timeout_s} s", reply)
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
