"""A streaming OpenAI chat-completions client with host monotonic timestamps.

Host timing (`host.sent_ns`, `first_byte_ns`, `done_ns`) is taken here; runner timing comes back
in ExecuServe's `x_execuserve` extension (docs/EXECUSERVE-CONTRACT.md) and is kept verbatim.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


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


def chat(base_url: str, key: str, body: dict, timeout_s: float = 900) -> Reply:
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    reply = Reply(sent_ns=time.monotonic_ns())
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    if reply.first_byte_ns is None:
                        reply.first_byte_ns = time.monotonic_ns()
                    reply.text += delta["content"]
                reply.tool_calls += delta.get("tool_calls") or []
                reply.finish_reason = choice.get("finish_reason") or reply.finish_reason
            reply.usage = event.get("usage") or reply.usage
            reply.extension = event.get("x_execuserve") or reply.extension
    reply.done_ns = time.monotonic_ns()
    return reply
