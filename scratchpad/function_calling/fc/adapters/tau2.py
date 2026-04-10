# pytest: skip_always
"""TAU2 ↔ OpenAI protocol translation.

TAU2 uses the OpenAI Python SDK and sends fully OpenAI-compatible
ChatCompletion requests to POST /v1/chat/completions. Translation here is
minimal: unpack the request, call pipeline.run(), wrap the result in an
OpenAI ChatCompletion response envelope.

Request body (POST /v1/chat/completions):
    Standard OpenAI ChatCompletion request — messages, tools, model, etc.

Response body:
    Standard OpenAI ChatCompletion response — id, object, created, model,
    choices[0].message (with role, content, tool_calls), usage.
"""

from __future__ import annotations

import time
import uuid
from typing import Any


def tau2_request_to_openai(request: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Extract messages and tools from a TAU2 (OpenAI-compatible) request.

    TAU2 sends standard OpenAI ChatCompletion requests, so this is a
    direct extraction with no format translation.

    Args:
        request: OpenAI ChatCompletion request dict.

    Returns:
        Tuple of (messages, tools) in OpenAI format, ready for FCPipeline.run().
        tools is an empty list if not present in the request.
    """
    messages = request.get("messages", [])
    tools = request.get("tools") or []
    return messages, tools


def openai_response_to_tau2(
    assistant_msg: dict[str, Any], model: str = "fc-agent"
) -> dict[str, Any]:
    """Wrap an OpenAI assistant message in a full ChatCompletion response envelope.

    TAU2 uses the OpenAI Python SDK, which expects a complete ChatCompletion
    response object, not just the assistant message.

    Args:
        assistant_msg: OpenAI-format assistant message from FCPipeline.run().
        model: Model name to echo back in the response. Defaults to "fc-agent".

    Returns:
        OpenAI ChatCompletion response dict compatible with the OpenAI SDK.
    """
    tool_calls = assistant_msg.get("tool_calls")
    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_msg.get("content"),
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
