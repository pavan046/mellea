# pytest: skip_always
"""BFCL ↔ OpenAI protocol translation.

BFCL's /chat endpoint receives messages in OpenAI format and tools in OpenAI
format. The response it expects back is a ChatResponse with type "tool_calls"
or "response".

Wire format (confirmed from internal implementation and BFCL source):

Request body (POST /chat):
    {
        "id": "optional-str",
        "messages": [{"role": "user|assistant|tool", "content": "str"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "str",
                    "description": "str",
                    "parameters": {"type": "object", "properties": {...}, "required": [...]}
                }
            }
        ]
    }

Response body:
    {
        "type": "tool_calls" | "response" | "error",
        "tool_calls": [{"name": "str", "arguments": {...}}],  # when type == "tool_calls"
        "content": "str"                                       # when type == "response"
    }

FCPipeline returns an OpenAI assistant message dict. This module translates
in both directions.
"""

from __future__ import annotations

import json
from typing import Any


def bfcl_request_to_openai(request: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Extract messages and tools from a BFCL /chat request.

    BFCL sends messages and tools in OpenAI format, so this is mostly
    a pass-through with light normalization.

    Args:
        request: BFCL /chat request dict with "messages" and "tools" fields.

    Returns:
        Tuple of (messages, tools) in OpenAI format, ready for FCPipeline.run().
    """
    messages = request.get("messages", [])
    tools = request.get("tools", [])

    # Normalize any assistant messages that have tool_calls serialized as
    # content strings (from prior turns in history).
    normalized_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Pass tool_calls through if present as structured field.
        if role == "assistant" and msg.get("tool_calls"):
            normalized_messages.append(msg)
            continue

        # Ensure content is a string — BFCL may send None for assistant
        # messages that had tool calls (content: null in OpenAI format).
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content)

        normalized_messages.append(
            {k: v for k, v in msg.items() if k != "content"} | {"content": content}
        )

    return normalized_messages, tools


def openai_response_to_bfcl(assistant_msg: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI assistant message to BFCL /chat response format.

    Args:
        assistant_msg: OpenAI-format assistant message from FCPipeline.run().
            Either {"role": "assistant", "content": "...", "tool_calls": null}
            or     {"role": "assistant", "content": null, "tool_calls": [...]}

    Returns:
        BFCL ChatResponse dict:
            {"type": "tool_calls", "tool_calls": [{"name": ..., "arguments": {...}}]}
            or
            {"type": "response", "content": "..."}
    """
    tool_calls = assistant_msg.get("tool_calls")

    if tool_calls:
        bfcl_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_args
            bfcl_tool_calls.append({"name": name, "arguments": arguments})
        return {"type": "tool_calls", "tool_calls": bfcl_tool_calls}

    content = assistant_msg.get("content") or ""
    return {"type": "response", "content": content}
