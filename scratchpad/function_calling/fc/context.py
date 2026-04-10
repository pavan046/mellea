# pytest: skip_always
"""ChatContext reconstruction from OpenAI-format message dicts.

FCPipeline receives a full message history on every call (stateless contract).
This module converts that list of OpenAI-format dicts back into a Mellea
ChatContext so the LoRA adapters can consume it.

OpenAI message roles handled:
    user        → Message("user", content)
    assistant   → Message("assistant", content) or assistant message with tool_calls
    tool        → ToolMessage(role="tool", content, name, tool_call_id)
    system      → Message("system", content)
"""

from __future__ import annotations

import json
from typing import Any

from mellea.stdlib.components import Message
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.context import ChatContext


def messages_to_chat_context(messages: list[dict[str, Any]]) -> ChatContext:
    r"""Reconstruct a ChatContext from a list of OpenAI-format message dicts.

    Args:
        messages: OpenAI-style message list, e.g.:
            [
                {"role": "user", "content": "What is the weather in SF?"},
                {"role": "assistant", "content": null,
                 "tool_calls": [{"id": "...", "type": "function",
                                 "function": {"name": "get_weather",
                                              "arguments": "{\"location\": \"SF\"}"}}]},
                {"role": "tool", "tool_call_id": "...", "content": "{\"temp_f\": 68}"}
            ]

    Returns:
        ChatContext populated with the corresponding Mellea Message objects.
    """
    ctx = ChatContext()
    for msg in messages:
        ctx = ctx.add(_openai_dict_to_message(msg))
    return ctx


def _openai_dict_to_message(msg: dict[str, Any]) -> Message | ToolMessage:
    """Convert a single OpenAI message dict to a Mellea Message or ToolMessage."""
    role = msg.get("role", "")
    content = msg.get("content") or ""

    if role == "tool":
        return ToolMessage(
            role="tool",
            content=content,
            tool_output=_try_parse_json(content),
            name=msg.get("name", ""),
            args={},
            tool=None,  # type: ignore[arg-type]
        )

    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Serialize tool_calls back to a JSON string stored in content.
            # Mellea's Message has no native tool_calls field; the assistant
            # message content carries the raw JSON for LoRA context.
            serialized = json.dumps(
                [
                    {
                        "name": tc["function"]["name"],
                        "arguments": _try_parse_json(
                            tc["function"].get("arguments", "{}")
                        ),
                    }
                    for tc in tool_calls
                ]
            )
            return Message(role="assistant", content=serialized)
        return Message(role="assistant", content=content)

    # user, system, or any other role
    return Message(role=role, content=content)


def _try_parse_json(value: str) -> Any:
    """Parse a JSON string, returning the original string on failure."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
