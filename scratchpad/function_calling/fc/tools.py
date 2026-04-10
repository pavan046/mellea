# pytest: skip_always
"""Tool conversion utilities for the fc sub-package.

Converts OpenAI tool schema dicts to MelleaTool instances with no-op callables,
so FCPipeline can accept tools from benchmarks without requiring executable functions.
The no-op callable is safe because FCPipeline never invokes tool.run(); it only
passes the schema to LoRA adapters for prompt construction.
"""

from __future__ import annotations

from typing import Any

from mellea.backends import tool as mellea_tool
from mellea.backends.tools import MelleaTool


def openai_tool_dict_to_mellea(schema: dict[str, Any]) -> MelleaTool:
    """Convert a single OpenAI tool schema dict to a MelleaTool with a no-op callable.

    Args:
        schema: OpenAI tool schema dict, e.g.:
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "...",
                    "parameters": {...}
                }
            }

    Returns:
        MelleaTool whose schema matches the input and whose callable is a no-op.

    Raises:
        ValueError: if the schema is missing required fields.
    """
    if schema.get("type") != "function":
        raise ValueError(
            f"Unsupported tool type: {schema.get('type')!r}. Only 'function' is supported."
        )

    func = schema.get("function")
    if not func or "name" not in func:
        raise ValueError(f"Tool schema missing 'function.name': {schema!r}")

    name = func["name"]

    def _noop(**kwargs: Any) -> None:  # noqa: ANN401
        pass

    _noop.__name__ = name

    return MelleaTool(name=name, tool_call=_noop, as_json_tool=schema)


def normalize_tools(tools: list[MelleaTool] | list[dict[str, Any]]) -> list[MelleaTool]:
    """Normalize a mixed or uniform tool list to list[MelleaTool].

    Accepts:
        - list[MelleaTool]: returned as-is.
        - list[dict]: each dict is converted via openai_tool_dict_to_mellea.

    Args:
        tools: tool list from the caller.

    Returns:
        list[MelleaTool] suitable for passing to Mellea backends.

    Raises:
        TypeError: if the list contains unsupported types.
        ValueError: if a dict schema is malformed.
    """
    if not tools:
        return []

    normalized: list[MelleaTool] = []
    for t in tools:
        if isinstance(t, MelleaTool):
            normalized.append(t)
        elif isinstance(t, dict):
            normalized.append(openai_tool_dict_to_mellea(t))
        else:
            raise TypeError(
                f"Unsupported tool type: {type(t).__name__}. Expected MelleaTool or dict."
            )
    return normalized
