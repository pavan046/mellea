# pytest: skip_always
"""LoRA inference nodes for the function-calling system.

Each node encapsulates exactly one LoRA call. Pipelines wire nodes together
using plain Python — no framework, no registry, no graph specification.

Node contract:
    - __init__ registers the adapter via ensure_adapter and holds a reference
      to the backend.
    - __call__ builds the ChatContext (prepending system_prompt from io.yaml
      if present), calls the LoRA, parses the output, and returns a typed result.
    - Nodes are stateless. The same node instance can be called repeatedly with
      different contexts.

Adding a new pipeline means instantiating the relevant nodes and wiring them
in a new flow class — not copying LoRA call logic.

See .claude/discussions/function-calling-system-design.md §7 for the design.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import mellea.stdlib.functional as mfuncs
from fc.lora import LocalIntrinsicAdapter, ensure_adapter
from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.tools import MelleaTool
from mellea.core import FancyLogger
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import Intrinsic
from mellea.stdlib.context import ChatContext

logger = FancyLogger.get_logger()

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Node(ABC):
    """Abstract base class for all LoRA inference nodes.

    A node wraps a single LoRA call. It registers its adapter at construction
    time and exposes a __call__ method that runs inference on a ChatContext.

    Subclasses must implement __call__.
    """

    @abstractmethod
    def __call__(self, ctx: ChatContext, **kwargs: Any) -> Any:
        """Run the node on the given context.

        Args:
            ctx: The conversation context built from OpenAI-format messages.
            **kwargs: Node-specific arguments (e.g. tools for executor nodes).

        Returns:
            Node-specific output type. See subclass docstrings.
        """
        ...

    def _system_prompt(self, backend: LocalHFBackend, adapter_name: str) -> str | None:
        """Return the system_prompt for a registered adapter, or None."""
        qualified = f"{adapter_name}_lora"
        adapter = backend._added_adapters.get(qualified)
        if isinstance(adapter, LocalIntrinsicAdapter):
            return adapter.system_prompt
        return None

    def _build_ctx(
        self,
        ctx: ChatContext,
        backend: LocalHFBackend,
        adapter_name: str,
        tool_summary: str | None = None,
    ) -> ChatContext:
        """Return a new context with system_prompt at position 0, if defined.

        ChatContext is an immutable linked list — there is no prepend operation.
        To guarantee the system message appears before all conversation messages,
        this method rebuilds the context: start a fresh ChatContext with the
        system message first, then replay all existing messages on top.

        If the existing context already has a system message at position 0 (from
        the caller), it is replaced by the node's system prompt. This prevents
        duplicate system messages when the caller and the adapter both define one.

        If tool_summary is provided, it is appended to the system prompt as a
        plain-text tool listing. Used by RouterNode so the router can reason about
        available tools without receiving them as structured tool-call objects.

        The returned context is node-scoped and discarded after the LoRA call.
        The original ctx passed in is never mutated.
        """
        system_prompt = self._system_prompt(backend, adapter_name)
        if not system_prompt and not tool_summary:
            return ctx

        messages = ctx.as_list()

        # Drop an existing system message at position 0 so we don't get two.
        if (
            messages
            and isinstance(messages[0], Message)
            and messages[0].role == "system"
        ):
            messages = messages[1:]

        combined_system = system_prompt or ""
        if tool_summary:
            combined_system = (
                combined_system.rstrip() + "\n\n" + tool_summary
            ).lstrip()

        new_ctx = ChatContext()
        new_ctx = new_ctx.add(Message("system", combined_system))
        for msg in messages:
            new_ctx = new_ctx.add(msg)
        return new_ctx

    @staticmethod
    def _tools_to_summary(tools: list[MelleaTool]) -> str:
        """Serialize tool names and descriptions as a plain-text listing.

        Used to inform the router about available tools without injecting them
        as structured tool-call objects (which would activate the model's
        tool-calling machinery instead of its classification behaviour).

        Example output:
            Available tools:
            - get_weather(location): Get the current weather for a location.
            - search_flights(origin, destination, date): Search for flights.
        """
        if not tools:
            return ""
        lines = ["Available tools:"]
        for t in tools:
            schema = t.as_json_tool or {}
            func = schema.get("function", {})
            name = func.get("name", t.name)
            description = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys())
            lines.append(f"- {name}({param_names}): {description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RouterNode
# ---------------------------------------------------------------------------


class RouterNode(Node):
    """Classifies a conversation into a routing category using the router LoRA.

    Calls fc_router and parses its JSON output into a category string.

    Args:
        backend: Loaded LocalHFBackend instance.
        adapter_path: Local path to the router LoRA checkpoint directory.
    """

    _ADAPTER = "fc_router"

    def __init__(self, backend: LocalHFBackend, adapter_path: str) -> None:
        self._backend = backend
        ensure_adapter(self._ADAPTER, adapter_path, backend)

    def __call__(
        self, ctx: ChatContext, tools: list[MelleaTool] | None = None, **kwargs: Any
    ) -> str:
        """Classify the conversation and return a routing category.

        Tools are serialized as a plain-text listing appended to the system
        prompt rather than passed as structured tool-call objects. This gives
        the router semantic awareness of available tools without activating the
        model's tool-calling machinery.

        Args:
            ctx: Conversation context.
            tools: Available tools. Summarized into the system prompt for routing
                awareness. Pass None or empty list if no tools are available.

        Returns:
            One of "parallel", "multi_step", or "conversational".
        """
        # TODO: replace print statements with proper logging (e.g. FancyLogger)
        tool_summary = self._tools_to_summary(tools) if tools else None
        router_ctx = self._build_ctx(
            ctx, self._backend, self._ADAPTER, tool_summary=tool_summary
        )
        logger.debug("Running %s", self._ADAPTER)
        mot, _ = mfuncs.act(
            Intrinsic(self._ADAPTER),
            router_ctx,
            self._backend,
            model_options={ModelOption.TEMPERATURE: 0.0},
            strategy=None,
        )
        assert mot.is_computed()
        category = self._parse_category(mot.value or "")
        logger.debug("%s → %s", self._ADAPTER, category)
        return category

    @staticmethod
    def _parse_category(raw: str) -> str:
        try:
            parsed = json.loads(raw)
            category = parsed.get("category", "conversational")
        except json.JSONDecodeError:
            category = "conversational"
            for cat in ("parallel", "multi_step", "conversational"):
                if cat in raw.lower():
                    category = cat
                    break
        valid = {"parallel", "multi_step", "conversational"}
        return category if category in valid else "conversational"


# ---------------------------------------------------------------------------
# ExecutorNode
# ---------------------------------------------------------------------------


class ExecutorNode(Node):
    """Generates a list of tool calls using a tool-calling LoRA.

    Handles both parallel and multi-step executor adapters; they share the
    same call and parse logic and differ only in the adapter they load.

    Args:
        backend: Loaded LocalHFBackend instance.
        adapter_path: Local path to the executor LoRA checkpoint directory.
        adapter_name: Logical adapter name (e.g. "fc_parallel", "fc_multi_step").
    """

    def __init__(
        self, backend: LocalHFBackend, adapter_path: str, adapter_name: str
    ) -> None:
        self._backend = backend
        self._adapter_name = adapter_name
        ensure_adapter(adapter_name, adapter_path, backend)

    def __call__(
        self, ctx: ChatContext, tools: list[MelleaTool], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate tool calls for the given context.

        Args:
            ctx: Conversation context.
            tools: Available tools. Required; executor nodes always need the
                tool list to generate valid call arguments.

        Returns:
            OpenAI-format assistant message dict with tool_calls populated:
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "call_N", "type": "function",
                                 "function": {"name": "...", "arguments": "..."}}]}
            On parse failure, returns a conversational error message.
        """
        executor_ctx = self._build_ctx(ctx, self._backend, self._adapter_name)
        logger.debug("Running %s", self._adapter_name)
        mot, _ = mfuncs.act(
            Intrinsic(self._adapter_name),
            executor_ctx,
            self._backend,
            model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
            strategy=None,
        )
        assert mot.is_computed()
        logger.debug("%s done", self._adapter_name)
        return _parse_tool_calls(mot.value or "")


# ---------------------------------------------------------------------------
# ConversationalNode
# ---------------------------------------------------------------------------


class ConversationalNode(Node):
    """Generates a natural language response using the conversational LoRA.

    Used when the router classifies the request as not requiring tool calls.

    Args:
        backend: Loaded LocalHFBackend instance.
        adapter_path: Local path to the conversational LoRA checkpoint directory.
    """

    _ADAPTER = "fc_conversational"

    def __init__(self, backend: LocalHFBackend, adapter_path: str) -> None:
        self._backend = backend
        ensure_adapter(self._ADAPTER, adapter_path, backend)

    def __call__(self, ctx: ChatContext, **kwargs: Any) -> dict[str, Any]:
        """Generate a natural language response.

        Args:
            ctx: Conversation context.

        Returns:
            OpenAI-format assistant message dict with content populated:
                {"role": "assistant", "content": "...", "tool_calls": None}
        """
        conv_ctx = self._build_ctx(ctx, self._backend, self._ADAPTER)
        logger.debug("Running %s", self._ADAPTER)
        mot, _ = mfuncs.act(
            Intrinsic(self._ADAPTER),
            conv_ctx,
            self._backend,
            model_options={ModelOption.TEMPERATURE: 0.0},
            strategy=None,
        )
        assert mot.is_computed()
        logger.debug("%s done", self._ADAPTER)
        raw = mot.value or ""
        try:
            content = json.loads(raw).get("response", raw)
        except json.JSONDecodeError:
            content = raw
        return {"role": "assistant", "content": content, "tool_calls": None}


# ---------------------------------------------------------------------------
# Shared parse helper
# ---------------------------------------------------------------------------


def _parse_tool_calls(raw: str) -> dict[str, Any]:
    """Parse a JSON tool call array from raw model output.

    Handles both bare array [...] and wrapped {"tool_calls": [...]} formats.

    Returns:
        OpenAI-format assistant message dict with tool_calls populated,
        or a conversational error message if parsing fails.
    """
    if not raw:
        return {
            "role": "assistant",
            "content": "[error: model produced no output]",
            "tool_calls": None,
        }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "role": "assistant",
            "content": f"[error: tool call output is not valid JSON — {exc} | raw={raw!r}]",
            "tool_calls": None,
        }

    if isinstance(parsed, dict):
        parsed = parsed.get("tool_calls", [])

    if not isinstance(parsed, list):
        return {
            "role": "assistant",
            "content": f"[error: expected list of tool calls, got {type(parsed).__name__}]",
            "tool_calls": None,
        }

    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("arguments", {})),
            },
        }
        for i, tc in enumerate(parsed)
    ]
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}
