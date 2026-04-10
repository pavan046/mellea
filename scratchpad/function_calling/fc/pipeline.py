# pytest: skip_always
"""FunctionCallingPipeline — stateless, single-pass FC inference.

Accepts a full OpenAI-format message history and a tool list, routes the request
to the appropriate LoRA adapter, and returns an OpenAI-format assistant message.
No state is held between calls. No tools are executed.

System prompts are defined in each adapter's io.yaml via the `system_prompt` field
(our extension to Mellea's schema, handled by LocalIntrinsicAdapter). If an adapter
has no system_prompt in io.yaml, no system message is prepended.

See scratchpad/function_calling/function-calling-system-design.md for the full design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fc.context import messages_to_chat_context
from fc.nodes import ConversationalNode, ExecutorNode, RouterNode
from fc.tools import normalize_tools
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.tools import MelleaTool

# Maps subdirectory names under adapters_dir to logical adapter names.
# This is the convention owned by FunctionCallingPipeline. A different
# pipeline would define its own mapping.
_DIR_TO_ADAPTER: dict[str, str] = {
    "router": "fc_router",
    "parallel_tool_calling": "fc_parallel",
    "multi_step_tool_calling": "fc_multi_step",
    "conversational_detection": "fc_conversational",
}


def resolve_adapter_paths(
    adapters_dir: str | Path, adapter_overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Resolve logical adapter names to local paths from a directory.

    Scans adapters_dir for subdirectories whose names match the known
    convention, then applies any overrides on top.

    Args:
        adapters_dir: Root directory containing adapter subdirectories.
            Expected subdirs: router, parallel_tool_calling,
            multi_step_tool_calling, conversational_detection.
        adapter_overrides: Optional mapping of logical adapter name to path.
            Entries here take precedence over auto-discovered paths.
            Example: {"fc_router": "/experiments/my_custom_router"}

    Returns:
        Dict mapping logical adapter name to resolved local path.

    Raises:
        FileNotFoundError: if adapters_dir does not exist.
        ValueError: if any required adapter subdirectory is missing and no
            override is provided for it.
    """
    root = Path(adapters_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"adapters_dir does not exist: {root}")

    resolved: dict[str, str] = {}
    for dir_name, adapter_name in _DIR_TO_ADAPTER.items():
        candidate = root / dir_name
        if candidate.is_dir():
            resolved[adapter_name] = str(candidate)

    if adapter_overrides:
        resolved.update(adapter_overrides)

    required = set(_DIR_TO_ADAPTER.values())
    missing = required - resolved.keys()
    if missing:
        raise ValueError(
            f"Missing adapter directories in {root}. "
            f"Could not resolve: {sorted(missing)}. "
            "Either add the expected subdirectories or supply adapter_overrides."
        )

    return resolved


class FunctionCallingPipeline:
    """Stateless, single-pass function-calling pipeline.

    Accepts an OpenAI-format message history and a tool list, routes the
    request to the appropriate LoRA adapter via nodes, and returns an
    OpenAI-format assistant message dict. No state is held between calls.
    No tools are executed.

    System prompts are sourced from each adapter's io.yaml (system_prompt
    field). No prompts are hardcoded in this class.

    Adapter discovery follows a directory naming convention. Pass adapters_dir
    to auto-discover all adapters from a standard layout. Use adapter_overrides
    to substitute individual adapters without abandoning auto-discovery.

    Args:
        backend: Loaded LocalHFBackend instance.
        adapters_dir: Root directory containing adapter subdirectories following
            the naming convention (router, parallel_tool_calling, etc.).
            Either this or adapter_paths must be provided.
        adapter_overrides: Optional mapping of logical adapter name to local path.
            Applied on top of auto-discovery. Entries here win over discovered paths.
            Example: {"fc_router": "/experiments/my_custom_router"}
        adapter_paths: Explicit mapping of logical adapter name to local path.
            Bypasses auto-discovery entirely. Use only when you need full manual
            control over all adapter locations.

    Raises:
        ValueError: if neither adapters_dir nor adapter_paths is provided, or
            if adapters_dir is missing required subdirectories with no override.
    """

    def __init__(
        self,
        backend: LocalHFBackend,
        adapters_dir: str | Path | None = None,
        adapter_overrides: dict[str, str] | None = None,
        adapter_paths: dict[str, str] | None = None,
    ) -> None:
        if adapters_dir is not None:
            paths = resolve_adapter_paths(adapters_dir, adapter_overrides)
        elif adapter_paths is not None:
            paths = adapter_paths
        else:
            raise ValueError("Either adapters_dir or adapter_paths must be provided.")

        self._router = RouterNode(backend, paths["fc_router"])
        self._parallel = ExecutorNode(backend, paths["fc_parallel"], "fc_parallel")
        self._multi_step = ExecutorNode(
            backend, paths["fc_multi_step"], "fc_multi_step"
        )
        self._conversational = ConversationalNode(backend, paths["fc_conversational"])

    def run(
        self,
        messages: list[dict[str, Any]],
        tools: list[MelleaTool] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a single inference pass and return an OpenAI-format assistant message.

        Args:
            messages: Full OpenAI-format conversation history ending with the
                most recent user message.
            tools: Available tools as MelleaTool instances or OpenAI tool schema
                dicts. Dicts are converted to schema-only MelleaTool internally.

        Returns:
            OpenAI assistant message dict. When tool calls are produced:
                {"role": "assistant", "content": null,
                 "tool_calls": [{"id": "...", "type": "function",
                                 "function": {"name": "...", "arguments": "..."}}]}
            When a conversational response is produced:
                {"role": "assistant", "content": "...", "tool_calls": null}
        """
        mellea_tools = normalize_tools(tools)
        ctx = messages_to_chat_context(messages)
        category = self._router(ctx, tools=mellea_tools)

        if category == "parallel":
            return self._parallel(ctx, tools=mellea_tools)
        elif category == "multi_step":
            return self._multi_step(ctx, tools=mellea_tools)
        else:
            return self._conversational(ctx)
