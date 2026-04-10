# pytest: skip_always
"""FunctionCallingAgent — stateful agent with agentic loop.

Wraps FunctionCallingPipeline with conversation history, tool execution,
and a loop that runs until the pipeline returns a conversational response
or the iteration budget is exhausted.

See function-calling-system-design.md for the full design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fc.pipeline import FunctionCallingPipeline
from mellea.backends.tools import MelleaTool


@dataclass
class AgentTurn:
    """Result of a single FunctionCallingAgent.chat() call.

    Attributes:
        message: Final OpenAI assistant message with content populated
            and tool_calls null. This is the response delivered to the user.
        tool_calls_made: All tool calls issued across pipeline iterations,
            in order. Each entry is an OpenAI tool call dict.
        tool_results: Corresponding tool results, parallel to tool_calls_made.
        iterations: Number of pipeline iterations used to produce this turn.
    """

    message: dict[str, Any]
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    iterations: int = 0


class FunctionCallingAgent:
    """Stateful agent that drives FunctionCallingPipeline in an agentic loop.

    Maintains conversation history across calls to chat(). Executes tool calls
    returned by the pipeline, appends results to history as OpenAI tool messages,
    and re-invokes the pipeline until a natural language response is produced.

    Args:
        pipeline: Constructed FunctionCallingPipeline instance.
        tools: MelleaTool instances with real callable implementations.
            These are executed when the pipeline returns tool calls.
        max_iterations: Maximum number of pipeline invocations per chat() call.
            Prevents infinite loops. Defaults to 10.
    """

    def __init__(
        self,
        pipeline: FunctionCallingPipeline,
        tools: list[MelleaTool],
        max_iterations: int = 10,
    ) -> None:
        self._pipeline = pipeline
        self._tools = {t.name: t for t in tools}
        self._max_iterations = max_iterations
        self._history: list[dict[str, Any]] = []

    def chat(self, user_message: str) -> AgentTurn:
        """Send a user message and return the agent's final response.

        Drives the pipeline in a loop: if the pipeline returns tool calls,
        executes them, appends results to history, and calls the pipeline
        again. Terminates when the pipeline returns a conversational response
        or max_iterations is reached.

        Args:
            user_message: Plain text message from the user.

        Returns:
            AgentTurn with the final assistant message and a record of all
            tool calls and results produced during this turn.
        """
        self._history.append({"role": "user", "content": user_message})

        tool_calls_made: list[dict[str, Any]] = []
        tool_results: list[Any] = []
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1
            assistant_msg = self._pipeline.run(
                messages=self._history, tools=list(self._tools.values())
            )

            if not assistant_msg.get("tool_calls"):
                # Terminal: pipeline returned a natural language response.
                self._history.append(assistant_msg)
                return AgentTurn(
                    message=assistant_msg,
                    tool_calls_made=tool_calls_made,
                    tool_results=tool_results,
                    iterations=iterations,
                )

            # Pipeline returned tool calls: execute and loop.
            self._history.append(assistant_msg)
            for tc in assistant_msg["tool_calls"]:
                result = self._execute_tool_call(tc)
                tool_calls_made.append(tc)
                tool_results.append(result)
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result)
                        if not isinstance(result, str)
                        else result,
                    }
                )

        # Budget exhausted: return what we have without a final response.
        fallback: dict[str, Any] = {
            "role": "assistant",
            "content": "[max_iterations reached without a final response]",
            "tool_calls": None,
        }
        self._history.append(fallback)
        return AgentTurn(
            message=fallback,
            tool_calls_made=tool_calls_made,
            tool_results=tool_results,
            iterations=iterations,
        )

    def reset(self) -> None:
        """Clear conversation history for a new session."""
        self._history = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> Any:
        """Execute a single OpenAI-format tool call dict.

        Args:
            tool_call: OpenAI tool call dict with "function.name" and
                "function.arguments" fields.

        Returns:
            Tool result, or an error dict if the tool is not found or
            argument parsing fails.
        """
        func_info = tool_call.get("function", {})
        name = func_info.get("name", "")
        raw_args = func_info.get("arguments", "{}")

        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name!r}"}

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {"error": f"could not parse arguments for {name!r}: {exc}"}

        try:
            return tool.run(**args)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"tool {name!r} raised: {exc}"}
