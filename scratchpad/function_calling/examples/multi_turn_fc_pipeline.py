# pytest: skip_always
"""Function-Calling Multi-Turn Pipeline — Mellea conventions.

Demonstrates a multi-turn conversation loop where:
- A router LoRA classifies each user turn as parallel, multi_step, or conversational
- An executor LoRA produces tool calls for non-conversational turns
- A conversational LoRA produces natural language responses
- A single ChatContext grows across turns with clean OpenAI-style messages only:
    user, assistant (raw LM output), tool (one per executed tool call)

Run
---
    uv run python scratchpad/function_calling/multi_turn_fc_pipeline.py \\
        --model ibm-granite/granite-4.0-micro \\
        --adapters /path/to/fc-system
"""

from __future__ import annotations

import argparse
import json
import random

import mellea.stdlib.functional as mfuncs
from mellea.backends import ModelOption, tool
from mellea.backends.adapters.adapter import Adapter, IntrinsicAdapter
from mellea.backends.adapters.catalog import (
    _INTRINSICS_CATALOG,
    _INTRINSICS_CATALOG_ENTRIES,
    AdapterType,
    IntriniscsCatalogEntry,
)
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.tools import MelleaTool, validate_tool_arguments
from mellea.core.base import ModelOutputThunk, ModelToolCall
from mellea.stdlib.components import Message
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.components.intrinsic import Intrinsic
from mellea.stdlib.context import ChatContext

# ---------------------------------------------------------------------------
# Intrinsic configs (io.yaml equivalents)
# ---------------------------------------------------------------------------

ROUTER_CONFIG = {
    "model": "fc_router",
    "response_format": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["parallel", "multi_step", "conversational"],
            },
            "reasoning": {"type": "string"},
        },
        "required": ["category", "reasoning"],
    },
    "transformations": None,
    "instruction": None,
    "logprobs_workaround": None,
    "docs_as_message": None,
    "parameters": {"max_completion_tokens": 256},
    "sentence_boundaries": None,
}

EXECUTOR_CONFIG = {
    "model": "fc_executor",
    "response_format": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"],
        },
    },
    "transformations": None,
    "instruction": None,
    "logprobs_workaround": None,
    "docs_as_message": None,
    "parameters": {"max_completion_tokens": 512},
    "sentence_boundaries": None,
}

CONVERSATIONAL_CONFIG = {
    "model": "fc_conversational",
    "response_format": {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
    },
    "transformations": None,
    "instruction": None,
    "logprobs_workaround": None,
    "docs_as_message": None,
    "parameters": {"max_completion_tokens": 512},
    "sentence_boundaries": None,
}

ROUTER_SYSTEM_MESSAGE = (
    "You are a routing assistant that classifies user requests into one of "
    "three categories based on the available tools:\n\n"
    "1. **parallel**: The request can be handled with one or more independent "
    "tool calls in a single turn. Tool calls do not depend on each other's "
    "results. All required parameters are available or can be reasonably "
    "inferred.\n\n"
    "2. **conversational**: The request should be handled with natural language "
    "response only. This includes:\n"
    "   - No relevant tools are available for this request\n"
    "   - Tools exist but required parameters are missing or ambiguous\n"
    "   - The request is conversational and doesn't require tool execution\n\n"
    "3. **multi_step**: The request requires sequential tool calls where later "
    "calls depend on the results of earlier ones.\n\n"
    "Analyze the user's request and the available tools, then respond with a "
    'JSON object:\n{"category": "<category_name>", "reasoning": "<brief explanation>"}'
)

EXECUTOR_SYSTEM_MESSAGE = (
    "You are a function-calling assistant. Given the user's request "
    "and available tools, respond ONLY with a JSON array of tool calls. "
    "Each element must have 'name' and 'arguments' fields. "
    "Do NOT wrap the array in a 'tool_calls' key. "
    'Example: [{"name": "get_weather", "arguments": {"location": "SF"}}]'
)

CONVERSATIONAL_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Respond to the user's message in natural language. "
    'Wrap your reply in a JSON object with a single key: {"response": "<your reply>"}'
)


# ---------------------------------------------------------------------------
# LocalIntrinsicAdapter — loads LoRA weights from a local path
# ---------------------------------------------------------------------------


class LocalIntrinsicAdapter(IntrinsicAdapter):
    """IntrinsicAdapter subclass for locally stored LoRA weights."""

    def __init__(
        self,
        intrinsic_name: str,
        adapter_path: str,
        config_dict: dict,
        adapter_type: AdapterType = AdapterType.LORA,
    ):
        """Register the intrinsic in the catalog if not already present."""
        if intrinsic_name not in _INTRINSICS_CATALOG:
            entry = IntriniscsCatalogEntry(
                name=intrinsic_name, repo_id="local", adapter_types=(adapter_type,)
            )
            _INTRINSICS_CATALOG_ENTRIES.append(entry)
            _INTRINSICS_CATALOG[intrinsic_name] = entry

        Adapter.__init__(self, intrinsic_name, adapter_type)
        self.intrinsic_name = intrinsic_name
        self.intrinsic_metadata = _INTRINSICS_CATALOG[intrinsic_name]
        self.base_model_name = None
        self.config: dict = config_dict
        self._adapter_path = adapter_path

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local adapter path."""
        return self._adapter_path

    def download_and_get_path(self, base_model_name: str) -> str:
        """Return the local adapter path (no download needed)."""
        return self._adapter_path


def _ensure_adapter(
    name: str, path: str, config: dict, backend: LocalHFBackend
) -> None:
    """Register a LoRA adapter with the backend if not already registered."""
    qualified = f"{name}_{AdapterType.LORA.value}"
    if qualified not in backend._added_adapters:
        backend.add_adapter(LocalIntrinsicAdapter(name, path, config))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a location.

    Args:
        location: City name, e.g. 'San Francisco'.
    """
    return {
        "tool_executed": "get_weather",
        "location": location,
        "condition": random.choice(["sunny", "partly cloudy", "overcast", "rainy"]),
        "temp_f": random.randint(45, 95),
        "humidity_pct": random.randint(30, 90),
        "source": "mock-weather-api",
    }


@tool
def book_hotel(city: str, checkin: str, checkout: str) -> dict:
    """Book a hotel room in a city.

    Args:
        city: City to book in.
        checkin: Check-in date (YYYY-MM-DD).
        checkout: Check-out date (YYYY-MM-DD).
    """
    return {
        "tool_executed": "book_hotel",
        "confirmation_id": "HTL-20240115-XK9",
        "city": city,
        "checkin": checkin,
        "checkout": checkout,
        "status": "confirmed",
        "hotel": "Mock Grand Hotel",
    }


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search for flights between two cities.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Date (YYYY-MM-DD).
    """
    return {
        "tool_executed": "search_flights",
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": [
            {
                "flight_id": "FL100",
                "departure": "08:00",
                "arrival": "14:30",
                "price_usd": 320,
            },
            {
                "flight_id": "FL200",
                "departure": "13:45",
                "arrival": "20:15",
                "price_usd": 275,
            },
        ],
    }


TOOLS: list[MelleaTool] = [get_weather, book_hotel, search_flights]


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def route(
    user_msg: str,
    tools: list[MelleaTool],
    conversation_ctx: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> dict:
    """Classify a user message as parallel, multi_step, or conversational.

    Builds a throwaway context for the router — never stored back to
    conversation_ctx. The router sees the full conversation history plus
    its own system prompt and the new user question.
    """
    _ensure_adapter("fc_router", adapter_paths["fc_router"], ROUTER_CONFIG, backend)

    ctx = conversation_ctx.add(Message("system", ROUTER_SYSTEM_MESSAGE)).add(
        Message("user", user_msg)
    )
    mot, _ = mfuncs.act(
        Intrinsic("fc_router"),
        ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    result_str = mot.value or ""

    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        for cat in ("parallel", "multi_step", "conversational"):
            if cat in result_str.lower():
                return {"category": cat, "reasoning": result_str}
        return {"category": "conversational", "reasoning": result_str}


def execute_fc(
    user_msg: str,
    tools: list[MelleaTool],
    category: str,
    conversation_ctx: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> list[Message]:
    """Run the appropriate executor LoRA and return resulting messages.

    Builds a throwaway context for the executor — never stored back to
    conversation_ctx. The executor sees the full conversation history plus
    its own system prompt and the new user message.

    Returns:
        For conversational: [Message("assistant", natural_language_response)]
        For FC categories: [Message("assistant", tool_call_json), ToolMessage, ...]
    """
    adapter_map = {
        "parallel": ("fc_parallel", adapter_paths["fc_parallel"], EXECUTOR_CONFIG),
        "multi_step": (
            "fc_multi_step",
            adapter_paths["fc_multi_step"],
            EXECUTOR_CONFIG,
        ),
        "conversational": (
            "fc_conversational",
            adapter_paths["fc_conversational"],
            CONVERSATIONAL_CONFIG,
        ),
    }
    if category not in adapter_map:
        raise ValueError(f"Unknown category: {category!r}")

    adapter_name, adapter_path, config = adapter_map[category]
    _ensure_adapter(adapter_name, adapter_path, config, backend)

    if category == "conversational":
        ctx = conversation_ctx.add(
            Message("system", CONVERSATIONAL_SYSTEM_MESSAGE)
        ).add(Message("user", user_msg))
        mot, _ = mfuncs.act(
            Intrinsic(adapter_name),
            ctx,
            backend,
            model_options={ModelOption.TEMPERATURE: 0.0},
            strategy=None,
        )
        assert mot.is_computed()
        raw = mot.value or ""
        try:
            content = json.loads(raw).get("response", raw)
        except json.JSONDecodeError:
            content = raw
        return [Message(role="assistant", content=content)]

    ctx = conversation_ctx.add(Message("system", EXECUTOR_SYSTEM_MESSAGE)).add(
        Message("user", user_msg)
    )
    mot, _ = mfuncs.act(
        Intrinsic(adapter_name),
        ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    return parse_and_execute_tool_calls(mot, tools)


def parse_and_execute_tool_calls(
    mot: ModelOutputThunk, tools: list[MelleaTool]
) -> list[Message]:
    """Parse a JSON tool call array from mot.value and execute each call.

    Handles the limitations in mellea's native to_tool_calls:
    - Supports both bare array [...] and wrapped {"tool_calls": [...]} formats.
    - Preserves duplicate calls to the same tool (e.g. two get_weather calls).

    Note: mellea's Message class has no tool_calls field (unlike OpenAI's format).
    Tool call JSON is stored in the assistant message's content field.

    Args:
        mot: Computed ModelOutputThunk whose value is a JSON array of tool calls.
        tools: Available tools to dispatch against.

    Returns:
        List of Messages: one assistant Message (tool call JSON in content),
        followed by one ToolMessage per executed tool call.
        On parse failure, returns a single assistant Message describing the error.
    """
    raw = mot.value or ""
    if not raw:
        return [Message(role="assistant", content="[error: model produced no output]")]

    tools_by_name = {t.name: t for t in tools}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [
            Message(
                role="assistant",
                content=f"[error: tool call output is not valid JSON — {exc} | raw={raw!r}]",
            )
        ]

    if isinstance(parsed, dict):
        parsed = parsed.get("tool_calls", [])
    if not isinstance(parsed, list):
        return [
            Message(
                role="assistant",
                content=f"[error: expected a list of tool calls, got {type(parsed).__name__} | raw={raw!r}]",
            )
        ]

    # Store the raw tool call JSON in the assistant message content.
    # mellea's Message class has no tool_calls field, so content is the only place.
    assistant_message = Message(role="assistant", content=json.dumps(parsed))
    tool_messages: list[ToolMessage] = []
    for tc in parsed:
        name = tc.get("name")
        args = tc.get("arguments", {})
        func = tools_by_name.get(name)
        if func is None:
            continue
        validated_args = validate_tool_arguments(func, args, strict=False)
        mtc = ModelToolCall(name, func, validated_args)
        result = mtc.call_func()
        tool_messages.append(
            ToolMessage(
                role="tool",
                content=str(result),
                tool_output=result,
                name=name,
                args=validated_args,
                tool=mtc,
            )
        )
    return [assistant_message, *tool_messages]


# ---------------------------------------------------------------------------
# Multi-turn loop
# ---------------------------------------------------------------------------


def print_context(conversation_ctx: ChatContext) -> None:
    """Print the current conversation context in a readable format."""
    messages = conversation_ctx.as_list() or []
    print("\n  --- conversation context ---")
    for item in messages:
        if isinstance(item, ToolMessage):
            print(f"  [tool/{item.name}]  {item.content[:120]}")
        elif isinstance(item, Message):
            content_preview = item.content[:120].replace("\n", " ")
            print(f"  [{item.role}]  {content_preview}")
        else:
            print(f"  [?]  {item!r}")
    print("  ----------------------------\n")


def run_multiturn(
    turns: list[str],
    tools: list[MelleaTool],
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> None:
    """Run a multi-turn conversation, maintaining a single clean ChatContext.

    After each turn appends to conversation_ctx:
      - Message("user", user_msg)
      - Message("assistant", lm_output)
      - ToolMessage per executed tool call (FC turns only)
    """
    conversation_ctx: ChatContext = ChatContext()

    for turn_num, user_msg in enumerate(turns, start=1):
        print(f"\n{'=' * 60}")
        print(f"Turn {turn_num}: {user_msg}")
        print("=" * 60)

        # Route
        route_result = route(user_msg, tools, conversation_ctx, backend, adapter_paths)
        category = route_result.get("category", "conversational")
        reasoning = route_result.get("reasoning", "")
        print(f"  [router] category={category}  reasoning={reasoning[:80]}")

        # Execute
        messages = execute_fc(
            user_msg, tools, category, conversation_ctx, backend, adapter_paths
        )

        # Update conversation context with clean OpenAI-style messages only.
        conversation_ctx = conversation_ctx.add(Message("user", user_msg))
        for message in messages:
            conversation_ctx = conversation_ctx.add(message)

        print_context(conversation_ctx)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-turn FC pipeline using Mellea LoRA intrinsics."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID to load (e.g. ibm-granite/granite-4.0-micro)",
    )
    parser.add_argument(
        "--adapters",
        required=True,
        help="Directory containing adapter subdirs (router, parallel_tool_calling, ...)",
    )
    args = parser.parse_args()

    adapter_paths = {
        "fc_router": f"{args.adapters}/router",
        "fc_parallel": f"{args.adapters}/parallel_tool_calling",
        "fc_multi_step": f"{args.adapters}/multi_step_tool_calling",
        "fc_conversational": f"{args.adapters}/conversational_detection",
    }

    print("Loading model...")
    backend = LocalHFBackend(model_id=args.model)

    run_multiturn(
        turns=[
            # Turn 1: parallel tool call — two independent weather lookups
            "What's the weather like in San Francisco and New York right now?",
            # Turn 2: conversational but on-topic — no tool needed
            "Which of those two cities would you recommend for an outdoor event?",
            # Turn 3: tool call — flight search
            "Search for flights from San Francisco to New York on January 15.",
            # Turn 4: tool call — hotel booking following turn 3
            "Book a hotel in New York from January 15 to January 16.",
            # Turn 5: conversational — wrap-up
            "Great, can you give me a summary of what we've planned so far?",
        ],
        tools=TOOLS,
        backend=backend,
        adapter_paths=adapter_paths,
    )
