# pytest: skip_always
"""Function-Calling Simple Pipeline — Mellea conventions version.

Same pipeline as simple_pipeline.py but uses Mellea's tool conventions:
- Tools defined with ``@tool`` decorator instead of raw dicts
- Tool schemas use the full OpenAI format (``{type: "function", function: {...}}``)
- Tool formatting uses ``MelleaTool.as_json_tool`` for consistency

Run
---
    uv run python scratchpad/function_calling/simple_pipeline_mellea_conventions.py
    uv run python scratchpad/function_calling/simple_pipeline_mellea_conventions.py --checkpoint-dir /path/to/fc-system
"""

from __future__ import annotations

import argparse
import json

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
from mellea.backends.tools import MelleaTool
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import Intrinsic
from mellea.stdlib.context import ChatContext

# ---------------------------------------------------------------------------
# Config: paths to local adapters
# ---------------------------------------------------------------------------

BASE_MODEL = "ibm-granite/granite-4.0-micro"

DEFAULT_CHECKPOINT_DIR = (
    "/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics/fc-system"
)


def make_adapter_paths(checkpoint_dir: str) -> dict:  # noqa: D103
    return {
        "fc_router": f"{checkpoint_dir}/router",
        "fc_parallel": f"{checkpoint_dir}/parallel_tool_calling",
        "fc_multi_step": f"{checkpoint_dir}/multi_step_tool_calling",
        "fc_conversational": f"{checkpoint_dir}/conversational_detection",
    }


# ---------------------------------------------------------------------------
# io.yaml-equivalent config dicts
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
    "response_format": None,
    "transformations": None,
    "instruction": None,
    "logprobs_workaround": None,
    "docs_as_message": None,
    "parameters": {"max_completion_tokens": 512},
    "sentence_boundaries": None,
}

TOOL_CALL_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        }
    },
    "required": ["tool_calls"],
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
    'JSON object:\n{"category": "<category_name>", "reasoning": '
    '"<brief explanation>"}'
)


# ---------------------------------------------------------------------------
# LocalIntrinsicAdapter
# ---------------------------------------------------------------------------


class LocalIntrinsicAdapter(IntrinsicAdapter):
    """IntrinsicAdapter subclass for locally stored LoRA weights."""

    def __init__(  # noqa: D107
        self,
        intrinsic_name: str,
        adapter_path: str,
        config_dict: dict,
        adapter_type: AdapterType = AdapterType.LORA,
    ):
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

    def get_local_hf_path(self, base_model_name: str) -> str:  # noqa: D102
        return self._adapter_path

    def download_and_get_path(self, base_model_name: str) -> str:  # noqa: D102
        return self._adapter_path


def _ensure_adapter(name: str, path: str, config: dict, backend: LocalHFBackend):
    """Register adapter once."""
    qualified = f"{name}_{AdapterType.LORA.value}"
    if qualified not in backend._added_adapters:
        backend.add_adapter(LocalIntrinsicAdapter(name, path, config))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def route_query(
    question: str,
    tools: list[MelleaTool],
    context: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> dict:
    """Classify a user query as parallel, multi_step, or conversational."""
    _ensure_adapter("fc_router", adapter_paths["fc_router"], ROUTER_CONFIG, backend)

    # Tools are passed via ModelOption.TOOLS so they go through the chat
    # template's native tool formatting (apply_chat_template(tools=...))
    # rather than being inlined as text in the user message.
    router_context = context.add(Message("system", ROUTER_SYSTEM_MESSAGE)).add(
        Message("user", question)
    )

    mot, _ = mfuncs.act(
        Intrinsic("fc_router"),
        router_context,
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


def execute_tool_call(
    question: str,
    tools: list[MelleaTool],
    category: str,
    context: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> str:
    """Execute the appropriate adapter based on the routed category."""
    adapter_map = {
        "parallel": ("fc_parallel", adapter_paths["fc_parallel"]),
        "multi_step": ("fc_multi_step", adapter_paths["fc_multi_step"]),
        "conversational": ("fc_conversational", adapter_paths["fc_conversational"]),
    }
    if category not in adapter_map:
        raise ValueError(f"Unknown category: {category}")

    name, path = adapter_map[category]

    if category == "conversational":
        from mellea.stdlib.session import MelleaSession

        return MelleaSession(backend, context).chat(question).content

    tc_config = EXECUTOR_CONFIG.copy()
    tc_config["response_format"] = TOOL_CALL_RESPONSE_FORMAT
    _ensure_adapter(name, path, tc_config, backend)

    # Tools passed via ModelOption.TOOLS — the chat template handles formatting.
    exec_context = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Given the user's request "
            "and available tools, respond ONLY with a JSON object containing "
            'a "tool_calls" array. Each element must have "name" and '
            '"arguments" fields. No explanation, only JSON.',
        )
    ).add(Message("user", question))

    mot, _ = mfuncs.act(
        Intrinsic(name),
        exec_context,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    return mot.value or ""


# ---------------------------------------------------------------------------
# Tools defined using Mellea's @tool decorator
#
# Each decorated function IS a MelleaTool instance with:
#   .as_json_tool  → full OpenAI format {"type": "function", "function": {...}}
#   .call_func()   → actually execute the tool
#   .name          → tool name
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a location.

    Args:
        location: City name, e.g. 'San Francisco'.
    """
    return {"location": location, "weather": "sunny", "temp_f": 72}


@tool
def book_hotel(city: str, checkin: str, checkout: str) -> dict:
    """Book a hotel room in a city.

    Args:
        city: City to book in.
        checkin: Check-in date (YYYY-MM-DD).
        checkout: Check-out date (YYYY-MM-DD).
    """
    return {"city": city, "checkin": checkin, "checkout": checkout, "status": "booked"}


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search for flights between two cities.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Date (YYYY-MM-DD).
    """
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": ["FL100", "FL200"],
    }


EXAMPLE_TOOLS: list[MelleaTool] = [get_weather, book_hotel, search_flights]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[MelleaTool], checkpoint_dir: str) -> None:  # noqa: D103
    adapter_paths = make_adapter_paths(checkpoint_dir)
    context = ChatContext()

    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Show the tool schemas being used
    print("\nTool schemas (OpenAI format):")
    for t in tools:
        print(f"  {json.dumps(t.as_json_tool, indent=2)[:100]}...")

    # Step 1 — Route
    print(f"\n[1] Routing query: {question}")
    route_result = route_query(question, tools, context, backend, adapter_paths)
    category = route_result.get("category", "conversational")
    reasoning = route_result.get("reasoning", "")
    print(f"    Category:  {category}")
    print(f"    Reasoning: {reasoning}")

    # Step 2 — Execute
    print(f"\n[2] Executing with {category} adapter...")
    result = execute_tool_call(
        question, tools, category, context, backend, adapter_paths
    )
    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing adapter subdirs (router, parallel_tool_calling, ...)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Test 1: Parallel — independent tool calls")
    print("=" * 60)
    run(
        "What's the weather in San Francisco and New York?",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )

    print("\n\n")
    print("=" * 60)
    print("Test 2: Multi-step — sequential dependency")
    print("=" * 60)
    run(
        "Find flights from SF to NYC on Jan 15, then book a hotel in NYC for that night.",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )

    print("\n\n")
    print("=" * 60)
    print("Test 3: Conversational — no tool needed")
    print("=" * 60)
    run("What is the capital of France?", EXAMPLE_TOOLS, args.checkpoint_dir)
