# pytest: skip_always
"""Route-and-Execute — the base modular function-calling pattern.

Pipeline
--------
    query + tools  ──►  Router (classify: parallel|multi_step|conversational)
                           │
                           ├─ parallel       ──►  parallel adapter  ──►  tool calls JSON
                           ├─ multi_step     ──►  multi-step adapter ──►  tool calls JSON
                           └─ conversational ──►  session.chat()    ──►  natural language

Flow equivalent
---------------
    configs/flows/simple_pipeline.yaml
    route → execute($steps.route.capability)

When to use
-----------
Use this as the building block for all function-calling pipelines.  The router
classifies the query into one of three categories, then the corresponding
specialized LoRA adapter handles execution.  Each adapter is trained on
category-specific data, so it outperforms a monolithic model on its category.

Run
---
    uv run python docs/examples/fc_patterns/101_route_and_execute.py
"""

from __future__ import annotations

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

CHECKPOINT_DIR = (
    "/proj/dmfexp/dgt/checkpoints/tuned/tc_capabilities/"
    "exp01_granite4_3b_rerun_tuned11/checkpoints"
)

ADAPTER_PATHS = {
    "fc_router": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_router",
    "fc_parallel": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_parallel_tool_calling",
    "fc_multi_step": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_multi_step_tool_calling",
    "fc_conversational": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_conversational_detection",
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

# ---------------------------------------------------------------------------
# Router system prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_MESSAGE = (
    "You are a routing assistant that classifies user requests into one of "
    "three categories based on the available tools:\n\n"
    "1. **parallel**: The request can be handled with one or more independent "
    "tool calls in a single turn.\n\n"
    "2. **conversational**: The request should be handled with natural language "
    "response only.\n\n"
    "3. **multi_step**: The request requires sequential tool calls where later "
    "calls depend on the results of earlier ones.\n\n"
    "Respond with a JSON object: "
    '{"category": "<name>", "reasoning": "<brief explanation>"}'
)

# ---------------------------------------------------------------------------
# LocalIntrinsicAdapter — loads LoRA from a local path instead of HF Hub
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
        return self._adapter_path

    def download_and_get_path(self, base_model_name: str) -> str:
        return self._adapter_path


# ---------------------------------------------------------------------------
# Helper: register adapter once
# ---------------------------------------------------------------------------


def _ensure_adapter(name: str, path: str, config: dict, backend: LocalHFBackend):
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
) -> dict:
    """Classify a query as parallel, multi_step, or conversational."""
    _ensure_adapter("fc_router", ADAPTER_PATHS["fc_router"], ROUTER_CONFIG, backend)

    router_ctx = context.add(Message("system", ROUTER_SYSTEM_MESSAGE)).add(
        Message("user", question)
    )

    mot, _ = mfuncs.act(
        Intrinsic("fc_router"),
        router_ctx,
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
) -> str:
    """Execute the appropriate adapter based on the routed category."""
    adapter_map = {
        "parallel": ("fc_parallel", ADAPTER_PATHS["fc_parallel"]),
        "multi_step": ("fc_multi_step", ADAPTER_PATHS["fc_multi_step"]),
        "conversational": ("fc_conversational", ADAPTER_PATHS["fc_conversational"]),
    }
    if category not in adapter_map:
        raise ValueError(f"Unknown category: {category}")

    name, path = adapter_map[category]

    if category == "conversational":
        from mellea.stdlib.session import MelleaSession

        return MelleaSession(backend, context).chat(question).content

    tc_config: dict = EXECUTOR_CONFIG.copy()
    tc_config["response_format"] = TOOL_CALL_RESPONSE_FORMAT
    _ensure_adapter(name, path, tc_config, backend)

    exec_ctx = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Respond ONLY with a JSON "
            'object containing a "tool_calls" array. Each element must have '
            '"name" and "arguments" fields. No explanation, only JSON.',
        )
    ).add(Message("user", question))

    mot, _ = mfuncs.act(
        Intrinsic(name),
        exec_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    return mot.value or ""


# ---------------------------------------------------------------------------
# Example tools — defined with Mellea's @tool decorator
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name, e.g. 'San Francisco'.
    """
    return {"location": location, "weather": "sunny", "temp_f": 72}


@tool
def book_hotel(city: str, check_in: str, check_out: str) -> dict:
    """Book a hotel room.

    Args:
        city: City to book in.
        check_in: Check-in date (YYYY-MM-DD).
        check_out: Check-out date (YYYY-MM-DD).
    """
    return {
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
        "status": "booked",
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
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": ["FL100", "FL200"],
    }


EXAMPLE_TOOLS: list[MelleaTool] = [get_weather, book_hotel, search_flights]

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[MelleaTool]) -> None:
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    print(f"\n[1] Routing: {question}")
    route_result = route_query(question, tools, context, backend)
    category = route_result.get("category", "conversational")
    print(f"    Category:  {category}")
    print(f"    Reasoning: {route_result.get('reasoning', '')}")

    print(f"\n[2] Executing with {category} adapter...")
    result = execute_tool_call(question, tools, category, context, backend)
    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Parallel")
    print("=" * 60)
    run("What's the weather in San Francisco and New York?", EXAMPLE_TOOLS)

    print("\n\n" + "=" * 60)
    print("Test 2: Multi-step")
    print("=" * 60)
    run(
        "Find flights from SF to NYC on Jan 15, then book a hotel for that night.",
        EXAMPLE_TOOLS,
    )

    print("\n\n" + "=" * 60)
    print("Test 3: Conversational")
    print("=" * 60)
    run("What is the capital of France?", EXAMPLE_TOOLS)
