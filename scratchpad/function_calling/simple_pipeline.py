# pytest: skip_always
"""Function-Calling Simple Pipeline — Router → Execute.

Translates the simple_pipeline.yaml flow from /proj/dmfexp/tool_reasoning_code/
kapanipa/function-calling-agent into a Mellea program using full intrinsic
registration with local LoRA adapters.

Flow (from configs/flows/simple_pipeline.yaml)
----------------------------------------------
    user query + tools  ──►  Router (classify: parallel|multi_step|conversational)
                                │
                                ├─ parallel      ──►  parallel_tool_calling adapter
                                ├─ multi_step    ──►  multi_step_tool_calling adapter
                                └─ conversational ──►  conversational_detection adapter

Adapter source
--------------
    /proj/dmfexp/dgt/checkpoints/tuned/tc_capabilities/
    exp01_granite4_3b_rerun_tuned11/checkpoints/

Run
---
    uv run python scratchpad/simple_pipeline.py
"""

from __future__ import annotations

import json

import mellea.stdlib.functional as mfuncs
from mellea.backends import ModelOption
from mellea.backends.adapters.adapter import Adapter, IntrinsicAdapter
from mellea.backends.adapters.catalog import (
    _INTRINSICS_CATALOG,
    _INTRINSICS_CATALOG_ENTRIES,
    AdapterType,
    IntriniscsCatalogEntry,
)
from mellea.backends.huggingface import LocalHFBackend
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
# io.yaml-equivalent config dicts for each skill
#
# Required fields: model, response_format, transformations
# Optional fields: logprobs_workaround, docs_as_message, instruction,
#                  parameters, sentence_boundaries
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

# Parallel / multi-step / conversational share a minimal config: just parse
# the raw JSON output without any logprob-based transformations.
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

# ---------------------------------------------------------------------------
# Router system prompt (from function-calling-agent/src/skills/router/skill.py)
# ---------------------------------------------------------------------------

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
# LocalIntrinsicAdapter — loads LoRA from a local path instead of HF Hub
# ---------------------------------------------------------------------------


class LocalIntrinsicAdapter(IntrinsicAdapter):
    """IntrinsicAdapter subclass for locally stored LoRA weights.

    Bypasses HuggingFace Hub download by pointing directly at a filesystem
    path.  Registers the intrinsic in Mellea's catalog if not already present.

    Args:
        intrinsic_name: Name for the intrinsic (e.g. ``"fc_router"``).
        adapter_path: Local filesystem path to the adapter weights directory.
        config_dict: io.yaml-equivalent configuration dict.
        adapter_type: LoRA or aLoRA.  Defaults to LoRA.
    """

    def __init__(  # noqa: D107
        self,
        intrinsic_name: str,
        adapter_path: str,
        config_dict: dict,
        adapter_type: AdapterType = AdapterType.LORA,
    ):
        # Patch the global catalog so fetch_intrinsic_metadata() will find us.
        if intrinsic_name not in _INTRINSICS_CATALOG:
            entry = IntriniscsCatalogEntry(
                name=intrinsic_name, repo_id="local", adapter_types=(adapter_type,)
            )
            _INTRINSICS_CATALOG_ENTRIES.append(entry)
            _INTRINSICS_CATALOG[intrinsic_name] = entry

        # Initialize Adapter base (skip IntrinsicAdapter's HF download logic).
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


# ---------------------------------------------------------------------------
# Convenience functions for function-calling intrinsics
# ---------------------------------------------------------------------------


def route_query(
    question: str, tools: list[dict], context: ChatContext, backend: LocalHFBackend
) -> dict:
    """Classify a user query as parallel, multi_step, or conversational.

    Args:
        question: The user's query.
        tools: List of tool schemas (OpenAI function-calling format).
        context: Conversation history.
        backend: Backend with the fc_router adapter loaded.

    Returns:
        Dict with ``category`` and ``reasoning`` keys.
    """
    qualified = f"fc_router_{AdapterType.LORA.value}"
    if qualified not in backend._added_adapters:
        adapter = LocalIntrinsicAdapter(
            "fc_router", ADAPTER_PATHS["fc_router"], ROUTER_CONFIG
        )
        backend.add_adapter(adapter)

    # Build the context: system prompt + tools + user question
    tools_str = "\n".join(json.dumps(t) for t in tools)
    router_context = (
        context.add(Message("system", ROUTER_SYSTEM_MESSAGE))
        .add(Message("user", f"Available tools:\n{tools_str}"))
        .add(Message("user", question))
    )

    intrinsic = Intrinsic("fc_router")
    mot, _ = mfuncs.act(
        intrinsic,
        router_context,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 256},
        strategy=None,
    )
    assert mot.is_computed()

    result_str = mot.value
    if result_str is None:
        raise ValueError("Router returned None")

    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        # Fall back: try to extract category from plain text
        for cat in ("parallel", "multi_step", "conversational"):
            if cat in result_str.lower():
                return {"category": cat, "reasoning": result_str}
        return {"category": "conversational", "reasoning": result_str}


def execute_tool_call(
    question: str,
    tools: list[dict],
    category: str,
    context: ChatContext,
    backend: LocalHFBackend,
) -> str:
    """Execute the appropriate function-calling adapter based on the routed category.

    Args:
        question: The user's query.
        tools: List of tool schemas.
        category: One of ``"parallel"``, ``"multi_step"``, ``"conversational"``.
        context: Conversation history.
        backend: Backend with the execution adapters loaded.

    Returns:
        Raw model output (tool calls as JSON or conversational response).
    """
    adapter_map = {
        "parallel": ("fc_parallel", ADAPTER_PATHS["fc_parallel"]),
        "multi_step": ("fc_multi_step", ADAPTER_PATHS["fc_multi_step"]),
        "conversational": ("fc_conversational", ADAPTER_PATHS["fc_conversational"]),
    }

    if category not in adapter_map:
        raise ValueError(f"Unknown category: {category}")

    name, path = adapter_map[category]

    # Build context with tools and an explicit instruction to produce JSON tool calls.
    # The intrinsic pipeline doesn't support the chat template's `tools` parameter
    # (only `documents` for RAG), so we format tools inline with a clear directive.
    tool_schemas = "\n".join(
        f"- {t['name']}: {t.get('description', '')}  "
        f"Parameters: {json.dumps(t.get('parameters', {}))}"
        for t in tools
    )

    if category == "conversational":
        # Conversational adapter generates natural language, not JSON.
        # The intrinsic pipeline always runs json.loads() on the output,
        # which fails for plain text.  Use session.chat() instead — it
        # goes through the standard generation path (no JSON parsing).
        from mellea.stdlib.session import MelleaSession

        session = MelleaSession(backend, context)
        reply = session.chat(question)
        return reply.content

    # For parallel/multi_step: register the adapter once, then generate.
    qualified = f"{name}_{AdapterType.LORA.value}"
    if qualified not in backend._added_adapters:
        tc_config = EXECUTOR_CONFIG.copy()
        tc_config["response_format"] = {
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
        adapter = LocalIntrinsicAdapter(name, path, tc_config)
        backend.add_adapter(adapter)

    exec_context = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Given the user's request "
            "and available tools, respond ONLY with a JSON object containing "
            'a "tool_calls" array. Each element must have "name" and '
            '"arguments" fields. No explanation, only JSON.',
        )
    ).add(Message("user", f"Available tools:\n{tool_schemas}\n\nRequest: {question}"))

    intrinsic = Intrinsic(name)
    mot, _ = mfuncs.act(
        intrinsic,
        exec_context,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 512},
        strategy=None,
    )
    assert mot.is_computed()
    return mot.value or ""


# ---------------------------------------------------------------------------
# Example tools (OpenAI function-calling format)
# ---------------------------------------------------------------------------

EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "book_hotel",
        "description": "Book a hotel room in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City to book in"},
                "checkin": {
                    "type": "string",
                    "description": "Check-in date (YYYY-MM-DD)",
                },
                "checkout": {
                    "type": "string",
                    "description": "Check-out date (YYYY-MM-DD)",
                },
            },
            "required": ["city", "checkin", "checkout"],
        },
    },
    {
        "name": "search_flights",
        "description": "Search for flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[dict]) -> None:  # noqa: D103
    context = ChatContext()

    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Step 1 — Route
    print(f"\n[1] Routing query: {question}")
    route_result = route_query(question, tools, context, backend)
    category = route_result.get("category", "conversational")
    reasoning = route_result.get("reasoning", "")
    print(f"    Category:  {category}")
    print(f"    Reasoning: {reasoning}")

    # Step 2 — Execute
    print(f"\n[2] Executing with {category} adapter...")
    result = execute_tool_call(question, tools, category, context, backend)
    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Parallel — independent tool calls")
    print("=" * 60)
    run("What's the weather in San Francisco and New York?", EXAMPLE_TOOLS)

    print("\n\n")
    print("=" * 60)
    print("Test 2: Multi-step — sequential dependency")
    print("=" * 60)
    run(
        "Find flights from SF to NYC on Jan 15, then book a hotel in NYC for that night.",
        EXAMPLE_TOOLS,
    )

    print("\n\n")
    print("=" * 60)
    print("Test 3: Conversational — no tool needed")
    print("=" * 60)
    run("What is the capital of France?", EXAMPLE_TOOLS)
