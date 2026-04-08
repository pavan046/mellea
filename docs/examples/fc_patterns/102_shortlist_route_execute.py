# pytest: skip_always
"""Shortlist-Route-Execute — tool catalog filtering before routing.

Pipeline
--------
    query + tools  ──►  Shortlister (filter relevant tools)
                           │
                           ▼  filtered tools
                        Router (classify)  ──►  Execute (with filtered tools)

Flow equivalent
---------------
    configs/flows/simple_w_shortlister.yaml
    shortlist → route → execute($steps.route.capability)

When to use
-----------
Use this when the tool catalog is large (10+ tools).  The shortlister LoRA is
trained to select only the relevant tools, reducing noise for the router and
executor.  This improves accuracy on catalogs with 20-50+ tools where the
router would otherwise be confused by irrelevant options.

Run
---
    uv run python docs/examples/fc_patterns/102_shortlist_route_execute.py
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
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "ibm-granite/granite-4.0-micro"

CHECKPOINT_DIR = (
    "/proj/dmfexp/dgt/checkpoints/tuned/tc_capabilities/"
    "exp01_granite4_3b_rerun_tuned11/checkpoints"
)

ADAPTER_PATHS = {
    "fc_router": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_router",
    "fc_shortlister": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_tool_shortlisting",
    "fc_parallel": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_parallel_tool_calling",
    "fc_multi_step": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_multi_step_tool_calling",
    "fc_conversational": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_conversational_detection",
}

# ---------------------------------------------------------------------------
# io.yaml-equivalent configs
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

SHORTLISTER_CONFIG = {
    "model": "fc_shortlister",
    "response_format": {
        "type": "object",
        "properties": {
            "relevant_tools": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["relevant_tools"],
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
    "1. **parallel**: Independent tool calls in a single turn.\n"
    "2. **conversational**: Natural language response only.\n"
    "3. **multi_step**: Sequential tool calls with dependencies.\n\n"
    "Respond with JSON: "
    '{"category": "<name>", "reasoning": "<brief explanation>"}'
)

SHORTLISTER_SYSTEM_MESSAGE = (
    "You are a tool selection assistant. Given the user's request and a "
    "catalog of available tools, identify which tools are relevant. "
    "Respond with JSON: "
    '{"relevant_tools": ["tool_name_1", "tool_name_2"]}'
)


# ---------------------------------------------------------------------------
# LocalIntrinsicAdapter
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


def _ensure_adapter(name: str, path: str, config: dict, backend: LocalHFBackend):
    """Register adapter once."""
    qualified = f"{name}_{AdapterType.LORA.value}"
    if qualified not in backend._added_adapters:
        backend.add_adapter(LocalIntrinsicAdapter(name, path, config))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def shortlist_tools(
    question: str, tools: list[dict], context: ChatContext, backend: LocalHFBackend
) -> list[dict]:
    """Filter a large tool catalog down to relevant tools.

    Returns:
        Filtered list of tool dicts (subset of input tools).
    """
    _ensure_adapter(
        "fc_shortlister", ADAPTER_PATHS["fc_shortlister"], SHORTLISTER_CONFIG, backend
    )

    tools_str = "\n".join(f"- {t['name']}: {t.get('description', '')}" for t in tools)
    shortlist_ctx = (
        context.add(Message("system", SHORTLISTER_SYSTEM_MESSAGE))
        .add(Message("user", f"Available tools:\n{tools_str}"))
        .add(Message("user", question))
    )

    mot, _ = mfuncs.act(
        Intrinsic("fc_shortlister"),
        shortlist_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        strategy=None,
    )
    assert mot.is_computed()
    result_str = mot.value or ""

    try:
        result = json.loads(result_str)
        relevant_names = set(result.get("relevant_tools", []))
    except json.JSONDecodeError:
        return tools  # fallback: return all tools

    return [t for t in tools if t["name"] in relevant_names]


def route_query(
    question: str, tools: list[dict], context: ChatContext, backend: LocalHFBackend
) -> dict:
    """Classify a query as parallel, multi_step, or conversational."""
    _ensure_adapter("fc_router", ADAPTER_PATHS["fc_router"], ROUTER_CONFIG, backend)

    tools_str = "\n".join(json.dumps(t) for t in tools)
    router_ctx = (
        context.add(Message("system", ROUTER_SYSTEM_MESSAGE))
        .add(Message("user", f"Available tools:\n{tools_str}"))
        .add(Message("user", question))
    )

    mot, _ = mfuncs.act(
        Intrinsic("fc_router"),
        router_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
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
    tools: list[dict],
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
    tool_schemas = "\n".join(
        f"- {t['name']}: {t.get('description', '')}  "
        f"Parameters: {json.dumps(t.get('parameters', {}))}"
        for t in tools
    )

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
            'object containing a "tool_calls" array. No explanation, only JSON.',
        )
    ).add(Message("user", f"Available tools:\n{tool_schemas}\n\nRequest: {question}"))

    mot, _ = mfuncs.act(
        Intrinsic(name),
        exec_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        strategy=None,
    )
    assert mot.is_computed()
    return mot.value or ""


# ---------------------------------------------------------------------------
# Large tool catalog (15 tools — shortlisting is valuable here)
# ---------------------------------------------------------------------------

LARGE_TOOL_CATALOG = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "book_hotel",
        "description": "Book a hotel room.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
    {
        "name": "search_flights",
        "description": "Search for flights.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "calculate_mortgage",
        "description": "Calculate mortgage payments.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "rate": {"type": "number"},
                "years": {"type": "integer"},
            },
            "required": ["principal", "rate", "years"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
            },
            "required": ["title", "date", "time"],
        },
    },
    {
        "name": "search_restaurants",
        "description": "Search for restaurants.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "cuisine": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert between currencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}, "time": {"type": "string"}},
            "required": ["message", "time"],
        },
    },
    {
        "name": "get_news",
        "description": "Get latest news headlines.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "play_music",
        "description": "Play music.",
        "parameters": {
            "type": "object",
            "properties": {"song": {"type": "string"}, "artist": {"type": "string"}},
            "required": ["song"],
        },
    },
    {
        "name": "order_food",
        "description": "Order food delivery.",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["restaurant", "items"],
        },
    },
    {
        "name": "get_directions",
        "description": "Get driving directions.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["origin", "destination"],
        },
    },
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[dict]) -> None:
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Step 1 — Shortlist
    print(f"\n[1] Shortlisting from {len(tools)} tools...")
    filtered = shortlist_tools(question, tools, context, backend)
    filtered_names = [t["name"] for t in filtered]
    print(f"    Kept {len(filtered)}/{len(tools)}: {filtered_names}")

    # Step 2 — Route (with filtered tools)
    print(f"\n[2] Routing: {question}")
    route_result = route_query(question, filtered, context, backend)
    category = route_result.get("category", "conversational")
    print(f"    Category: {category}")

    # Step 3 — Execute (with filtered tools)
    print(f"\n[3] Executing with {category} adapter...")
    result = execute_tool_call(question, filtered, category, context, backend)
    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    run(
        "What's the weather in San Francisco and book a hotel there for Jan 15-17?",
        LARGE_TOOL_CATALOG,
    )
