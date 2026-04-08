# pytest: skip_always
"""Confidence-Gated Routing — fall back to monolithic when confidence is low.

Pipeline
--------
    query + tools  ──►  Router (classify + confidence score)
                           │
                           ├─ confidence >= threshold ──►  specialized adapter
                           └─ confidence <  threshold ──►  combined_baseline adapter

Flow equivalent
---------------
    configs/flows/confidence_routing.yaml
    route → execute_routed (if confident) OR execute_fallback (if not)

When to use
-----------
Use this when the router may misclassify edge-case queries.  The confidence
score lets you fall back to a monolithic LoRA (trained on all categories) when
the router is unsure.  This trades per-category specialization for robustness.

Note: The existing router LoRA was not trained to output a calibrated confidence
score, so values may be approximate.  Treat the threshold as a tunable parameter.

Run
---
    uv run python docs/examples/fc_patterns/103_confidence_gated_routing.py
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
CONFIDENCE_THRESHOLD = 0.7

CHECKPOINT_DIR = (
    "/proj/dmfexp/dgt/checkpoints/tuned/tc_capabilities/"
    "exp01_granite4_3b_rerun_tuned11/checkpoints"
)

ADAPTER_PATHS = {
    "fc_router": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_router",
    "fc_parallel": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_parallel_tool_calling",
    "fc_multi_step": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_multi_step_tool_calling",
    "fc_conversational": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_conversational_detection",
    "fc_baseline": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_combined_baseline",
}

# ---------------------------------------------------------------------------
# Configs — router includes confidence field
# ---------------------------------------------------------------------------

ROUTER_CONFIG_WITH_CONFIDENCE = {
    "model": "fc_router",
    "response_format": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["parallel", "multi_step", "conversational"],
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["category", "confidence", "reasoning"],
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
    "Respond with JSON including a confidence score (0.0 to 1.0): "
    '{"category": "<name>", "confidence": <0.0-1.0>, '
    '"reasoning": "<brief explanation>"}'
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


def route_query_with_confidence(
    question: str, tools: list[dict], context: ChatContext, backend: LocalHFBackend
) -> dict:
    """Classify a query and return a confidence score.

    Returns:
        Dict with ``category``, ``confidence`` (float), and ``reasoning``.
    """
    _ensure_adapter(
        "fc_router", ADAPTER_PATHS["fc_router"], ROUTER_CONFIG_WITH_CONFIDENCE, backend
    )

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
        result = json.loads(result_str)
        result.setdefault("confidence", 0.5)
        return result
    except json.JSONDecodeError:
        for cat in ("parallel", "multi_step", "conversational"):
            if cat in result_str.lower():
                return {"category": cat, "confidence": 0.5, "reasoning": result_str}
        return {
            "category": "conversational",
            "confidence": 0.0,
            "reasoning": result_str,
        }


def execute_with_adapter(
    question: str,
    tools: list[dict],
    adapter_name: str,
    adapter_path: str,
    context: ChatContext,
    backend: LocalHFBackend,
) -> str:
    """Execute with a specific named adapter."""
    tool_schemas = "\n".join(
        f"- {t['name']}: {t.get('description', '')}  "
        f"Parameters: {json.dumps(t.get('parameters', {}))}"
        for t in tools
    )

    tc_config: dict = EXECUTOR_CONFIG.copy()
    tc_config["response_format"] = TOOL_CALL_RESPONSE_FORMAT
    _ensure_adapter(adapter_name, adapter_path, tc_config, backend)

    exec_ctx = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Respond ONLY with JSON "
            'containing a "tool_calls" array. No explanation, only JSON.',
        )
    ).add(Message("user", f"Available tools:\n{tool_schemas}\n\nRequest: {question}"))

    mot, _ = mfuncs.act(
        Intrinsic(adapter_name),
        exec_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        strategy=None,
    )
    assert mot.is_computed()
    return mot.value or ""


# ---------------------------------------------------------------------------
# Example tools
# ---------------------------------------------------------------------------

EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather.",
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
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[dict]) -> None:
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Step 1 — Route with confidence
    print(f"\n[1] Routing: {question}")
    route_result = route_query_with_confidence(question, tools, context, backend)
    category = route_result.get("category", "conversational")
    confidence = route_result.get("confidence", 0.0)
    print(f"    Category:   {category}")
    print(f"    Confidence: {confidence:.2f}  (threshold: {CONFIDENCE_THRESHOLD})")

    # Step 2 — Decide: specialized or fallback
    if category == "conversational":
        from mellea.stdlib.session import MelleaSession

        print("\n[2] Conversational — using session.chat()...")
        result = MelleaSession(backend, context).chat(question).content
    elif confidence >= CONFIDENCE_THRESHOLD:
        adapter_map = {
            "parallel": ("fc_parallel", ADAPTER_PATHS["fc_parallel"]),
            "multi_step": ("fc_multi_step", ADAPTER_PATHS["fc_multi_step"]),
        }
        name, path = adapter_map[category]
        print(f"\n[2] High confidence — using specialized {category} adapter...")
        result = execute_with_adapter(question, tools, name, path, context, backend)
    else:
        print(f"\n[2] Low confidence ({confidence:.2f}) — falling back to baseline...")
        result = execute_with_adapter(
            question,
            tools,
            "fc_baseline",
            ADAPTER_PATHS["fc_baseline"],
            context,
            backend,
        )

    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Parallel (should be high confidence)")
    print("=" * 60)
    run("What's the weather in San Francisco and New York?", EXAMPLE_TOOLS)
