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
    uv run python docs/examples/fc_patterns/103_confidence_gated_routing.py --checkpoint-dir /path/to/fc-system
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
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "ibm-granite/granite-4.0-micro"
CONFIDENCE_THRESHOLD = 0.7

DEFAULT_CHECKPOINT_DIR = (
    "/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics/fc-system"
)


def make_adapter_paths(checkpoint_dir: str) -> dict:
    """Build adapter path map from a checkpoint directory.

    The directory is expected to contain subdirectories named:
        router, parallel_tool_calling, multi_step_tool_calling,
        conversational_detection, combined_baseline
    """
    return {
        "fc_router": f"{checkpoint_dir}/router",
        "fc_parallel": f"{checkpoint_dir}/parallel_tool_calling",
        "fc_multi_step": f"{checkpoint_dir}/multi_step_tool_calling",
        "fc_conversational": f"{checkpoint_dir}/conversational_detection",
        "fc_baseline": f"{checkpoint_dir}/combined_baseline",
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
    question: str,
    tools: list[MelleaTool],
    context: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> dict:
    """Classify a query and return a confidence score.

    Returns:
        Dict with ``category``, ``confidence`` (float), and ``reasoning``.
    """
    _ensure_adapter(
        "fc_router", adapter_paths["fc_router"], ROUTER_CONFIG_WITH_CONFIDENCE, backend
    )

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
    tools: list[MelleaTool],
    adapter_name: str,
    adapter_path: str,
    context: ChatContext,
    backend: LocalHFBackend,
) -> str:  # adapter_path is resolved by caller from adapter_paths dict
    """Execute with a specific named adapter."""
    tc_config: dict = EXECUTOR_CONFIG.copy()
    tc_config["response_format"] = TOOL_CALL_RESPONSE_FORMAT
    _ensure_adapter(adapter_name, adapter_path, tc_config, backend)

    exec_ctx = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Respond ONLY with JSON "
            'containing a "tool_calls" array. No explanation, only JSON.',
        )
    ).add(Message("user", question))

    mot, _ = mfuncs.act(
        Intrinsic(adapter_name),
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
    """Get current weather.

    Args:
        location: City name.
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
    """Search for flights.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Date (YYYY-MM-DD).
    """
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": ["FL100"],
    }


EXAMPLE_TOOLS: list[MelleaTool] = [get_weather, book_hotel, search_flights]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[MelleaTool], checkpoint_dir: str) -> None:
    adapter_paths = make_adapter_paths(checkpoint_dir)
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Step 1 — Route with confidence
    print(f"\n[1] Routing: {question}")
    route_result = route_query_with_confidence(
        question, tools, context, backend, adapter_paths
    )
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
            "parallel": ("fc_parallel", adapter_paths["fc_parallel"]),
            "multi_step": ("fc_multi_step", adapter_paths["fc_multi_step"]),
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
            adapter_paths["fc_baseline"],
            context,
            backend,
        )

    print(f"\n>> Result:\n   {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing adapter subdirs (router, combined_baseline, ...)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Test: Parallel (should be high confidence)")
    print("=" * 60)
    run(
        "What's the weather in San Francisco and New York?",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )
