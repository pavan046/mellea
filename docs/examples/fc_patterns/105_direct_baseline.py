# pytest: skip_always
"""Direct Baseline — monolithic LoRA, no routing.

Pipeline
--------
    query + tools  ──►  combined_baseline adapter  ──►  result

Flow equivalent
---------------
    configs/flows/direct_baseline.yaml
    execute (single step, combined_baseline skill)

When to use
-----------
Use this as a comparison baseline.  The combined_baseline LoRA is trained on
a mix of all categories (parallel, multi_step, conversational), so it can
handle any query without routing.  It will generally underperform the modular
pipeline (101) on each individual category, but it is simpler and avoids
routing errors.  Run the same test queries as 101 to compare.

Run
---
    uv run python docs/examples/fc_patterns/105_direct_baseline.py
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
    "fc_baseline": f"{CHECKPOINT_DIR}/exp01_granite4_3b_rerun_tuned11_combined_baseline"
}

BASELINE_CONFIG = {
    "model": "fc_baseline",
    "response_format": {
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
    },
    "transformations": None,
    "instruction": None,
    "logprobs_workaround": None,
    "docs_as_message": None,
    "parameters": {"max_completion_tokens": 512},
    "sentence_boundaries": None,
}


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
# Example tools (same as 101 for comparison)
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
# Pipeline — no routing, single adapter
# ---------------------------------------------------------------------------


def run(question: str, tools: list[dict]) -> None:
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    _ensure_adapter(
        "fc_baseline", ADAPTER_PATHS["fc_baseline"], BASELINE_CONFIG, backend
    )

    tool_schemas = "\n".join(
        f"- {t['name']}: {t.get('description', '')}  "
        f"Parameters: {json.dumps(t.get('parameters', {}))}"
        for t in tools
    )

    exec_ctx = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Respond ONLY with JSON "
            'containing a "tool_calls" array. If no tools are needed, respond '
            'with {"tool_calls": []}. No explanation, only JSON.',
        )
    ).add(Message("user", f"Available tools:\n{tool_schemas}\n\nRequest: {question}"))

    print(f"\n[1] Direct execution (no routing): {question}")

    mot, _ = mfuncs.act(
        Intrinsic("fc_baseline"),
        exec_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        strategy=None,
    )
    assert mot.is_computed()
    print(f"\n>> Result:\n   {mot.value}")


if __name__ == "__main__":
    # Same test queries as 101 for direct comparison
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
