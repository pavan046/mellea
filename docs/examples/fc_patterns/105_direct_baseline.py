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
    uv run python docs/examples/fc_patterns/105_direct_baseline.py --checkpoint-dir /path/to/fc-system
"""

from __future__ import annotations

import argparse

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

DEFAULT_CHECKPOINT_DIR = (
    "/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics/fc-system"
)


def make_adapter_paths(checkpoint_dir: str) -> dict:
    """Build adapter path map from a checkpoint directory.

    The directory is expected to contain a subdirectory named: combined_baseline
    """
    return {"fc_baseline": f"{checkpoint_dir}/combined_baseline"}


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
# Example tools (same as 101 for comparison) — defined with @tool decorator
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
# Pipeline — no routing, single adapter
# ---------------------------------------------------------------------------


def run(question: str, tools: list[MelleaTool], checkpoint_dir: str) -> None:
    adapter_paths = make_adapter_paths(checkpoint_dir)
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    _ensure_adapter(
        "fc_baseline", adapter_paths["fc_baseline"], BASELINE_CONFIG, backend
    )

    exec_ctx = context.add(
        Message(
            "system",
            "You are a function-calling assistant. Respond ONLY with JSON "
            'containing a "tool_calls" array. If no tools are needed, respond '
            'with {"tool_calls": []}. No explanation, only JSON.',
        )
    ).add(Message("user", question))

    print(f"\n[1] Direct execution (no routing): {question}")

    mot, _ = mfuncs.act(
        Intrinsic("fc_baseline"),
        exec_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    print(f"\n>> Result:\n   {mot.value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing adapter subdirs (combined_baseline, ...)",
    )
    args = parser.parse_args()

    # Same test queries as 101 for direct comparison
    print("=" * 60)
    print("Test 1: Parallel")
    print("=" * 60)
    run(
        "What's the weather in San Francisco and New York?",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )

    print("\n\n" + "=" * 60)
    print("Test 2: Multi-step")
    print("=" * 60)
    run(
        "Find flights from SF to NYC on Jan 15, then book a hotel for that night.",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )

    print("\n\n" + "=" * 60)
    print("Test 3: Conversational")
    print("=" * 60)
    run("What is the capital of France?", EXAMPLE_TOOLS, args.checkpoint_dir)
