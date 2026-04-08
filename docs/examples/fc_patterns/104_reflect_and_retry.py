# pytest: skip_always
"""Reflect-and-Retry — the IVR analogy for function calling.

This is the key pattern.  It shows how Mellea's Instruct-Validate-Repair
concept applies to function-calling: generate tool calls, validate them,
reflect on errors using a dedicated LoRA adapter, and retry.

IVR Mapping
-----------
    IVR Step     RAG 104 (Hallucination Repair)     FC 104 (Reflect-and-Retry)
    ─────────    ──────────────────────────────      ──────────────────────────
    Instruct     session.instruct()                  mfuncs.act() on intrinsic
    Validate     flag_hallucinated_content()          validate_tool_calls() (schema)
    Repair       RepairTemplateStrategy + reasons     Reflector adapter + guidance
    Loop         Built-in loop_budget=3               Manual retry loop

Pipeline
--------
    query + tools  ──►  route  ──►  execute  ──►  validate
                                                     │
                                                     ├─ valid    ──►  return result
                                                     └─ invalid  ──►  reflect  ──►  retry
                                                                       (adapter)    (loop)

Flow equivalent
---------------
    configs/flows/simple_w_reflector.yaml
    route → [reflect if error] → execute

When to use
-----------
Use this when tool calls may have schema errors (wrong parameter types,
missing required fields, hallucinated tool names) and you want the model to
self-correct using a trained reflection adapter rather than a generic prompt.

Run
---
    uv run python docs/examples/fc_patterns/104_reflect_and_retry.py
    uv run python docs/examples/fc_patterns/104_reflect_and_retry.py --checkpoint-dir /path/to/fc-system
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
MAX_RETRIES = 3

DEFAULT_CHECKPOINT_DIR = (
    "/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics/fc-system"
)


def make_adapter_paths(checkpoint_dir: str) -> dict:
    """Build adapter path map from a checkpoint directory.

    The directory is expected to contain subdirectories named:
        router, parallel_tool_calling, multi_step_tool_calling,
        conversational_detection, reflector
    """
    return {
        "fc_router": f"{checkpoint_dir}/router",
        "fc_parallel": f"{checkpoint_dir}/parallel_tool_calling",
        "fc_multi_step": f"{checkpoint_dir}/multi_step_tool_calling",
        "fc_conversational": f"{checkpoint_dir}/conversational_detection",
        "fc_reflector": f"{checkpoint_dir}/reflector",
    }


# ---------------------------------------------------------------------------
# Configs
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
    "1. **parallel**: Independent tool calls in a single turn.\n"
    "2. **conversational**: Natural language response only.\n"
    "3. **multi_step**: Sequential tool calls with dependencies.\n\n"
    "Respond with JSON: "
    '{"category": "<name>", "reasoning": "<brief explanation>"}'
)

REFLECTOR_SYSTEM_MESSAGE = (
    "You are an error analysis assistant. Given a failed tool call and the "
    "error message, briefly analyze what went wrong and suggest a correction "
    "in 2-3 sentences."
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
# VALIDATE step — programmatic schema check
# ---------------------------------------------------------------------------


def validate_tool_calls(result_str: str, tools: list[MelleaTool]) -> tuple[bool, str]:
    """Validate that tool call output matches the available tool schemas.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        parsed = json.loads(result_str)
    except json.JSONDecodeError:
        return False, f"Output is not valid JSON: {result_str[:100]}"

    tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else parsed
    if not isinstance(tool_calls, list):
        return False, f"Expected a list of tool calls, got: {type(tool_calls)}"

    tool_names = {t.name for t in tools}
    tool_params = {
        t.name: set(
            t.as_json_tool.get("function", {}).get("parameters", {}).get("required", [])
        )
        for t in tools
    }

    errors = []
    for i, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            errors.append(f"Tool call {i} is not a dict")
            continue
        name = call.get("name")
        if name not in tool_names:
            errors.append(f"Tool call {i}: unknown tool '{name}'")
            continue
        args = call.get("arguments", {})
        missing = tool_params.get(name, set()) - set(args.keys())
        if missing:
            errors.append(f"Tool call {i} ({name}): missing required params: {missing}")

    if errors:
        return False, "; ".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# REFLECT step — uses reflector adapter via session.chat()
# (Reflector outputs plain text, not JSON — same approach as conversational)
# ---------------------------------------------------------------------------


def reflect_on_error(
    question: str,
    result_str: str,
    error_msg: str,
    context: ChatContext,
    backend: LocalHFBackend,
) -> str:
    """Use the reflector adapter to analyze a failed tool call.

    Returns:
        Plain text guidance (2-3 sentences).
    """
    from mellea.stdlib.session import MelleaSession

    reflect_ctx = context.add(Message("system", REFLECTOR_SYSTEM_MESSAGE)).add(
        Message(
            "user",
            f"The following tool call failed:\n{result_str}\n\n"
            f"Error: {error_msg}\n\n"
            f"Original request: {question}\n\n"
            "Briefly analyze the error and suggest a correction.",
        )
    )

    session = MelleaSession(backend, reflect_ctx)
    reply = session.chat("What went wrong and how should the tool call be corrected?")
    return reply.content


# ---------------------------------------------------------------------------
# Convenience functions (route + execute from 101)
# ---------------------------------------------------------------------------


def route_query(
    question: str,
    tools: list[MelleaTool],
    context: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> dict:
    """Classify a query as parallel, multi_step, or conversational."""
    _ensure_adapter("fc_router", adapter_paths["fc_router"], ROUTER_CONFIG, backend)

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
    adapter_paths: dict,
    guidance: str | None = None,
) -> str:
    """Execute with optional reflection guidance injected into the prompt."""
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

    tc_config: dict = EXECUTOR_CONFIG.copy()
    tc_config["response_format"] = TOOL_CALL_RESPONSE_FORMAT
    _ensure_adapter(name, path, tc_config, backend)

    system_msg = (
        "You are a function-calling assistant. Respond ONLY with JSON "
        'containing a "tool_calls" array. No explanation, only JSON.'
    )
    if guidance:
        system_msg += f"\n\nPrevious attempt failed. Guidance:\n{guidance}"

    exec_ctx = context.add(Message("system", system_msg)).add(Message("user", question))

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
# Main loop: Execute → Validate → Reflect → Retry
# ---------------------------------------------------------------------------


def run_with_retry(
    question: str,
    tools: list[MelleaTool],
    category: str,
    context: ChatContext,
    backend: LocalHFBackend,
    adapter_paths: dict,
) -> str:
    """Execute tool calls with reflection-based retry on validation failure.

    This is the IVR loop:
        Instruct (execute) → Validate (schema check) → Repair (reflect) → Retry
    """
    guidance = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n    --- Attempt {attempt}/{MAX_RETRIES} ---")

        # INSTRUCT: generate tool calls
        result = execute_tool_call(
            question,
            tools,
            category,
            context,
            backend,
            adapter_paths,
            guidance=guidance,
        )
        print(f"    Output: {result[:120]}...")

        # VALIDATE: check tool call schema
        is_valid, error_msg = validate_tool_calls(result, tools)

        if is_valid:
            print("    PASSED: tool calls are valid")
            return result

        print(f"    FAILED: {error_msg}")

        if attempt < MAX_RETRIES:
            # REPAIR: reflect on the error
            print("    Reflecting on error...")
            guidance = reflect_on_error(question, result, error_msg, context, backend)
            print(f"    Guidance: {guidance[:120]}...")

    print("    Budget exhausted — returning last attempt")
    return result


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

    # Step 1 — Route
    print(f"\n[1] Routing: {question}")
    route_result = route_query(question, tools, context, backend, adapter_paths)
    category = route_result.get("category", "conversational")
    print(f"    Category: {category}")

    # Step 2 — Execute with validate-reflect-retry loop
    print(f"\n[2] Executing with {category} adapter (with reflection retry)...")
    if category == "conversational":
        result = execute_tool_call(
            question, tools, category, context, backend, adapter_paths
        )
    else:
        result = run_with_retry(
            question, tools, category, context, backend, adapter_paths
        )

    print(f"\n>> Final Result:\n   {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing adapter subdirs (router, reflector, ...)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Test: Parallel with validation + reflection retry")
    print("=" * 60)
    run(
        "What's the weather in San Francisco and New York?",
        EXAMPLE_TOOLS,
        args.checkpoint_dir,
    )
