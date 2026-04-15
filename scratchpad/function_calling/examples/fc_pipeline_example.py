# pytest: skip_always
"""FunctionCallingPipeline — multi-turn example.

Demonstrates FCPipeline driven manually across multiple turns, simulating
the stateless benchmark pattern: each call receives the full message history
and the pipeline returns one OpenAI-format assistant message per call.

Tool calls are executed locally and appended to the history as OpenAI tool
messages before the next turn is sent. This mirrors exactly what a benchmark
like BFCL does externally.

Run
---
    uv run python scratchpad/function_calling/examples/fc_pipeline_example.py \\
        --model ibm-granite/granite-4.0-micro \\
        --adapters /path/to/fc-system
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Make the fc package and mellea importable when running outside uv/venv.
_repo_root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, _repo_root)

from fc.pipeline import FunctionCallingPipeline

from mellea.backends import tool
from mellea.backends.huggingface import LocalHFBackend

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
        "location": location,
        "condition": random.choice(["sunny", "partly cloudy", "overcast", "rainy"]),
        "temp_f": random.randint(45, 95),
        "humidity_pct": random.randint(30, 90),
    }


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search for available flights between two cities.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Travel date (YYYY-MM-DD).
    """
    return {
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


@tool
def book_hotel(city: str, checkin: str, checkout: str) -> dict:
    """Book a hotel room in a city.

    Args:
        city: City to book in.
        checkin: Check-in date (YYYY-MM-DD).
        checkout: Check-out date (YYYY-MM-DD).
    """
    return {
        "confirmation_id": "HTL-20240115-XK9",
        "city": city,
        "checkin": checkin,
        "checkout": checkout,
        "status": "confirmed",
        "hotel": "Mock Grand Hotel",
    }


TOOLS = [get_weather, search_flights, book_hotel]

# ---------------------------------------------------------------------------
# Conversation turns
# Each entry is a plain user message string. Tool execution is handled
# inline below, appended to history before the next turn is sent.
# ---------------------------------------------------------------------------

TURNS = [
    # Turn 1: parallel tool call — two independent weather lookups
    "What's the weather like in San Francisco and New York right now?",
    # Turn 2: conversational — no tool needed
    "Which of those two cities would you recommend for an outdoor event?",
    # Turn 3: tool call — flight search
    "Search for flights from San Francisco to New York on January 15.",
    # Turn 4: tool call — hotel booking
    "Book a hotel in New York from January 15 to January 16.",
    # Turn 5: conversational — wrap-up
    "Great, can you summarize what we've planned so far?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def execute_tool_calls(
    assistant_msg: dict[str, Any], tools_by_name: dict
) -> list[dict[str, Any]]:
    """Execute all tool calls in an assistant message and return tool messages.

    Args:
        assistant_msg: OpenAI-format assistant message with tool_calls populated.
        tools_by_name: Dict mapping tool name to MelleaTool.

    Returns:
        List of OpenAI tool role messages, one per executed call.
    """
    tool_messages = []
    for tc in assistant_msg.get("tool_calls") or []:
        name = tc["function"]["name"]
        raw_args = tc["function"].get("arguments", "{}")
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        tool_fn = tools_by_name.get(name)
        if tool_fn is None:
            result: Any = {"error": f"unknown tool: {name!r}"}
        else:
            result = tool_fn.run(**args)
        tool_messages.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)}
        )
    return tool_messages


def print_message(msg: dict[str, Any]) -> None:
    """Pretty-print a single OpenAI-format message."""
    role = msg.get("role", "?")
    if role == "tool":
        print(f"  [tool/{msg.get('tool_call_id', '')}]  {msg['content'][:120]}")
    elif msg.get("tool_calls"):
        calls = ", ".join(
            f"{tc['function']['name']}({tc['function']['arguments'][:60]})"
            for tc in msg["tool_calls"]
        )
        print(f"  [assistant/tool_calls]  {calls}")
    else:
        content = (msg.get("content") or "")[:120].replace("\n", " ")
        print(f"  [{role}]  {content}")


def print_history(history: list[dict[str, Any]]) -> None:
    print("\n  --- conversation history ---")
    for msg in history:
        print_message(msg)
    print("  ----------------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(pipeline: FunctionCallingPipeline) -> None:
    """Drive FCPipeline through the sample conversation."""
    tools_by_name = {t.name: t for t in TOOLS}
    history: list[dict[str, Any]] = []

    for turn_num, user_msg in enumerate(TURNS, start=1):
        print(f"\n{'=' * 60}")
        print(f"Turn {turn_num}: {user_msg}")
        print("=" * 60)

        history.append({"role": "user", "content": user_msg})

        assistant_msg = pipeline.run(messages=history, tools=TOOLS)
        history.append(assistant_msg)

        if assistant_msg.get("tool_calls"):
            tool_messages = execute_tool_calls(assistant_msg, tools_by_name)
            for tm in tool_messages:
                history.append(tm)

        print_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCPipeline multi-turn example.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        help="HuggingFace model ID (e.g. ibm-granite/granite-4.0-micro)",
    )
    model_group.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().parents[3] / "FCIntrinsics" / "granite-4.1-3b"),
        help="Local path to a model directory",
    )
    parser.add_argument(
        "--adapters",
        default=str(Path(__file__).resolve().parents[3] / "FCIntrinsics" / "fc-system"),
        help="Root directory containing adapter subdirs (router, parallel_tool_calling, "
        "multi_step_tool_calling, conversational_detection). Paths are auto-discovered "
        "by convention.",
    )
    parser.add_argument(
        "--adapter-override",
        action="append",
        metavar="NAME=PATH",
        default=[],
        help="Override a single adapter path. May be repeated. "
        "Example: --adapter-override fc_router=/experiments/my_router",
    )
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    for item in args.adapter_override:
        if "=" not in item:
            parser.error(f"--adapter-override must be NAME=PATH, got: {item!r}")
        name, path = item.split("=", 1)
        overrides[name] = path

    model_id = args.model or args.model_path
    print(f"Loading model from {'local path' if args.model_path else 'Hub'}: {model_id}")
    backend = LocalHFBackend(model_id=model_id)

    pipeline = FunctionCallingPipeline(
        backend=backend, adapters_dir=args.adapters, adapter_overrides=overrides or None
    )
    run(pipeline)
