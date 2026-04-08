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
    question: str,
    tools: list[MelleaTool],
    context: ChatContext,
    backend: LocalHFBackend,
) -> list[MelleaTool]:
    """Filter a large tool catalog down to relevant tools.

    Returns:
        Filtered list of MelleaTools (subset of input tools).
    """
    _ensure_adapter(
        "fc_shortlister", ADAPTER_PATHS["fc_shortlister"], SHORTLISTER_CONFIG, backend
    )

    shortlist_ctx = context.add(Message("system", SHORTLISTER_SYSTEM_MESSAGE)).add(
        Message("user", question)
    )

    mot, _ = mfuncs.act(
        Intrinsic("fc_shortlister"),
        shortlist_ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.TOOLS: tools},
        strategy=None,
    )
    assert mot.is_computed()
    result_str = mot.value or ""

    try:
        result = json.loads(result_str)
        relevant_names = set(result.get("relevant_tools", []))
    except json.JSONDecodeError:
        return tools  # fallback: return all tools

    return [t for t in tools if t.name in relevant_names]


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
            'object containing a "tool_calls" array. No explanation, only JSON.',
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
# Large tool catalog (15 tools — shortlisting is valuable here)
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.

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


@tool
def calculate_mortgage(principal: float, rate: float, years: int) -> dict:
    """Calculate mortgage payments.

    Args:
        principal: Loan amount in dollars.
        rate: Annual interest rate as a decimal.
        years: Loan term in years.
    """
    monthly = principal * rate / 12 / (1 - (1 + rate / 12) ** (-years * 12))
    return {"monthly_payment": round(monthly, 2)}


@tool
def translate_text(text: str, target_lang: str) -> dict:
    """Translate text between languages.

    Args:
        text: Text to translate.
        target_lang: Target language code, e.g. 'es', 'fr'.
    """
    return {"translated": text, "target_lang": target_lang}


@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """
    return {"to": to, "subject": subject, "status": "sent"}


@tool
def create_calendar_event(title: str, date: str, time: str) -> dict:
    """Create a calendar event.

    Args:
        title: Event title.
        date: Event date (YYYY-MM-DD).
        time: Event time (HH:MM).
    """
    return {"title": title, "date": date, "time": time, "status": "created"}


@tool
def search_restaurants(city: str, cuisine: str = "") -> dict:
    """Search for restaurants.

    Args:
        city: City to search in.
        cuisine: Optional cuisine type, e.g. 'Italian'.
    """
    return {
        "city": city,
        "cuisine": cuisine,
        "results": ["Restaurant A", "Restaurant B"],
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price.

    Args:
        symbol: Stock ticker symbol, e.g. 'AAPL'.
    """
    return {"symbol": symbol, "price": 150.0}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert between currencies.

    Args:
        amount: Amount to convert.
        from_currency: Source currency code, e.g. 'USD'.
        to_currency: Target currency code, e.g. 'EUR'.
    """
    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
    }


@tool
def set_reminder(message: str, time: str) -> dict:
    """Set a reminder.

    Args:
        message: Reminder message text.
        time: Reminder time (HH:MM or ISO datetime).
    """
    return {"message": message, "time": time, "status": "set"}


@tool
def get_news(topic: str) -> dict:
    """Get latest news headlines.

    Args:
        topic: News topic or keyword.
    """
    return {"topic": topic, "headlines": ["Headline 1", "Headline 2"]}


@tool
def play_music(song: str, artist: str = "") -> dict:
    """Play music.

    Args:
        song: Song title.
        artist: Optional artist name.
    """
    return {"song": song, "artist": artist, "status": "playing"}


@tool
def order_food(restaurant: str, items: list) -> dict:
    """Order food delivery.

    Args:
        restaurant: Restaurant name.
        items: List of item names to order.
    """
    return {"restaurant": restaurant, "items": items, "status": "ordered"}


@tool
def get_directions(origin: str, destination: str) -> dict:
    """Get driving directions.

    Args:
        origin: Starting location.
        destination: Ending location.
    """
    return {"origin": origin, "destination": destination, "duration": "30 min"}


LARGE_TOOL_CATALOG: list[MelleaTool] = [
    get_weather,
    book_hotel,
    search_flights,
    calculate_mortgage,
    translate_text,
    send_email,
    create_calendar_event,
    search_restaurants,
    get_stock_price,
    convert_currency,
    set_reminder,
    get_news,
    play_music,
    order_food,
    get_directions,
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str, tools: list[MelleaTool]) -> None:
    context = ChatContext()
    print("Loading model...")
    backend = LocalHFBackend(model_id=BASE_MODEL)

    # Step 1 — Shortlist
    print(f"\n[1] Shortlisting from {len(tools)} tools...")
    filtered = shortlist_tools(question, tools, context, backend)
    filtered_names = [t.name for t in filtered]
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
