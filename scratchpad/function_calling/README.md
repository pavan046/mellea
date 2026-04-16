# Function-Calling Pipeline

## BenchmarkServer

### Installation

Install the backend and server extras from the repo root:

```bash
uv sync --extra backends --extra server
```

### Starting the server

```bash
uv run python scratchpad/function_calling/fc/server.py \
    --model ibm-granite/granite-4.0-micro \
    --adapters scratchpad/lora/fc-system-4.0 \
    --port 8080
```

To override a single adapter without changing the rest:

```bash
uv run python scratchpad/function_calling/fc/server.py \
    --model ibm-granite/granite-4.0-micro \
    --adapters scratchpad/lora/fc-system-4.0 \
    --adapter-override fc_router=/experiments/my_router \
    --port 8080
```

Once the server is up, the interactive API docs are available at `http://localhost:8080/docs`.

---

### Sample requests

#### Opening turn (Turn 1) — POST /chat

`/chat` is the BFCL and ACEBench endpoint. The body contains only `messages` and `tools`.

```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What'\''s the weather like in San Francisco and New York right now?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name, e.g. '\''San Francisco'\''."}
            },
            "required": ["location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "search_flights",
          "description": "Search for available flights between two cities.",
          "parameters": {
            "type": "object",
            "properties": {
              "origin": {"type": "string", "description": "Departure city."},
              "destination": {"type": "string", "description": "Arrival city."},
              "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)."}
            },
            "required": ["origin", "destination", "date"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "book_hotel",
          "description": "Book a hotel room in a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City to book in."},
              "checkin": {"type": "string", "description": "Check-in date (YYYY-MM-DD)."},
              "checkout": {"type": "string", "description": "Check-out date (YYYY-MM-DD)."}
            },
            "required": ["city", "checkin", "checkout"]
          }
        }
      }
    ]
  }'
```

Expected response:

```json
{
  "type": "tool_calls",
  "tool_calls": [
    {"name": "get_weather", "arguments": {"location": "San Francisco"}},
    {"name": "get_weather", "arguments": {"location": "New York"}}
  ],
  "content": null
}
```

---

#### Turn 4 (multi-turn, hotel booking) — POST /v1/chat/completions

`/v1/chat/completions` is the TAU2 endpoint. It wraps the same `messages` and `tools` in a standard OpenAI ChatCompletion envelope and returns a full ChatCompletion response object.

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fc-agent",
    "messages": [
      {"role": "user", "content": "What'\''s the weather like in San Francisco and New York right now?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}},
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"New York\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_0", "content": "{\"location\": \"San Francisco\", \"condition\": \"sunny\", \"temp_f\": 72, \"humidity_pct\": 55}"},
      {"role": "tool", "tool_call_id": "call_1", "content": "{\"location\": \"New York\", \"condition\": \"overcast\", \"temp_f\": 58, \"humidity_pct\": 70}"},
      {"role": "assistant", "content": "San Francisco is sunny at 72°F. New York is overcast at 58°F.", "tool_calls": null},
      {"role": "user", "content": "Which of those two cities would you recommend for an outdoor event?"},
      {"role": "assistant", "content": "San Francisco would be the better choice given the sunny skies and warmer temperature.", "tool_calls": null},
      {"role": "user", "content": "Search for flights from San Francisco to New York on January 15."},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "search_flights", "arguments": "{\"origin\": \"San Francisco\", \"destination\": \"New York\", \"date\": \"2024-01-15\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_0", "content": "{\"origin\": \"San Francisco\", \"destination\": \"New York\", \"date\": \"2024-01-15\", \"flights\": [{\"flight_id\": \"FL100\", \"departure\": \"08:00\", \"arrival\": \"14:30\", \"price_usd\": 320}, {\"flight_id\": \"FL200\", \"departure\": \"13:45\", \"arrival\": \"20:15\", \"price_usd\": 275}]}"},
      {"role": "assistant", "content": "Found two flights: FL100 at 08:00 for $320 and FL200 at 13:45 for $275.", "tool_calls": null},
      {"role": "user", "content": "Book a hotel in New York from January 15 to January 16."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name, e.g. '\''San Francisco'\''."}
            },
            "required": ["location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "search_flights",
          "description": "Search for available flights between two cities.",
          "parameters": {
            "type": "object",
            "properties": {
              "origin": {"type": "string", "description": "Departure city."},
              "destination": {"type": "string", "description": "Arrival city."},
              "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)."}
            },
            "required": ["origin", "destination", "date"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "book_hotel",
          "description": "Book a hotel room in a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City to book in."},
              "checkin": {"type": "string", "description": "Check-in date (YYYY-MM-DD)."},
              "checkout": {"type": "string", "description": "Check-out date (YYYY-MM-DD)."}
            },
            "required": ["city", "checkin", "checkout"]
          }
        }
      }
    ]
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "fc-agent",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "book_hotel", "arguments": "{\"city\": \"New York\", \"checkin\": \"2024-01-15\", \"checkout\": \"2024-01-16\"}"}}
      ]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

---

## Open Questions and Known Limitations

### 1. GPU hardware is the first and most impactful lever

On a MacBook Pro (MPS), each LoRA forward pass takes roughly 2-3 seconds for a 3B
model. `FunctionCallingPipeline` makes 2 serial passes per request (router +
executor), which matches the observed ~13s/request in BFCL. On a server GPU (A100,
H100), a single forward pass for the same model takes ~100-200ms. That alone
brings per-request latency from ~13s to ~0.5-1s with no code changes whatsoever.

**ROI: High, unconditional.** Hardware is the only intervention with no
architectural conditions. Everything else discussed below has conditions that may
not hold.

### 2. Mellea's intrinsic system serializes all generation behind a single lock

`LocalHFBackend` acquires `_generation_lock` for every generation call. This is
correct: HF's PEFT adapter API allows only one adapter active at a time, so no two
requests can overlap regardless of how many threads or HTTP connections are open.
The lock is a Mellea constraint specific to `LocalHFBackend`, not a fundamental
multi-LoRA constraint.

vLLM's LoRA serving keeps multiple adapters resident in GPU memory simultaneously
and batches across them at the token level, eliminating this serialization entirely.
However, wiring vLLM into Mellea's intrinsic system is not a drop-in change: the
intrinsic catalog, adapter loading, and `_generate_from_intrinsic` are all built
around HF's PEFT API.

**ROI: Medium, conditional.** vLLM multi-LoRA improves throughput (requests
processed per second) but not per-request latency — the router → executor causal
chain is irreducible. The investment is justified only if you have many concurrent
benchmark clients saturating the server. A single BFCL run with sequential requests
sees no benefit beyond what GPU hardware already provides. Additionally, vLLM
requires all adapters served from one instance to share the same base model and the
same LoRA rank. If adapters were trained with different ranks (common when different
tasks need different capacity), they cannot be co-served.

### 3. Multi-node pipelines with more LoRAs face batch fragmentation

The current pipeline has 2 LoRA calls per request. Consider a 5-LoRA pipeline where
each node routes requests to different downstream nodes. The problem is batch
fragmentation: each routing node splits the incoming batch into subgroups by
category. With N requests and K routing categories, subgroup sizes shrink at each
hop. By node 3 or 4, the subgroups may be too small to amortize the overhead of a
batched forward pass, and batching provides diminishing returns.

The severity depends on topology:

- **Linear chain** (A→B→C→D→E): fully serial, no fan-out, batching helps throughput
  linearly but fragmentation is not a concern since all requests follow the same path.
- **Routing fan-out** (A routes to one of B/C/D, then E): batch shrinks at A. If the
  router sends 30%/50%/20% to three branches, each executor sees a smaller batch.
  With N=100 requests, the smallest branch has ~20 requests — still useful. With
  N=10, the smallest branch has ~2 requests — batching provides almost no benefit.
- **Parallel independent branches** (A→{B,C}→D): B and C have no data dependency
  and could run concurrently on separate GPU streams. This requires the scheduler to
  know about independence, which the current imperative `run()` does not express.

**ROI: Topology-dependent and hard to predict without profiling.** Batch
fragmentation is a well-known problem in mixture-of-experts routing. The only way to
know whether it matters for a specific pipeline is to measure subgroup size
distributions under realistic request loads.

### 4. LangGraph and LangChain do not solve the batching problem

LangGraph supports parallel execution of independent nodes within a single graph
invocation (superstep parallelism). If two nodes have no data dependency, they run
concurrently within one request. This is useful for fan-out patterns such as calling
three tools at once. It does not address request-level batching: each graph
invocation is still one request, and there is no built-in mechanism to batch N
requests through a shared model node. LangChain's `batch()` method batches
independent calls to a single stateless model, not requests through a multi-node
pipeline where nodes share model weights.

**Conclusion: LangGraph solves intra-request parallelism. The problem here is
inter-request batching across shared model state. These are different problems and
LangGraph does not address the latter.**

### 5. A topology-aware scheduler is the general solution but a significant investment

To support batching generically across any pipeline topology, the pipeline would
need to declare its computation as a DAG so a scheduler can derive synchronization
points automatically and group requests into per-node batches. This is the
architecture used by Ray Serve (deployment graph), NVIDIA Triton (ensemble
pipelines), and Apache Beam (dataflow). The current `FunctionCallingPipeline.run()`
is an imperative sequential function — changing it to a declared DAG is a non-trivial
redesign.

**ROI: Only justified once the pipeline topology stabilizes and throughput is a
first-class requirement at scale.** For the current proof-of-concept and benchmark
evaluation, GPU hardware (item 1) is the right investment. The scheduler question
should be revisited if and when FCPipeline is promoted to a production Mellea
component serving concurrent users.

---

## Legacy files and bug notes

### Files

| File | Description |
|------|-------------|
| `simple_pipeline.py` | Original FC pipeline using raw dicts for tool schemas |
| `simple_pipeline_mellea_conventions.py` | Same pipeline using `@tool` decorator and Mellea conventions |
| `multi_turn_fc_pipeline.py` | Multi-turn pipeline with router, executor, and conversational LoRAs; native tool execution via `parse_and_execute_tool_calls` |

### Bugs Found in Mellea

### 1. `find_func` only recurses into the first value of a dict

**File:** `mellea/backends/tools.py`

**Function:** `find_func`

**Problem:** The fallback recursion at the end of `find_func` uses `return` inside a `for` loop, so it only ever inspects the first value of a dict and returns immediately:

```python
for v in d.values():
    return find_func(v)  # returns on first iteration, never loops
```

This means that when the model returns a wrapper dict like `{"tool_calls": [...]}`, `find_func` recurses into the list value, hits `not isinstance(d, dict)`, and returns `(None, None)`. The inner tool call objects inside the list are never reached.

**Impact:** Any model output with a `{"tool_calls": [...]}` wrapper produces `mot.tool_calls = None`.

**Expected fix:** The fallback should iterate all values recursively and collect results, not short-circuit on the first one.

---

### 2. `to_tool_calls` uses a dict keyed by tool name, dropping duplicate calls

**File:** `mellea/backends/utils.py`

**Function:** `to_tool_calls`

**Problem:** Tool calls are accumulated into a `dict[str, ModelToolCall]` keyed by tool name:

```python
model_tool_calls: dict[str, ModelToolCall] = dict()
for tool_name, tool_args in parse_tools(decoded_result):
    model_tool_calls[tool_name] = ModelToolCall(...)  # overwrites previous entry
```

For parallel tool calls where the same function is called more than once (e.g., `get_weather` for two cities), only the last call survives.

**Impact:** Parallel calls to the same tool silently lose all but the last invocation. `mot.tool_calls` is structurally incapable of representing this case.

**Expected fix:** `mot.tool_calls` should be a `list[ModelToolCall]` rather than `dict[str, ModelToolCall]`. This is a broader API change that also affects `_call_tools`, `react`, and any downstream consumer of `mot.tool_calls`.

---

### 3. `_generate_from_intrinsic` hardcodes `tool_calls=False` and `tools={}`

**File:** `mellea/backends/huggingface.py`

**Function:** `_generate_from_intrinsic`

**Problem:** The `_post_process` callback is wired with hardcoded values:

```python
output._post_process = functools.partial(
    self.post_processing,
    ...
    tool_calls=False,
    tools={},
    ...
)
```

This means `to_tool_calls` is never called for intrinsic-based generation, even when `ModelOption.TOOLS` is passed and `tool_calls=True` is set by the caller. The tools dict is already available inside `_generate_from_intrinsic` via `model_options[ModelOption.TOOLS]` but is never forwarded to `post_processing`.

**Impact:** `mot.tool_calls` is always `None` for any intrinsic (LoRA adapter) generation, regardless of what the caller requests.

**Expected fix:** Derive `tools` and `tool_calls` from `model_options` inside `_generate_from_intrinsic` before wiring `_post_process`, the same way `_generate_from_context_standard` does it.

---

### 4. `message_to_openai_message` omits `tool_call_id` for tool messages

**File:** `mellea/helpers/openai_compatible_helpers.py`

**Function:** `message_to_openai_message`

**Problem:** For a `ToolMessage` (role `"tool"`), the function returns only `{"role": "tool", "content": "..."}`. The Granite formatter's `ToolResultMessage` Pydantic model requires `tool_call_id: str` as a non-optional field. When `ChatCompletion.model_validate` is called on the resulting dict in `_generate_from_intrinsic`, Pydantic raises a validation error.

**Impact:** Any multi-turn conversation that includes a `ToolMessage` in history will fail validation when passed to an intrinsic (LoRA) on the HF backend. This affects both hand-built pipelines and any future use of `_call_tools` followed by another intrinsic call.

**Fix applied:** `message_to_openai_message` now checks `isinstance(msg, ToolMessage)` first and includes a generated `tool_call_id`. The Granite formatter discards the ID during prompt rendering (see `granite3/input.py`, `_message_to_prompt_string`), so the value only needs to satisfy the Pydantic schema — a fresh UUID is sufficient.

---

### Workarounds

`multi_turn_fc_pipeline.py` implements `parse_and_execute_tool_calls`, a local helper that sidesteps bugs 1–3 by parsing `mot.value` directly:

- Handles both bare array `[{...}, {...}]` and wrapped `{"tool_calls": [{...}]}` formats
- Preserves duplicate tool calls (e.g., two `get_weather` calls)
- Runs `validate_tool_arguments` for type coercion
- Returns a `list[Message]` (one assistant message with tool call JSON in content, followed by one `ToolMessage` per executed call)

Bug 4 is fixed directly in `mellea/helpers/openai_compatible_helpers.py`.
