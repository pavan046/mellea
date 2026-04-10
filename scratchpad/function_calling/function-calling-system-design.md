# Function-Calling System Design

## Table of Contents

1. [Goal](#1-goal)
2. [Concepts and Naming Conventions](#2-concepts-and-naming-conventions)
3. [Architecture Overview](#3-architecture-overview)
4. [Wire Format: OpenAI as the Lingua Franca](#4-wire-format-openai-as-the-lingua-franca)
5. [Tool Representation](#5-tool-representation)
   - [FCPipeline: schema-only tools accepted](#51-fcpipeline-schema-only-tools-accepted)
   - [FCAgent: real callables required](#52-fcagent-real-callables-required)
   - [BenchmarkServer: owns all translation](#53-benchmarkserver-owns-all-translation)
6. [Adapter Configuration: io.yaml Convention](#6-adapter-configuration-ioyaml-convention)
   - [Co-location rule](#61-co-location-rule)
   - [system_prompt extension](#62-system_prompt-extension)
   - [How system_prompt flows through the code](#63-how-system_prompt-flows-through-the-code)
7. [Node and Flow Architecture](#7-node-and-flow-architecture)
   - [Why nodes](#71-why-nodes)
   - [Node structure](#72-node-structure)
   - [Pipeline as a thin flow](#73-pipeline-as-a-thin-flow)
   - [Multiple pipelines reusing the same nodes](#74-multiple-pipelines-reusing-the-same-nodes)
   - [What we are not building](#75-what-we-are-not-building)
8. [FunctionCallingPipeline (FCPipeline)](#8-functioncallingpipeline-fcpipeline)
   - [Responsibilities](#81-responsibilities)
   - [Interface](#82-interface)
   - [Internal Flow](#83-internal-flow)
9. [FunctionCallingAgent (FCAgent)](#9-functioncallingagent-fcagent)
   - [Responsibilities](#91-responsibilities)
   - [Interface](#92-interface)
   - [Internal Loop](#93-internal-loop)
10. [How Benchmarks Work](#10-how-benchmarks-work)
    - [The Stateless Contract](#101-the-stateless-contract)
    - [Benchmark Integration Summary](#102-benchmark-integration-summary)
    - [BFCL](#103-bfcl)
    - [TAU2 Bench](#104-tau2-bench)
    - [ACEBench](#105-acebench)
11. [BenchmarkServer](#11-benchmarkserver)
    - [Why It Exists](#111-why-it-exists)
    - [Endpoints](#112-endpoints)
    - [Protocol Translation](#113-protocol-translation)
    - [What the Server Must Not Do](#114-what-the-server-must-not-do)
12. [How a Live System Works](#12-how-a-live-system-works)
13. [Build Order](#13-build-order)
14. [Open Questions](#14-open-questions)

---

## 1. Goal

The immediate goal is to evaluate a Mellea-based function-calling system against standard benchmarks: BFCL, TAU2 Bench, and ACEBench. These benchmarks measure the quality of tool selection, argument generation, multi-turn reasoning, and conversational handling.

The longer-term goal is to ship a usable `FunctionCallingAgent` that developers can embed in real applications without needing to know anything about LoRA adapters, routing logic, or Mellea internals.

These two goals create different interface requirements, which is why the system is designed in two layers rather than one.

---

## 2. Concepts and Naming Conventions

Naming in this system is deliberate. Before any code is written, the terminology must be clear so that contributors can reason about the system without ambiguity.

### Pipeline

A **pipeline** is stateless and single-pass. It takes a fully specified input, runs it through one or more processing steps, and returns a result. It does not execute tools, does not hold conversation state between calls, and does not loop. Given the same input, it produces a deterministic output.

`FunctionCallingPipeline` (FCPipeline for short) is the core processing unit. It accepts a message history and a tool list in OpenAI format, routes the request to the appropriate LoRA adapter, and returns an OpenAI-format response containing either tool calls or a natural language message. It does not decide what to do with tool results. That is the caller's responsibility.

### Agent

An **agent** is stateful and cyclical. It has a goal, it can act to pursue that goal, and it observes the results of its actions before deciding what to do next. The cycle is: decide → act → observe → decide again. An agent holds conversation state and owns a tool registry with real executable functions. It internalizes the full agentic loop.

`FunctionCallingAgent` (FCAgent for short) is the stateful wrapper around FCPipeline. It owns the conversation history, registers and executes tools, and drives the pipeline in a loop until a terminal condition is reached (a final natural language response, or a loop budget is exhausted).

### Why This Distinction Matters

Calling a stateless dispatcher an "agent" is a misnomer. An agent implies autonomy and the ability to close the observe cycle. FCPipeline does not observe tool results. It cannot be autonomous. Keeping these names honest makes the system easier to reason about, easier to test, and easier to extend.

| Concept | Stateful | Executes tools | Has loop | Who uses it |
|---|---|---|---|---|
| `FunctionCallingPipeline` | No | No | No | BenchmarkServer, tests |
| `FunctionCallingAgent` | Yes | Yes | Yes | Application developers |

### BenchmarkServer

The **BenchmarkServer** is a FastAPI application that acts as a protocol adapter between benchmark CLIs and FCPipeline. It owns all translation between benchmark-specific wire formats and the OpenAI format that FCPipeline speaks. It holds no business logic of its own.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Application Developer                                      │
│                                                             │
│  FunctionCallingAgent  (FCAgent)                            │
│    - owns conversation history (OpenAI message dicts)       │
│    - owns tool registry (list[MelleaTool] with callables)   │
│    - drives agentic loop                                    │
│    - calls FCPipeline.run() each iteration                  │
└──────────────────────┬──────────────────────────────────────┘
                       │  OpenAI messages + MelleaTool list
┌──────────────────────▼──────────────────────────────────────┐
│  FunctionCallingPipeline  (FCPipeline)                      │
│    - stateless, single-pass                                 │
│    - accepts OpenAI messages + list[MelleaTool|dict]        │
│    - routes to appropriate LoRA                             │
│    - returns OpenAI-format assistant message                │
└──────────────────────┬──────────────────────────────────────┘
                       │  Mellea internals only below this line
┌──────────────────────▼──────────────────────────────────────┐
│  Mellea Backend  (LocalHFBackend)                           │
│    - loads base model once at startup                       │
│    - hot-swaps LoRA adapters per request                    │
│    - fc_router, fc_parallel, fc_multi_step,                 │
│      fc_conversational, fc_self_correct                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  BenchmarkServer  (FastAPI, thin protocol adapter)          │
│                                                             │
│    POST /chat               ← BFCL, ACEBench                │
│    POST /v1/chat/completions ← TAU2 (OpenAI-compatible)     │
│    GET  /health             ← startup health poll           │
│                                                             │
│    Owns ALL translation between benchmark wire formats      │
│    and the OpenAI format FCPipeline expects.                │
│    Never holds conversation state.                          │
└─────────────────────────────────────────────────────────────┘
```

BenchmarkServer sits beside FCAgent in the diagram, not above it. Both are callers of FCPipeline. Benchmarks manage their own state externally; they do not need FCAgent.

---

## 4. Wire Format: OpenAI as the Lingua Franca

FCPipeline and FCAgent speak OpenAI format exclusively on their external interfaces. This is a deliberate constraint, not a convenience shortcut.

OpenAI's message and tool schema formats are the de facto standard in the function-calling ecosystem. Every benchmark either already uses them (TAU2) or can be translated to/from them with a thin adapter (BFCL, ACEBench). Using OpenAI format means:

- FCPipeline and FCAgent have no knowledge of benchmark-specific schemas.
- Adding support for a new benchmark requires only a new protocol adapter in BenchmarkServer.
- The pipeline and agent can be tested independently of any benchmark.
- Application developers using FCAgent get an interface that is familiar and well-documented.

**Messages** follow OpenAI's chat format:

```json
[
  {"role": "user", "content": "What is the weather in San Francisco?"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "{\"temp_f\": 68, \"condition\": \"sunny\"}"
  }
]
```

**Responses** from FCPipeline are OpenAI assistant messages:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}
    }
  ]
}
```

or, for a conversational response:

```json
{
  "role": "assistant",
  "content": "The weather in San Francisco is currently 68°F and sunny.",
  "tool_calls": null
}
```

The presence or absence of `tool_calls` is sufficient to distinguish the two cases. No custom `type` field is needed.

---

## 5. Tool Representation

Tools flow through the system in two different forms depending on the layer.

### 5.1 FCPipeline: schema-only tools accepted

FCPipeline accepts tools as `list[MelleaTool] | list[dict]`. When dicts are provided, they must be in OpenAI's tool schema format:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string", "description": "City name"}
      },
      "required": ["location"]
    }
  }
}
```

FCPipeline converts dicts to `MelleaTool` internally using a no-op callable:

```python
MelleaTool(
    name=schema["function"]["name"],
    tool_call=lambda **_: None,   # never invoked; pipeline does not execute tools
    as_json_tool=schema,
)
```

This is safe because `MelleaTool._call_tool` is only invoked if someone explicitly calls `tool.run()` or `ModelToolCall.call_func()`. FCPipeline never does this: it only passes the tool schema to the LoRA for prompt construction and parses the model's output. The callable is not touched during inference.

No `SchemaMelleaTool` subclass is needed. The distinction between schema-only and executable tools is a usage contract, not a type-level difference.

### 5.2 FCAgent: real callables required

FCAgent accepts only `list[MelleaTool]` with real callable implementations. It is the only layer that calls `tool.run()`, so the callable must produce a real result. Passing schema-only tools to FCAgent would cause silent no-op execution and incorrect conversation history.

### 5.3 BenchmarkServer: owns all translation

Benchmarks send tools in their own formats. BenchmarkServer is responsible for translating them into OpenAI tool schema dicts before calling FCPipeline. FCPipeline never sees a benchmark-specific tool representation.

```
Benchmark sends tools  →  BenchmarkServer translates to OpenAI dicts
                       →  FCPipeline converts to MelleaTool(no-op callable)
                       →  LoRA sees schema only
                       →  FCPipeline returns OpenAI assistant message
                       →  BenchmarkServer translates response to benchmark format
```

---

## 6. Adapter Configuration: io.yaml Convention

### 6.1 Co-location rule

Every LoRA adapter used as a Mellea intrinsic must include an `io.yaml` file placed directly in its checkpoint directory, alongside `adapter_config.json` and the weight files:

```
my_adapter/
    io.yaml                        ← required
    adapter_config.json
    adapter_model.safetensors
```

Mellea's `IntrinsicAdapter` accepts a `config_file=` path at load time. The fc-system uses `LocalIntrinsicAdapter` (in `fc/lora.py`) to load adapters from local directories instead of the HF Hub. `LocalIntrinsicAdapter` reads `io.yaml` from the adapter path, validates it, and registers the adapter in Mellea's internal catalog.

The three required fields in every `io.yaml` are `model`, `response_format`, and `transformations`. Optional fields are documented in `scratchpad/function_calling/how_to_write_io_yaml.md`.

### 6.2 `system_prompt` extension

`system_prompt` is a custom field we added to the `io.yaml` schema. It is not part of Mellea's native `io.yaml` specification. It holds the verbatim system message that the LoRA adapter was trained with, co-located with the adapter weights so that application code does not need to manage it.

Mellea's internal validator (`make_config_dict`) raises `ValueError` on unknown fields. To prevent this, `LocalIntrinsicAdapter.__init__` pops `system_prompt` from the raw config dict **before** assigning `self.config`. The field is stored separately as `self.system_prompt`.

```yaml
# example: router/io.yaml
system_prompt: |
  You are a request classifier. Given a user message and the available
  tools, classify the request as one of: parallel, multi_step, conversational.
  Output a JSON object with fields "category" and "reasoning".
```

For adapters that are not yet integrated (or have no known role), use the generic fallback:

```yaml
system_prompt: |
  You are a helpful assistant. Always respond in JSON.
  Wrap your reply in a JSON object with a single key:
  {"response": "<your reply>"}
```

**Do not use `instruction` for this purpose.** Mellea's native `instruction` field appends content as a `user` role message at the end of the conversation, not as a `system` message. Use `system_prompt` when the LoRA was trained with a system message; use `instruction` only if the LoRA was trained with the guiding prompt as the final user message.

### 6.3 How `system_prompt` flows through the code

```
LocalIntrinsicAdapter.__init__(adapter_path)
  │
  ▼
reads io.yaml → raw_config dict
  │
  ▼
self.system_prompt = raw_config.pop("system_prompt", None)
  │  (removed before Mellea sees the dict)
  ▼
self.config = raw_config   ← Mellea validator sees clean dict, no ValueError
```

At inference time, `Node._build_ctx()` (inherited by all node classes) retrieves the adapter object from the backend's `_added_adapters` registry, reads `adapter.system_prompt`, and prepends a `system` role `Message` to the `ChatContext` if the prompt is non-null:

```python
def _build_ctx(self, ctx: ChatContext, backend: LocalHFBackend, adapter_name: str) -> ChatContext:
    adapter = backend._added_adapters.get(f"{adapter_name}_lora")
    if isinstance(adapter, LocalIntrinsicAdapter) and adapter.system_prompt:
        return ctx.add(Message("system", adapter.system_prompt))
    return ctx
```

This means the system prompt travels with the adapter weights, not with the pipeline code. Changing the prompt for a specific LoRA requires only updating its `io.yaml`.

---

## 7. Node and Flow Architecture

### 7.1 Why nodes

FCPipeline currently calls four LoRA adapters: router, parallel executor, multi-step executor, and conversational. As more flows are added (for example, a shortlisting pipeline that activates different LoRAs, or a reflector-augmented pipeline), the logic for calling each adapter would need to be duplicated across pipelines.

The node abstraction prevents this. Each LoRA call is encapsulated in a standalone `Node` class. A pipeline is a thin class that wires nodes together using plain Python conditionals. Adding a new pipeline means composing existing nodes in a new order, not copying LoRA call logic.

### 7.2 Node structure

All nodes live in a single file: `fc/nodes.py`. This is the right scale for the current codebase. A nodes package with per-node files is only warranted if a second pipeline forces copy-paste; until then, one file is easier to navigate.

The base class is `Node`, an abstract class with a single required method:

```python
class Node(ABC):
    @abstractmethod
    def __call__(self, ctx: ChatContext, **kwargs) -> Any: ...
```

Concrete nodes in the initial build:

| Node | LoRA | Input | Output |
|---|---|---|---|
| `RouterNode` | `fc_router` | `ChatContext` | `str` (category: `"parallel"`, `"multi_step"`, `"conversational"`) |
| `ExecutorNode` | `fc_parallel` or `fc_multi_step` | `ChatContext` + tools | OpenAI assistant message dict (tool calls) |
| `ConversationalNode` | `fc_conversational` | `ChatContext` | OpenAI assistant message dict (content) |

`ExecutorNode` is parameterized by adapter name so both parallel and multi-step executors can share the same class body.

### 7.3 Pipeline as a thin flow

`FunctionCallingPipeline` in `fc/pipeline.py` wires nodes with plain Python:

```python
class FunctionCallingPipeline:
    def __init__(self, ...):
        self._router    = RouterNode(backend, adapter_paths["fc_router"])
        self._parallel  = ExecutorNode(backend, adapter_paths["fc_parallel"], "fc_parallel")
        self._multi     = ExecutorNode(backend, adapter_paths["fc_multi_step"], "fc_multi_step")
        self._conv      = ConversationalNode(backend, adapter_paths["fc_conversational"])

    def run(self, messages, tools):
        ctx = messages_to_chat_context(messages)
        category = self._router(ctx)

        if category == "parallel":
            return self._parallel(ctx, tools=tools)
        elif category == "multi_step":
            return self._multi(ctx, tools=tools)
        else:
            return self._conv(ctx)
```

The pipeline does not know what a router is. It just calls `self._router(ctx)` and branches on the result. The routing logic and LoRA call are entirely inside `RouterNode`.

### 7.4 Multiple pipelines reusing the same nodes

When a second pipeline is needed (for example, a `ShortlistingPipeline` that inserts a tool shortlisting step before routing), it instantiates the relevant nodes and wires them in a new `run()` method:

```python
class ShortlistingPipeline:
    def __init__(self, ...):
        self._shortlist = ShortlistingNode(backend, adapter_paths["fc_shortlisting"])
        self._router    = RouterNode(backend, adapter_paths["fc_router"])
        self._parallel  = ExecutorNode(...)
        ...

    def run(self, messages, tools):
        ctx = messages_to_chat_context(messages)
        tools = self._shortlist(ctx, tools=tools)   # prune tool list first
        category = self._router(ctx)
        ...
```

`RouterNode`, `ExecutorNode`, and `ConversationalNode` are reused unchanged. The shortlisting step is added by composing a new node without touching existing ones.

### 7.5 What we are not building

This is not a graph execution framework. There is no node registry, no edge specification language, no topological sort, and no dynamic dispatch based on node names. The "flow" is just a Python method body. If the requirement ever grows to the point where a real graph framework (LangGraph, Prefect) is justified, the node classes are the natural unit to port. Until then, plain Python is sufficient, more readable, and easier to debug.

Rule: extract a node when the LoRA call logic would otherwise be copy-pasted across two pipelines. Do not extract until that moment.

---

## 8. FunctionCallingPipeline (FCPipeline)

### 8.1 Responsibilities

- Resolve adapter paths from a root directory following the naming convention, with optional per-adapter overrides.
- Accept a messages array in OpenAI format and a tool list as `list[MelleaTool] | list[dict]`.
- Convert dict tools to `MelleaTool` with no-op callables internally.
- Reconstruct a `ChatContext` from the messages array on every call. There is no cached state between calls.
- Route the request to the appropriate LoRA adapter using `fc_router`.
- Call the executor LoRA (`fc_parallel`, `fc_multi_step`, or `fc_conversational`) based on the routing decision.
- Return an OpenAI-format assistant message dict.

### 8.2 Interface

```python
class FunctionCallingPipeline:
    def __init__(
        self,
        backend: LocalHFBackend,
        adapters_dir: str | Path | None = None,       # primary: auto-discover by convention
        adapter_overrides: dict[str, str] | None = None,  # override individual adapters
        adapter_paths: dict[str, str] | None = None,  # escape hatch: full manual control
    ) -> None: ...

    def run(
        self,
        messages: list[dict],                      # OpenAI-format message history
        tools: list[MelleaTool] | list[dict],      # MelleaTool or OpenAI tool schema dicts
    ) -> dict:                                     # OpenAI-format assistant message
        ...
```

**Adapter discovery** follows a directory naming convention owned by `FunctionCallingPipeline`:

| Subdirectory name | Logical adapter name |
|---|---|
| `router` | `fc_router` |
| `parallel_tool_calling` | `fc_parallel` |
| `multi_step_tool_calling` | `fc_multi_step` |
| `conversational_detection` | `fc_conversational` |

The primary usage pattern passes only `adapters_dir`:

```python
pipeline = FunctionCallingPipeline(backend, adapters_dir="/path/to/fc-system")
```

To substitute a single adapter without abandoning auto-discovery, pass `adapter_overrides`. Explicit entries win over discovered paths:

```python
pipeline = FunctionCallingPipeline(
    backend,
    adapters_dir="/path/to/fc-system",
    adapter_overrides={"fc_router": "/experiments/my_custom_router"},
)
```

To bypass auto-discovery entirely and control all paths manually, pass `adapter_paths` instead of `adapters_dir`. Either `adapters_dir` or `adapter_paths` must be provided; providing both is an error.

The return value of `run()` is always an OpenAI assistant message dict. Callers distinguish tool call responses from conversational responses by checking whether `result["tool_calls"]` is non-empty.

The method is named `run`, not `chat`, to reinforce that this is a pipeline step, not a conversational interface.

### 8.3 Internal Flow

```
run(messages, tools)
  │
  ▼
convert list[dict] tools → list[MelleaTool] (no-op callables)
  │
  ▼
reconstruct ChatContext from messages
  (handles user, assistant, tool roles)
  │
  ▼
route(context, tools)  [fc_router LoRA]
  returns: "parallel" | "multi_step" | "conversational"
  │
  ├── parallel      → fc_parallel LoRA      → parse tool calls
  │                                         → return OpenAI assistant message
  │                                           with tool_calls populated
  │
  ├── multi_step    → fc_multi_step LoRA    → parse tool calls
  │                                         → return OpenAI assistant message
  │                                           with tool_calls populated
  │
  └── conversational → fc_conversational LoRA → return OpenAI assistant message
                                               with content populated, tool_calls null
```

There is no loop inside `run()`. A single call produces a single decision. The pipeline does not inspect tool results or decide whether to invoke more tools. That is the caller's concern. This keeps FCPipeline testable in isolation and directly usable by BenchmarkServer without hidden state.

---

## 9. FunctionCallingAgent (FCAgent)

### 9.1 Responsibilities

- Maintain conversation history as a list of OpenAI-format message dicts across turns.
- Own a registered set of `MelleaTool` instances with real callable implementations.
- Drive FCPipeline in a loop until a terminal condition is reached.
- Execute tool calls returned by the pipeline, format results as OpenAI `tool` role messages, and append them to history.
- Expose a `chat(user_message)` interface that returns a final natural language response.
- Provide a `reset()` method to clear conversation history for a new session.

### 9.2 Interface

```python
@dataclass
class AgentTurn:
    message: dict              # final OpenAI assistant message (content populated)
    tool_calls_made: list[dict]  # all tool calls across all pipeline iterations
    tool_results: list[dict]     # corresponding tool results
    iterations: int              # number of pipeline iterations used

class FunctionCallingAgent:
    def __init__(
        self,
        pipeline: FunctionCallingPipeline,
        tools: list[MelleaTool],
        max_iterations: int = 10,
    ) -> None: ...

    def chat(self, user_message: str) -> AgentTurn: ...

    def reset(self) -> None: ...
```

FCAgent takes a `FunctionCallingPipeline` instance rather than constructing one internally. This keeps construction concerns separate and makes FCAgent testable with a mock pipeline.

`AgentTurn.message` is always an OpenAI assistant message dict with `content` populated and `tool_calls` null: it is the final response delivered to the user after the loop completes.

### 9.3 Internal Loop

```
chat(user_message)
  │
  ▼
append {"role": "user", "content": user_message} to history
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  AGENTIC LOOP  (max_iterations budget)                   │
│                                                          │
│  assistant_msg = pipeline.run(history, tools)            │
│                                                          │
│  if assistant_msg["tool_calls"] is None:                 │
│    append assistant_msg to history                       │
│    return AgentTurn(message=assistant_msg, ...)  ← done  │
│                                                          │
│  if assistant_msg["tool_calls"] is not None:             │
│    append assistant_msg to history                       │
│    for each tool_call in assistant_msg["tool_calls"]:    │
│      execute tool → result                               │
│      append {"role": "tool",                             │
│               "tool_call_id": tool_call["id"],           │
│               "content": json.dumps(result)}             │
│    (continue loop)                                       │
│                                                          │
│  if iterations exhausted:                                │
│    return AgentTurn with partial state                   │
└──────────────────────────────────────────────────────────┘
```

The loop terminates naturally when FCPipeline returns an assistant message with no tool calls, meaning the routing LoRA determined a natural language response is appropriate given the accumulated context (including tool results from prior iterations). The loop is also bounded by `max_iterations` to prevent runaway execution.

---

## 10. How Benchmarks Work

### 10.1 The Stateless Contract

All benchmarks in scope manage conversation state themselves. On each turn, they send a complete message history: all prior user messages, assistant responses, tool call records, and tool results. BenchmarkServer receives full context on every request and must produce one decision per request. There is no persistent session on the server side.

This means benchmarks interact exclusively with FCPipeline (via BenchmarkServer). They do not use FCAgent. From the benchmark's perspective, the system is a stateless HTTP service that accepts messages and returns tool calls or a response.

The multi-turn loop that benchmarks exercise is driven externally:

```
Benchmark                              BenchmarkServer → FCPipeline
─────────                              ─────────────────────────────
POST /chat                        ──►  translate → run → return
  {messages: [user_msg],               assistant_msg with tool_calls
   tools: [...]}
◄──  {tool_calls: [...]}

execute tools (benchmark-side)
append tool results to messages

POST /chat                        ──►  translate → run → return
  {messages: [user_msg,                assistant_msg with content
              assistant(tool_calls),
              tool_results],
   tools: [...]}
◄──  {content: "Here is your answer..."}

score response against ground truth
```

The benchmark and BenchmarkServer together form the same loop that FCAgent internalizes. The difference is where the loop boundary sits.

### 10.2 Benchmark Integration Summary

| Benchmark | Protocol | Endpoint | Discovery | Tool Execution |
|---|---|---|---|---|
| BFCL | Custom REST | `POST /chat` | `$SERVER_URL` env var | Benchmark-side (mock implementations) |
| TAU2 Bench | OpenAI-compatible | `POST /v1/chat/completions` | `--agent-llm-args api_base=...` | Benchmark-side (environment simulation) |
| ACEBench | Custom REST | `POST /chat` | `$SERVER_URL` env var | Benchmark-side |

### 10.3 BFCL

BFCL uses a two-phase CLI: `bfcl generate` (inference) then `bfcl evaluate` (scoring). The generate phase reads `$SERVER_URL` and uses `fc-agent` as the model name identifier. It sends requests in a custom format to `POST /chat`. Categories include `simple`, `parallel`, `multiple`, and `multi_turn`. For `multi_turn` categories, the benchmark drives the turn loop by sending accumulated message history on each call.

BFCL's tool schema format differs from OpenAI's. BenchmarkServer translates incoming BFCL tool dicts to OpenAI tool schema dicts before calling FCPipeline, and translates FCPipeline's OpenAI response back to BFCL's expected response format before returning.

### 10.4 TAU2 Bench

TAU2 is the outlier. It uses the OpenAI Python SDK internally and expects a fully OpenAI-compatible `/v1/chat/completions` endpoint. The server URL is passed via `--agent-llm-args '{"api_base": "http://localhost:PORT"}'`. Domains include `airline`, `retail`, and `telecom`. Tool execution is handled by TAU2's environment simulation layer.

Because TAU2 already speaks OpenAI format, the translation in BenchmarkServer for this endpoint is minimal: deserialize the OpenAI request, call `pipeline.run()`, serialize the OpenAI response. The `/v1/chat/completions` handler is essentially a pass-through.

### 10.5 ACEBench

ACEBench has a two-phase evaluation: `generate.py` (inference) then `eval_main.py` (scoring). It uses the same custom `/chat` format as BFCL, configured via `$SERVER_URL`. It additionally requires `$RITS_API_KEY` for a user-side LLM (separate from the inference server). Language and category are configurable at run time. BenchmarkServer handles ACEBench via the same `/chat` endpoint as BFCL.

---

## 11. BenchmarkServer

### 11.1 Why It Exists

Benchmarks are external tools with their own CLIs and HTTP clients. They cannot be modified to call Python functions directly. BenchmarkServer is the boundary adapter that makes FCPipeline look like what each benchmark expects. It is the thinnest possible layer between HTTP and the pipeline.

All protocol translation lives here. FCPipeline and FCAgent know nothing about benchmark-specific formats.

### 11.2 Endpoints

```
POST /chat
  Used by:  BFCL, ACEBench
  In:       benchmark-specific request format
  Action:   translate tools and messages to OpenAI format
            call pipeline.run(messages, tools)
            translate OpenAI response to benchmark-specific response format
  Out:      benchmark-specific response format

POST /v1/chat/completions
  Used by:  TAU2 Bench (via OpenAI Python SDK)
  In:       OpenAI ChatCompletion request
  Action:   call pipeline.run(messages, tools) directly (minimal translation)
  Out:      OpenAI ChatCompletion response

GET /health
  Used by:  startup health-poll loop in benchmark scripts
  Out:      {"status": "ok"}
```

Startup sequence:
1. Parse CLI args: `--model`, `--adapters`, `--port`.
2. Load base model into `LocalHFBackend`.
3. Register all LoRA adapters from the adapters directory.
4. Construct one `FunctionCallingPipeline` instance.
5. Serve. All endpoints share the single pipeline instance.

### 11.3 Protocol Translation

Each endpoint handler follows the same three-step pattern:

```
1. Deserialize and translate request to OpenAI format
2. call pipeline.run(openai_messages, openai_tools)
3. Translate OpenAI response to benchmark format and return
```

Step 1 and step 3 are the only places that know about benchmark-specific schemas. If BFCL changes its request format, only step 1 of the `/chat` handler changes. FCPipeline is untouched.

### 11.4 What the Server Must Not Do

- Hold conversation state between requests. Every request is self-contained.
- Execute tools. That is FCAgent's job when used in a live system; in the benchmark case, the benchmark executes tools.
- Add retry logic, fallback behavior, or routing decisions.
- Let any logic creep beyond the three-step translate-run-translate pattern.

If it is tempting to add logic to a request handler, that logic belongs in FCPipeline.

---

## 12. How a Live System Works

When a developer uses FCAgent in an application, the interface is simple. The developer constructs a pipeline, registers tools with real callables, and calls `agent.chat()` with plain user messages. All LoRA routing, tool execution, loop management, and history accumulation are internal.

```python
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

pipeline = FunctionCallingPipeline(
    backend=backend,
    adapter_paths={
        "fc_router": "/adapters/router",
        "fc_parallel": "/adapters/parallel_tool_calling",
        "fc_multi_step": "/adapters/multi_step_tool_calling",
        "fc_conversational": "/adapters/conversational_detection",
    },
)

agent = FunctionCallingAgent(
    pipeline=pipeline,
    tools=[get_weather, book_hotel, search_flights],  # real MelleaTool callables
    max_iterations=10,
)

turn = agent.chat("What is the weather in San Francisco?")
print(turn.message["content"])

turn = agent.chat("And what about New York?")   # history maintained automatically
print(turn.message["content"])

agent.reset()   # clear history for a new conversation
```

The caller never sees `ChatContext`, `Intrinsic`, LoRA names, routing decisions, or Mellea internals. The interface is indistinguishable from calling a hosted chat API.

---

## 13. Build Order

Each step is independently testable before moving to the next.

1. **FCPipeline core (flat)**: implement `FunctionCallingPipeline` with LoRA call logic inlined. Reconstruct `ChatContext` from a messages array, handle dict-to-`MelleaTool` conversion, route, execute, and return an OpenAI assistant message. Unit-test with canned message arrays and mock LoRA outputs.

2. **Node refactor**: extract `RouterNode`, `ExecutorNode`, and `ConversationalNode` into `fc/nodes.py`. Rewrite `FunctionCallingPipeline.run()` to wire nodes with plain Python. Behavior is unchanged; existing tests must still pass.

3. **End-to-end pipeline validation**: run a single BFCL-style multi-turn conversation against a real backend with no server. Confirm the returned OpenAI messages are correct for both tool call and conversational cases.

4. **BenchmarkServer `/chat` endpoint**: thin wrapper with BFCL/ACEBench protocol translation. Test with `curl`. Confirm BFCL can connect and score a single-turn example.

5. **BenchmarkServer `/v1/chat/completions` endpoint**: OpenAI-compatible pass-through for TAU2. Test with the OpenAI Python SDK pointed at the local server.

6. **FCAgent**: stateful wrapper with agentic loop, tool registry, history management, and `reset()`. Unit-test the loop logic with a mock pipeline that returns predetermined sequences of tool call and response messages.

7. **Benchmark script generation**: generate shell scripts that start BenchmarkServer, health-poll until ready, run the benchmark CLI, and clean up. Modeled on the `evaluate_v2.py` pattern from the existing system.

---

## 14. Open Questions

- **Self-correct LoRA**: resolved in principle, deferred in implementation. Self-correction belongs inside FCPipeline as an optional post-execution step: the executor LoRA produces tool calls, `fc_self_correct` reviews and potentially rewrites them, and the corrected result is returned to the caller. The caller (benchmark or FCAgent) never observes the intermediate state. This is distinct from FCAgent retrying after a bad tool result, which is a reaction to real-world effects. Self-correction is purely about fixing the model's own output before it leaves the pipeline. Implementation is out of scope for the initial build.

- **Workflow strategies for FCAgent**: the current design has one fixed loop strategy (call pipeline, execute tools, repeat). If plan-and-execute or other strategies are needed, FCAgent should accept a pluggable workflow object. Deferred until there is a concrete use case.

- **Multi-step loop internalization**: the current design keeps the multi-step loop external (benchmark-driven or FCAgent-driven). The `multi_step` route in FCPipeline returns a single batch of tool calls per invocation. If a future requirement needs FCPipeline itself to iterate (e.g., execute a chain of dependent calls within a single `run()` call), that would require revisiting FCPipeline's stateless contract and is out of scope for now.
