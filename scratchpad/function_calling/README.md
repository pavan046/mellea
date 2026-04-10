# Function-Calling Pipeline: Notes and Bugs

## Files

| File | Description |
|------|-------------|
| `simple_pipeline.py` | Original FC pipeline using raw dicts for tool schemas |
| `simple_pipeline_mellea_conventions.py` | Same pipeline using `@tool` decorator and Mellea conventions |
| `multi_turn_fc_pipeline.py` | Multi-turn pipeline with router, executor, and conversational LoRAs; native tool execution via `parse_and_execute_tool_calls` |

## Bugs Found in Mellea

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

## Workarounds

`multi_turn_fc_pipeline.py` implements `parse_and_execute_tool_calls`, a local helper that sidesteps bugs 1–3 by parsing `mot.value` directly:

- Handles both bare array `[{...}, {...}]` and wrapped `{"tool_calls": [{...}]}` formats
- Preserves duplicate tool calls (e.g., two `get_weather` calls)
- Runs `validate_tool_arguments` for type coercion
- Returns a `list[Message]` (one assistant message with tool call JSON in content, followed by one `ToolMessage` per executed call)

Bug 4 is fixed directly in `mellea/helpers/openai_compatible_helpers.py`.
