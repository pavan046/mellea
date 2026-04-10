# How to Write an `io.yaml` for a LoRA Adapter

Every LoRA adapter that will be used as a Mellea intrinsic must include an `io.yaml`
file co-located with its weights. This file tells Mellea how to format the model's
input and how to interpret its output. Without it, the adapter cannot be loaded.

Place `io.yaml` directly in the adapter's checkpoint directory, alongside
`adapter_config.json` and the weight files:

```
my_adapter/
    io.yaml                       ← this file
    adapter_config.json
    adapter_model.safetensors
```

---

## Table of Contents

1. [Minimal example](#1-minimal-example)
2. [Required fields](#2-required-fields)
   - [model](#21-model)
   - [response_format](#22-response_format)
   - [transformations](#23-transformations)
3. [Optional fields](#3-optional-fields)
   - [parameters](#31-parameters)
   - [system_prompt](#32-system_prompt)
   - [instruction](#33-instruction)
   - [docs_as_message](#34-docs_as_message)
   - [sentence_boundaries](#35-sentence_boundaries)
   - [logprobs_workaround](#36-logprobs_workaround)
4. [Default JSON output convention](#4-default-json-output-convention)
5. [fc-system adapter examples](#5-fc-system-adapter-examples)
6. [Checklist before sharing your adapter](#6-checklist-before-sharing-your-adapter)

---

## 1. Minimal example

If your adapter outputs free-form text wrapped in a simple JSON object, this is all
you need:

```yaml
model: ~
response_format: |
  {
    "type": "object",
    "properties": {
      "response": { "type": "string" }
    },
    "required": ["response"]
  }
transformations: null
```

`~` means null in YAML. `transformations: null` means no post-processing. These are
valid values — do not omit them. All three fields are required.

---

## 2. Required fields

### 2.1 `model`

The logical model name this adapter targets, or `~` (null) to use whatever base model
is currently loaded.

**Almost always use `~`.** Hardcoding a model name pins the adapter to a specific
base model string and will cause errors if the adapter is run against a differently
named model, even if the weights are identical.

```yaml
model: ~
```

Use a specific name only if your adapter is strictly incompatible with other base
models and you want Mellea to enforce that at load time.

### 2.2 `response_format`

A JSON Schema describing the structure of your adapter's output. Mellea uses this to
validate and parse the model's response.

**Important:** the value is written as a YAML block scalar (using `|`), which means
it is treated as a raw string and then parsed as JSON internally. Do not write it as
inline YAML — write it as a JSON string.

```yaml
response_format: |
  {
    "type": "object",
    "properties": {
      "label": { "type": "string" },
      "confidence": { "type": "number" }
    },
    "required": ["label", "confidence"]
  }
```

For adapters that output a list of items (e.g. tool calls):

```yaml
response_format: |
  {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "arguments": { "type": "object" }
      },
      "required": ["name", "arguments"]
    }
  }
```

Your adapter's training data should produce output that matches this schema. If the
schema does not match what the model outputs, downstream parsing will fail.

### 2.3 `transformations`

A list of post-processing rules applied to the model's JSON output before it is
returned to the caller. Set to `null` if you do not need any post-processing.

```yaml
transformations: null
```

When non-null, this is a list of transformation steps applied in order. Each step
has a `type` and an `input_path` (a JSON path into the output object, or `[]` for
the root). The supported types are:

| Type | What it does |
|------|--------------|
| `likelihood` | Converts a categorical label (e.g. "answerable") to a numeric probability using logprobs. Requires `categories_to_values` dict. |
| `nest` | Wraps the value at `input_path` in a single-field object. Requires `field_name`. |
| `decode_sentences` | Expands sentence reference indices (e.g. `<c0>`) back into begin/end/text spans. Requires `source` and `output_names`. Pairs with `sentence_boundaries`. |
| `explode` | Expands a list-valued field so each element becomes its own record. Requires `target_field`. |
| `drop_duplicates` | Removes duplicate records from a list. Requires `target_fields`. |
| `project` | Keeps only specified fields in each record. Requires `fields`. |
| `merge_spans` | Merges adjacent contiguous spans. Requires `group_fields`, `begin_field`, `end_field`. |

Example: convert a classification label to a likelihood score.

```yaml
transformations:
  - type: likelihood
    input_path: []
    categories_to_values:
      "yes": 1.0
      "no": 0.0
  - type: nest
    input_path: []
    field_name: "is_relevant_likelihood"
```

For most function-calling adapters, `transformations: null` is the right choice.
Transformations are primarily used by RAG and classification intrinsics.

---

## 3. Optional fields

These fields can be omitted entirely. If omitted, Mellea treats them as null
internally.

### 3.1 `parameters`

Model generation parameters passed to the inference backend. The most commonly used
key is `max_completion_tokens`.

```yaml
parameters:
  max_completion_tokens: 512
```

Set this to a value that comfortably fits your adapter's expected output length.
Leaving it unset lets the backend use its own default, which may be too large (slow)
or too small (truncated output).

Rule of thumb by output type:

| Output type | Suggested `max_completion_tokens` |
|-------------|----------------------------------|
| Classification / routing (short JSON) | 128–256 |
| Single tool call or response | 512 |
| Multi-tool call arrays | 512–1024 |
| Long natural language responses | 1024+ |

### 3.2 `system_prompt`

> **Note:** This field is an extension to Mellea's built-in io.yaml schema and is
> not supported by Mellea natively. It is handled by `LocalIntrinsicAdapter` in the
> fc-system integration layer, which pops the field before passing the config to
> Mellea's internal validator. Do not use this field if your adapter is loaded
> through Mellea's standard Hub-based adapter loading.

A string injected as a `system` role message prepended to the conversation
immediately before inference. This is the correct field to use if your LoRA was
trained with the guiding prompt as a system message.

```yaml
system_prompt: |
  You are a function-calling assistant. Given the user's request and
  available tools, respond ONLY with a JSON array of tool calls.
```

If `system_prompt` is set, the message list that reaches the tokenizer becomes:

```
[system]  "<contents of system_prompt>"     ← injected from io.yaml
[user]    "Search for flights from SF to NYC on Jan 15."
```

**Use `system_prompt` when:** your LoRA was trained with the guiding prompt as a
`system` role message and you want to co-locate that prompt with the adapter weights
rather than managing it in application code.

**Do not use `system_prompt` when:** your LoRA was not trained with a system message,
or when you need the prompt to be dynamic (per-call). For dynamic prompts, inject
them in application code before calling the pipeline.

### 3.3 `instruction`

A string appended to the conversation as a new user message immediately before
generation. It is a template: use `{last_message}` to reference the content of
the last message in the conversation, and any other `{key}` placeholders for
values passed at call time.

**How `instruction` changes the message structure**

Without `instruction`, the message list that reaches the tokenizer is exactly
what the caller provided:

```
[system]  "You are a helpful assistant."
[user]    "Search for flights from SF to NYC on Jan 15."
```

With `instruction` set in `io.yaml`, Mellea appends a new user message after
all existing messages:

```
[system]  "You are a helpful assistant."
[user]    "Search for flights from SF to NYC on Jan 15."
[user]    "Given the conversation above, generate a JSON array of tool calls."   ← appended
```

The instruction is always role `user`, always the final message, and always
appended after everything else. It is not a system message and it is not merged
into any existing message.

**Important: only use `instruction` if your adapter was trained with this message
structure.** If your LoRA was trained with the guiding prompt as a `system` role
message, using `instruction` will produce a different message structure at
inference time than the model saw during training, which will degrade output
quality. In that case, leave `instruction` null and inject the system prompt in
your application code.

> **Training tip:** If you are training a new LoRA and want to simplify Mellea
> integration, consider training with the guiding prompt as a final `user`
> message rather than a `system` message. This lets you embed the prompt
> directly in `io.yaml` via `instruction`, removing the need for any
> application-side prompt management.

```yaml
instruction: |
  Given the conversation above and the available tools, generate a JSON array
  of tool calls. Each element must have "name" and "arguments" fields.
```

If you manage the prompt in application code, leave this null or omit it.

### 3.4 `docs_as_message`

Controls how RAG documents (passed via `extra_body/documents`) are injected into the
prompt. Only relevant if your adapter was trained with document context. Most
function-calling adapters can omit this.

| Value | Behavior |
|-------|----------|
| omitted or `null` | Documents stay in `extra_body`. Works with servers that support it. |
| `"string"` | Documents are serialized as plain text and prepended to the first user message. |
| `"json"` | Documents are serialized as a raw JSON array in the first user message. |
| `"roles"` | Each document becomes a separate message with role `"document {id}"`. Use for Ollama. |

### 3.5 `sentence_boundaries`

Inserts sentence boundary markers into the input before inference, enabling the model
to reference specific sentences in its output by index (e.g. `<c0>`, `<c1>`). Only
relevant for extractive or citation-aware adapters trained with this convention.

```yaml
sentence_boundaries:
  last_message: "c"          # user message sentences become <c0>, <c1>, ...
  documents: "d"             # document sentences become <d0>, <d1>, ...
  all_but_last_message: "h"  # history sentences become <h0>, <h1>, ...
```

Pairs with the `decode_sentences` transformation to convert sentence indices in the
output back into begin/end/text spans. Omit entirely for function-calling adapters.

### 3.6 `logprobs_workaround`

Set to `true` if the adapter was trained in a setup where the inference server may
corrupt the output by failing to strip internal control tokens (Harmony-style tokens
like `<|channel|>`, `<|message|>`, `<|end|>`). When enabled, Mellea reconstructs
the output from logprob tokens instead of `message.content`.

```yaml
logprobs_workaround: true
```

Leave omitted or null unless you have a specific reason to enable this. It requires
the backend to return logprobs, which not all servers support.

---

## 4. Default JSON output convention

All fc-system intrinsics are expected to produce valid JSON output. This is not
enforced by Mellea itself, but it is a convention we rely on: every intrinsic should
wrap its output in a JSON object or array so that downstream code can parse it
reliably.

If your adapter does not have a more specific output format, the baseline expectation
is a single-key JSON object:

```json
{"response": "<your output here>"}
```

When training a new adapter without a specific structured output requirement, train
it to produce this format and set `response_format` accordingly:

```yaml
response_format: |
  {
    "type": "object",
    "properties": {
      "response": { "type": "string" }
    },
    "required": ["response"]
  }
```

This convention ensures that:

- Parsing code has a consistent contract across all intrinsics.
- Free-form text output is never returned as a raw unstructured string.
- Adapters that do not yet have a known integration point (such as `reflector` or
  `tool_shortlisting` in the fc-system) can still be loaded and called without
  breaking the pipeline.

Adapters with more specific output shapes (tool call arrays, routing categories)
should use those shapes instead of the generic `response` wrapper. The convention
is a floor, not a ceiling.

---

## 5. fc-system adapter examples

The following are the `io.yaml` files used by the fc-system LoRA adapters. Use
these as concrete references.

### router

Classifies a user request into one of three routing categories.

```yaml
model: ~
response_format: |
  {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "enum": ["parallel", "multi_step", "conversational"]
      },
      "reasoning": { "type": "string" }
    },
    "required": ["category", "reasoning"]
  }
transformations: null
parameters:
  max_completion_tokens: 256
```

### parallel_tool_calling / multi_step_tool_calling

Generates a list of tool calls.

```yaml
model: ~
response_format: |
  {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "arguments": { "type": "object" }
      },
      "required": ["name", "arguments"]
    }
  }
transformations: null
parameters:
  max_completion_tokens: 512
```

### conversational_detection

Generates a natural language response.

```yaml
model: ~
response_format: |
  {
    "type": "object",
    "properties": {
      "response": { "type": "string" }
    },
    "required": ["response"]
  }
transformations: null
parameters:
  max_completion_tokens: 512
```

---

## 6. Checklist before sharing your adapter

Before handing off a trained adapter, verify the following:

- [ ] `io.yaml` exists in the root of the checkpoint directory (same level as `adapter_config.json`)
- [ ] All three required fields are present: `model`, `response_format`, `transformations`
- [ ] `model` is set to `~` unless there is a specific reason to pin it
- [ ] `response_format` is valid JSON written as a YAML block scalar (`|`)
- [ ] `response_format` matches the actual output format the adapter was trained to produce
- [ ] `parameters.max_completion_tokens` is set and appropriate for the output length
- [ ] `system_prompt` is set. If the adapter has a known role, use a specific prompt. If unknown or not yet integrated, use the default JSON fallback: `You are a helpful assistant. Always respond in JSON. Wrap your reply in a JSON object with a single key: {"response": "<your reply>"}`
- [ ] Adapter has been tested with a real inference call and the output parses correctly against `response_format`
