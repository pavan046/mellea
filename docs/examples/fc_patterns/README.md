# Function-Calling Pattern Examples

Five progressive patterns showing how to compose modular LoRA adapters into function-calling pipelines. Each pattern adds a capability on top of the previous one.

All examples use `LocalHFBackend` with `ibm-granite/granite-4.0-micro` and locally trained LoRA adapters from the function-calling-agent project.

## Prerequisites

Adapter checkpoints must be available at:
```
/proj/dmfexp/dgt/checkpoints/tuned/tc_capabilities/
exp01_granite4_3b_rerun_tuned11/checkpoints/
```

## Files

### 101_route_and_execute.py

The base modular pattern. A router LoRA classifies the query, then a specialized adapter executes it.

**Key Features:**
- `LocalIntrinsicAdapter` for loading local LoRA weights as Mellea intrinsics
- Router classifies: parallel, multi_step, or conversational
- Dynamic adapter dispatch based on classification
- `session.chat()` bypass for conversational (plain text) output

### 102_shortlist_route_execute.py

Adds tool catalog filtering before routing. The shortlister LoRA selects relevant tools from a large catalog (15+), reducing noise for downstream steps.

**Key Features:**
- `shortlist_tools()` using the tool_shortlisting adapter
- Large 15-tool catalog to demonstrate filtering value
- Filtered tools passed to both router and executor

### 103_confidence_gated_routing.py

Adds confidence scoring to the router. When confidence is below a threshold, falls back to a monolithic baseline adapter instead of risking a misclassification.

**Key Features:**
- Router response_format extended with `confidence` field
- Tunable `CONFIDENCE_THRESHOLD` (default: 0.7)
- Fallback to `combined_baseline` adapter on low confidence

### 104_reflect_and_retry.py

**The key pattern.** Implements the Instruct-Validate-Repair loop for function calling: generate tool calls, validate against schemas, reflect on errors using a dedicated LoRA, and retry with guidance.

**Key Features:**
- `validate_tool_calls()` — programmatic schema validation
- `reflect_on_error()` — reflector adapter for error analysis
- `run_with_retry()` — manual IVR loop with `MAX_RETRIES`
- Reflection guidance injected into retry context

### 105_direct_baseline.py

Single monolithic adapter, no routing. Same test queries as 101 for direct comparison of modular vs monolithic approaches.

**Key Features:**
- `combined_baseline` LoRA trained on all categories
- Simplest pipeline — one adapter, one step
- Comparison baseline for measuring routing benefit

## Concepts Demonstrated

- **Modular LoRA Routing**: Specialized adapters for each task type
- **Tool Catalog Filtering**: Pre-processing large tool sets
- **Confidence-Gated Fallback**: Robustness via uncertainty handling
- **Reflect-and-Retry as IVR**: Intrinsic-based validate-repair loop
- **Local Adapter Registration**: `LocalIntrinsicAdapter` for filesystem checkpoints

## Pipeline Architecture

```
              101 Route-and-Execute (base)
              ┌────────────────────────────┐
  query ──►  route  ──►  execute (dynamic)
              └────────────────────────────┘

  102 Shortlist              103 Confidence          104 Reflect-and-Retry     105 Baseline
  ┌──────────────┐          ┌──────────────┐        ┌──────────────────┐      ┌──────────┐
  │ shortlist ──►│          │ route +conf  │        │ execute ──►      │      │ execute  │
  │ route ──►    │          │   high? ──►  │        │   validate ──►   │      │ (mono)   │
  │ execute      │          │   low? ──►   │        │   reflect ──►    │      └──────────┘
  │ (filtered)   │          │   fallback   │        │   retry (loop)   │
  └──────────────┘          └──────────────┘        └──────────────────┘
```

## Key Pattern: Reflect-and-Retry as IVR

The central insight of Pattern 104 — the IVR mapping:

| IVR Step | RAG 104 (Hallucination Repair) | FC 104 (Reflect-and-Retry) |
|----------|-------------------------------|---------------------------|
| **Instruct** | `session.instruct()` | `mfuncs.act()` on intrinsic |
| **Validate** | `flag_hallucinated_content()` | `validate_tool_calls()` |
| **Repair** | `RepairTemplateStrategy` | Reflector adapter + guidance |
| **Loop** | Built-in `loop_budget=3` | Manual retry loop |

```python
for attempt in range(MAX_RETRIES):
    result = execute_tool_call(question, tools, category, ...)     # Instruct
    is_valid, error_msg = validate_tool_calls(result, tools)       # Validate
    if is_valid:
        return result
    guidance = reflect_on_error(question, result, error_msg, ...)  # Repair
    # guidance is injected into the next attempt's system prompt
```

## Related Documentation

- See `rag_patterns/` for RAG intrinsic compositions
- See `instruct_validate_repair/` for the core IVR pattern
- See `intrinsics/` for individual intrinsic usage examples
- See `mellea/backends/adapters/` for adapter system internals
