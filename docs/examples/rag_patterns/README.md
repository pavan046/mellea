# RAG Pattern Examples

Five progressive patterns showing how to compose retrieval, intrinsics, and Mellea's Instruct-Validate-Repair loop into production-ready RAG pipelines.

Each pattern builds on the previous one. All examples use `LocalHFBackend` with `ibm-granite/granite-4.0-micro` and a mock document retriever.

## Prerequisites

Download the model and adapter weights before running any example:

```bash
uv run python scratchpad/download_rag_assets.py
```

## Files

### 101_rewrite_retrieve_generate.py

The simplest intrinsic-enhanced RAG pipeline. Query rewriting resolves coreferences and makes retrieval more effective.

**Key Features:**
- `rag.rewrite_question()` for query rewriting
- `session.instruct()` with `grounding_context` for grounded generation
- Building block for all other patterns

### 102_answerability_gated_rag.py

Adds a quality gate: only generate an answer if the documents actually contain one.

**Key Features:**
- `rag.check_answerability()` returns a float (0-1)
- Tunable threshold for conservative vs. permissive gating
- Demonstrates both passing and failing queries

### 103_query_clarification_rag.py

Handles ambiguous queries by asking the user for clarification before generating.

**Key Features:**
- `rag.clarify_query()` returns `"CLEAR"` or a clarification question
- Interactive branching — the only pattern with user interaction mid-pipeline
- Port of the Langflow "Query Clarification (post-retriever) RAG" flow

### 104_hallucination_repair_rag.py

**The key pattern.** Composes hallucination detection with IVR's validate-repair loop.

**Key Features:**
- `rag.flag_hallucinated_content()` used as a `Requirement.validation_fn`
- `RepairTemplateStrategy` feeds flagged sentences back to the model
- `return_sampling_results=True` to inspect the repair loop
- Shows how intrinsics compose with IVR's existing machinery

### 105_citation_grounding_rag.py

Post-processing enrichment: annotate a response with citations and faithfulness scores.

**Key Features:**
- `rag.find_citations()` maps response sentences to source documents
- `rag.flag_hallucinated_content()` scores sentence-level faithfulness
- No loop — pure enrichment for downstream consumers (UI, audit logs)
- Documents use `doc_id` for citation tracking

## Concepts Demonstrated

- **Query Rewriting**: Reformulating conversational questions for retrieval
- **Answerability Gating**: Quality gate before generation
- **Query Clarification**: Interactive disambiguation
- **Hallucination Detection as IVR Validation**: Intrinsic as `validation_fn`
- **Citation Extraction**: Mapping responses back to sources
- **Grounding Context**: Feeding documents to `session.instruct()`
- **Intrinsic-IVR Composition**: The main architectural insight

## Pipeline Architecture

```
                    101 RRG (base)
                    ┌──────────────────────┐
  question ──► rewrite ──► retrieve ──► generate
                    └──────────────────────┘

        102 RVG               103 RCG              104 GDR               105 GCV
   (answerability gate)  (clarification gate)  (hallucination repair)  (citation enrichment)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       ┌──────────────┐
   │ ... ──► check │      │ ... ──► clarify│    │ ... ──► IVR   │      │ ... ──► cite  │
   │   score≥T? ──►│      │   CLEAR?  ──► │    │   loop with  │      │         flag  │
   │   generate    │      │   generate    │    │   faithfulness│      │         enrich│
   └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

## Key Pattern: Intrinsic as IVR Validator

The central insight of Pattern 104 is that any intrinsic that returns a quality
score can be wrapped as a `Requirement.validation_fn`:

```python
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.sampling import RepairTemplateStrategy

def make_faithfulness_validator(documents, context, backend):
    def validate(ctx):
        output = ctx.last_output()
        flags = rag.flag_hallucinated_content(output.value, documents, context, backend)
        hallucinated = [f for f in flags if f["faithfulness_likelihood"] < 0.5]
        if hallucinated:
            return ValidationResult(False, reason=f"Hallucinated: {hallucinated}")
        return ValidationResult(True)
    return validate

result = session.instruct(
    "Answer the question: {{q}}",
    requirements=[Requirement("Be faithful", validation_fn=make_faithfulness_validator(...))],
    strategy=RepairTemplateStrategy(loop_budget=3),
)
```

This same pattern works with `check_answerability`, `check_context_relevance`,
or any custom scoring function.

## Related Documentation

- See `instruct_validate_repair/` for the core IVR pattern
- See `intrinsics/` for individual intrinsic usage examples
- See `rag/` for vector-search-based RAG examples
- See `mellea/stdlib/components/intrinsic/rag.py` for all RAG intrinsic functions
- See `mellea/stdlib/sampling/` for sampling strategies
