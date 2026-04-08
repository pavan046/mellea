# pytest: skip_always
"""Generate-Detect-Repair (GDR) — Hallucination Mitigation RAG via IVR.

This is the key pattern: it shows how Mellea's Instruct-Validate-Repair loop
composes with RAG intrinsics.  The hallucination detection intrinsic becomes
the *validate* step, and RepairTemplateStrategy feeds the flagged sentences
back to the model as repair feedback.

Pipeline
--------
    question  ──►  rewrite  ──►  retrieve  ──►  instruct(requirements=[faithfulness])
                                                       │
                                                 ┌─────┴─────┐
                                                 │  IVR loop  │
                                                 │  validate: │
                                                 │    flag_hallucinated_content()
                                                 │  repair:   │
                                                 │    RepairTemplateStrategy
                                                 └────────────┘
                                                       │
                                                       ▼
                                                  final answer

Langflow equivalent
-------------------
    ChatInput → QueryRewrite → Retriever → GraniteBaseModel
                                              → HallucinationDetection
                                              → HallucinationFeedback → GraniteBaseModel (loop)

When to use
-----------
Use GDR when faithfulness to source documents is critical (e.g. enterprise Q&A,
medical, legal).  The model generates an answer, the hallucination intrinsic
checks every sentence, and any unfaithful sentences trigger a repair cycle with
targeted feedback.

Run
---
    uv run python docs/examples/rag_patterns/104_hallucination_repair_rag.py
"""

import os
from pathlib import Path

MODEL_ID = "ibm-granite/granite-4.0-micro"
HF_HOME = Path("/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics")
model_cache_dir = HF_HOME / "hub" / ("models--" + MODEL_ID.replace("/", "--"))

if not model_cache_dir.exists():
    raise RuntimeError(
        f"Model not found at {model_cache_dir}. "
        "Run 'uv run python scratchpad/download_rag_assets.py' first."
    )

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"

from mellea.backends.huggingface import LocalHFBackend
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import RepairTemplateStrategy
from mellea.stdlib.session import MelleaSession

# ---------------------------------------------------------------------------
# Mock document corpus
# ---------------------------------------------------------------------------
_CORPUS = [
    Document(
        "ClapNQ is a long-form question answering benchmark derived from Natural "
        "Questions. It focuses on questions whose answers require full Wikipedia "
        "passages rather than short spans."
    ),
    Document(
        "Natural Questions (NQ) is a dataset released by Google consisting of real "
        "queries issued to the Google search engine, paired with Wikipedia pages "
        "that contain the answer."
    ),
    Document(
        "Retrieval-Augmented Generation (RAG) is a technique that combines a "
        "retrieval step with a generative language model so the model can ground "
        "its answers in retrieved documents."
    ),
    Document(
        "ELSER (Elastic Learned Sparse EncodeR) is a sparse retrieval model from "
        "Elastic used for semantic search without requiring dense vector embeddings."
    ),
]


def retrieve(query: str) -> list[Document]:
    """Mock retriever — returns the full corpus regardless of query."""
    return _CORPUS


# ---------------------------------------------------------------------------
# Faithfulness validator — the bridge between intrinsics and IVR
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.5


def make_faithfulness_validator(documents, chat_context, backend):
    """Create a validation_fn that checks hallucination via the intrinsic.

    Returns a ``Callable[[Context], ValidationResult]`` as required by
    ``Requirement.validation_fn``.  The closure captures ``documents``,
    ``chat_context``, and ``backend`` so the IVR loop can call it
    without any extra arguments.
    """

    def validate(ctx):
        output = ctx.last_output()
        if output is None or output.value is None:
            return ValidationResult(False, reason="No output to validate.")

        # flag_hallucinated_content returns a list of dicts, each containing
        # response_text, faithfulness_likelihood, explanation, etc.
        flags = rag.flag_hallucinated_content(
            output.value, documents, chat_context, backend
        )

        # If it's a string (edge case), treat as not parseable
        if isinstance(flags, str):
            return ValidationResult(True)

        hallucinated = [
            f
            for f in flags
            if f.get("faithfulness_likelihood", 1.0) < FAITHFULNESS_THRESHOLD
        ]

        if hallucinated:
            reasons = "; ".join(
                f"'{h.get('response_text', '?')}' "
                f"(faithfulness={h.get('faithfulness_likelihood', 0):.2f})"
                for h in hallucinated
            )
            return ValidationResult(
                False, reason=f"Hallucinated content detected: {reasons}"
            )

        return ValidationResult(True)

    return validate


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(question: str) -> None:
    context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))

    print("Loading model...")
    backend = LocalHFBackend(model_id=MODEL_ID)

    # Step 1 — Rewrite
    print(f"\n[1] Rewriting question: {question}")
    rewritten = str(rag.rewrite_question(question, context, backend))
    print(f"    Rewritten: {rewritten}")

    # Step 2 — Retrieve
    print("\n[2] Retrieving documents...")
    documents = retrieve(rewritten)
    print(f"    Retrieved {len(documents)} document(s)")

    # Step 3 — Generate with IVR faithfulness validation
    print("\n[3] Generating answer with hallucination repair loop...")

    faithfulness_req = Requirement(
        "The response must be entirely faithful to the provided documents. "
        "Do not include any information not present in the documents.",
        validation_fn=make_faithfulness_validator(documents, context, backend),
    )

    session = MelleaSession(backend, context)
    result = session.instruct(
        "Using only the provided documents, answer the question: `{{question}}`",
        user_variables={"question": rewritten},
        grounding_context={f"doc{i}": doc.text for i, doc in enumerate(documents)},
        requirements=[faithfulness_req],
        strategy=RepairTemplateStrategy(loop_budget=3),
        return_sampling_results=True,
    )

    # Show the IVR loop details
    print(f"\n    Attempts: {len(result.sample_generations)}")
    print(f"    Success:  {result.success}")

    for i, (gen, validations) in enumerate(
        zip(result.sample_generations, result.sample_validations), 1
    ):
        print(f"\n    --- Attempt {i} ---")
        failed = [v for _, v in validations if not v.as_bool()]
        if failed:
            for val in failed:
                if val.reason:
                    print(f"    FAILED: {val.reason[:120]}...")
        else:
            print("    PASSED: all sentences faithful")

    print(f"\n>> Final Answer:\n   {result.value}")


if __name__ == "__main__":
    run("What is ClapNQ and how does it relate to NLP benchmarks?")
