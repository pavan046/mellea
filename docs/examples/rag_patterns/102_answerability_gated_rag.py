# pytest: skip_always
"""Retrieve-Verify-Generate (RVG) — Answerability-gated RAG.

Pipeline
--------
    question  ──►  rewrite  ──►  retrieve  ──►  check_answerability()
                                                   │
                                                   ├─ score >= threshold ──►  generate answer
                                                   └─ score <  threshold ──►  decline

Langflow equivalent
-------------------
    ChatInput → QueryRewrite → ElserRetriever → Answerability
                                                  → ConditionalRouter → [GraniteBaseModel | DeclineMessage]

When to use
-----------
Use RVG when you want to avoid generating answers from insufficient context.
The answerability intrinsic returns a float (0-1) representing how confident
the model is that the retrieved documents contain the answer.  The threshold
is a tunable parameter — higher thresholds are more conservative.

Run
---
    uv run python docs/examples/rag_patterns/102_answerability_gated_rag.py
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
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext
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
# Pipeline
# ---------------------------------------------------------------------------

ANSWERABILITY_THRESHOLD = 0.5


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

    # Step 3 — Answerability gate
    print("\n[3] Checking answerability...")
    score = rag.check_answerability(question, documents, context, backend)
    print(
        f"    Answerability score: {score:.4f}  (threshold: {ANSWERABILITY_THRESHOLD})"
    )

    # Step 4 — Conditional: generate or decline
    if score >= ANSWERABILITY_THRESHOLD:
        print("\n[4] Score above threshold — generating answer...")
        session = MelleaSession(backend, context)
        answer = session.instruct(
            "Using only the provided documents, answer the question: `{{question}}`",
            user_variables={"question": rewritten},
            grounding_context={f"doc{i}": doc.text for i, doc in enumerate(documents)},
        )
        print(f"\n>> Answer:\n   {answer.value}")
    else:
        print(
            "\n>> Declined: I don't have enough information in the retrieved "
            "documents to answer that question."
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Answerable question")
    print("=" * 60)
    run("What is ClapNQ and how does it relate to NLP benchmarks?")

    print("\n\n")
    print("=" * 60)
    print("Test 2: Unanswerable question (not in corpus)")
    print("=" * 60)
    run("What is the recipe for chocolate cake?")
