# pytest: skip_always
"""Rewrite-Retrieve-Generate (RRG) — the simplest intrinsic-enhanced RAG pattern.

Pipeline
--------
    question  ──►  rewrite_question()  ──►  retrieve(rewritten)  ──►  session.instruct()
                                                                       with grounding_context

Langflow equivalent
-------------------
    ChatInput → QueryRewrite → ElserRetriever → GraniteBaseModel → ChatOutput

When to use
-----------
Use RRG as the building block for all RAG pipelines. Query rewriting makes the
retrieval step more effective by resolving coreferences and rephrasing questions
into self-contained queries suitable for a search engine.

Run
---
    uv run python docs/examples/rag_patterns/101_rewrite_retrieve_generate.py
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
# Mock document corpus (replace with a real retriever in production)
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


def run(question: str) -> None:
    context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))

    print("Loading model...")
    backend = LocalHFBackend(model_id=MODEL_ID)

    # Step 1 — Rewrite the question for retrieval
    print("\n[1] Rewriting question...")
    rewritten = str(rag.rewrite_question(question, context, backend))
    print(f"    Original : {question}")
    print(f"    Rewritten: {rewritten}")

    # Step 2 — Retrieve documents
    print("\n[2] Retrieving documents...")
    documents = retrieve(rewritten)
    print(f"    Retrieved {len(documents)} document(s)")

    # Step 3 — Generate a grounded answer
    print("\n[3] Generating answer...")
    session = MelleaSession(backend, context)
    answer = session.instruct(
        "Using only the provided documents, answer the question: `{{question}}`",
        user_variables={"question": rewritten},
        grounding_context={f"doc{i}": doc.text for i, doc in enumerate(documents)},
    )

    print(f"\n>> Answer:\n   {answer.value}")


if __name__ == "__main__":
    run("What is ClapNQ and how does it relate to NLP benchmarks?")
