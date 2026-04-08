"""Query Clarification (post-retriever) RAG — Mellea equivalent of the Langflow ClapNQ flow.

Langflow node → Mellea equivalent
----------------------------------
ChatInput          → function argument
Memory             → ChatContext threaded through each call
QueryRewrite       → rag.rewrite_question()
ElserRetriever     → retrieve()  (mocked with a fixed document list)
QueryClarification → rag.clarify_query()  returns "CLEAR" or a clarifying question
ConditionalRouter  → plain Python if/else
GraniteBaseModel   → MelleaSession.instruct()
ChatOutput         → print()

Run:
    uv run python scratchpad/query_clarification_rag.py
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

from mellea.backends.huggingface import LocalHFBackend  # noqa: E402
from mellea.stdlib.components import Document, Message  # noqa: E402
from mellea.stdlib.components.intrinsic import rag  # noqa: E402
from mellea.stdlib.context import ChatContext  # noqa: E402
from mellea.stdlib.session import MelleaSession  # noqa: E402

# ---------------------------------------------------------------------------
# Mock document corpus (replaces the ELSER / Elasticsearch retriever)
# ---------------------------------------------------------------------------
_CORPUS = [
    Document(
        "ClapNQ is a long-form question answering benchmark derived from Natural Questions. "
        "It focuses on questions whose answers require full Wikipedia passages rather than short spans."
    ),
    Document(
        "Natural Questions (NQ) is a dataset released by Google consisting of real queries "
        "issued to the Google search engine, paired with Wikipedia pages that contain the answer."
    ),
    Document(
        "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval step "
        "with a generative language model so the model can ground its answers in retrieved documents."
    ),
    Document(
        "ELSER (Elastic Learned Sparse EncodeR) is a sparse retrieval model from Elastic "
        "used for semantic search without requiring dense vector embeddings."
    ),
]


def retrieve(query: str) -> list[Document]:
    """Mock retriever — returns the full corpus regardless of query.

    In production this would be replaced by an ELSER / dense vector search
    against an Elasticsearch index.
    """
    return _CORPUS


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(question: str) -> None:
    # Conversation history — replaces the Memory node
    context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))

    # Load model once; the same backend instance is reused for intrinsics
    # AND for final answer generation so weights are not loaded twice.
    print("Loading model...")
    backend = LocalHFBackend(model_id=MODEL_ID)

    # Step 1 — QueryRewrite node
    print("\n[1] Rewriting question for retrieval...")
    rewritten = rag.rewrite_question(question, context, backend)
    print(f"    Original : {question}")
    print(f"    Rewritten: {rewritten}")

    # Step 2 — ElserRetriever node (mocked)
    print("\n[2] Retrieving documents...")
    documents = retrieve(rewritten)
    print(f"    Retrieved {len(documents)} document(s)")

    # Step 3 — QueryClarification node
    print("\n[3] Checking whether clarification is needed...")
    clarification = rag.clarify_query(question, documents, context, backend)
    print(f"    Result: {clarification!r}")

    # Step 4 — ConditionalRouter node
    if clarification != "CLEAR":
        # Route to ChatOutput (clarification branch)
        print(f"\n>> Clarification needed:\n   {clarification}")
    else:
        # Route to GraniteBaseModel → ChatOutput (answer branch)
        print("\n[4] Generating final answer...")
        session = MelleaSession(backend, context)
        answer = session.instruct(
            "Using only the provided documents, answer the question: `{{question}}`",
            user_variables={"question": rewritten},
            grounding_context={f"doc{i}": doc.text for i, doc in enumerate(documents)},
        )
        print(f"\n>> Answer:\n   {answer.value}")


if __name__ == "__main__":
    run("What is ClapNQ and how does it relate to NLP benchmarks?")
