# pytest: skip_always
"""Generate-Cite-Verify (GCV) — Citation & Grounding Enrichment RAG.

Pipeline
--------
    question  ──►  rewrite  ──►  retrieve  ──►  generate answer
                                                       │
                                                       ├──►  find_citations()
                                                       └──►  flag_hallucinated_content()
                                                       │
                                                       ▼
                                                  enriched result
                                                  (answer + citations + faithfulness)

Langflow equivalent
-------------------
    ChatInput → Retriever → GraniteBaseModel → Citations
                                             → HallucinationDetection

When to use
-----------
Use GCV when you need to annotate a generated response with source citations
and faithfulness metadata for downstream consumers (UI highlighting, audit
logs, compliance).  Unlike GDR (Pattern 4), this pattern does NOT loop — it
enriches the output in a single pass.

Run
---
    uv run python docs/examples/rag_patterns/105_citation_grounding_rag.py
"""

import json
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
# Mock document corpus — each document has a doc_id for citation tracking
# ---------------------------------------------------------------------------
_CORPUS = [
    Document(
        "ClapNQ is a long-form question answering benchmark derived from Natural "
        "Questions. It focuses on questions whose answers require full Wikipedia "
        "passages rather than short spans.",
        doc_id="clapnq-overview",
    ),
    Document(
        "Natural Questions (NQ) is a dataset released by Google consisting of real "
        "queries issued to the Google search engine, paired with Wikipedia pages "
        "that contain the answer.",
        doc_id="nq-overview",
    ),
    Document(
        "Retrieval-Augmented Generation (RAG) is a technique that combines a "
        "retrieval step with a generative language model so the model can ground "
        "its answers in retrieved documents.",
        doc_id="rag-definition",
    ),
    Document(
        "ELSER (Elastic Learned Sparse EncodeR) is a sparse retrieval model from "
        "Elastic used for semantic search without requiring dense vector embeddings.",
        doc_id="elser-overview",
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

    # Step 1 — Rewrite
    print(f"\n[1] Rewriting question: {question}")
    rewritten = str(rag.rewrite_question(question, context, backend))
    print(f"    Rewritten: {rewritten}")

    # Step 2 — Retrieve
    print("\n[2] Retrieving documents...")
    documents = retrieve(rewritten)
    print(f"    Retrieved {len(documents)} document(s)")

    # Step 3 — Generate answer
    print("\n[3] Generating answer...")
    session = MelleaSession(backend, context)
    answer = session.instruct(
        "Using only the provided documents, answer the question: `{{question}}`",
        user_variables={"question": rewritten},
        grounding_context={f"doc{i}": doc.text for i, doc in enumerate(documents)},
    )
    print(f"    Answer: {answer.value}")

    # Step 4 — Enrich with citations
    print("\n[4] Extracting citations...")
    citations = rag.find_citations(answer.value, documents, context, backend)

    # Step 5 — Enrich with hallucination flags
    print("\n[5] Checking faithfulness...")
    faithfulness = rag.flag_hallucinated_content(
        answer.value, documents, context, backend
    )

    # Final enriched output
    enriched = {
        "answer": answer.value,
        "citations": citations,
        "faithfulness": faithfulness,
    }

    print("\n" + "=" * 60)
    print("ENRICHED RESULT")
    print("=" * 60)

    print(f"\n>> Answer:\n   {enriched['answer']}")

    print("\n>> Citations:")
    if isinstance(citations, list):
        for c in citations:
            print(
                f"   [{c.get('citation_doc_id', '?')}] "
                f"'{c.get('response_text', '')}' "
                f"← '{c.get('citation_text', '')}'"
            )
    else:
        print(f"   {citations}")

    print("\n>> Faithfulness:")
    if isinstance(faithfulness, list):
        for f in faithfulness:
            score = f.get("faithfulness_likelihood", "?")
            text = f.get("response_text", "")
            print(f"   {score:.2f} — '{text}'")
    else:
        print(f"   {faithfulness}")

    # The enriched dict can be serialized for downstream consumers
    print(f"\n>> JSON (for downstream):\n{json.dumps(enriched, indent=2)}")


if __name__ == "__main__":
    run("What is ClapNQ and how does it relate to NLP benchmarks?")
