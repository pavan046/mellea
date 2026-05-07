"""Standard-library memory components built on ``mellea.core.memory``.

Re-exports the core memory contracts alongside the default concrete stores,
policies, and RAG-as-memory plumbing shipped with mellea:

* ``InMemoryStore`` — dict-backed reference store (tests, small scripts).
* ``VectorStore`` — dense-vector external memory for RAG.
* ``EpisodicStore`` — A-Mem-style note network with SQLite persistence.
* ``chunk_text`` / ``Chunk`` — sentence-aware sliding-window chunker.
* ``HashingEmbedder`` / ``SentenceTransformersEmbedder`` — embedding backends.
* ``DefaultRetrievalPolicy`` / ``DefaultCompactionPolicy`` — baseline policies.
* ``memory_block_to_document`` / ``memory_blocks_to_documents`` — citation glue
  into the existing RAG intrinsics.
* ``llm_note_enricher`` / ``llm_link_generator`` / ``llm_evolver`` /
  ``llm_summarizer`` — A-Mem phase callables for use with ``EpisodicStore``.

Start here when you want memory without picking a vector DB or graph backend.
"""

from __future__ import annotations

from ...core.memory import CompactionScope, MemoryRecord, MemoryStore
from .chunker import Chunk, chunk_text
from .citations import memory_block_to_document, memory_blocks_to_documents
from .embedders import (
    Embedder,
    HashingEmbedder,
    SentenceTransformersEmbedder,
    cosine_similarity,
)
from .episodic import (
    EpisodicRecord,
    EpisodicStore,
    llm_evolver,
    llm_link_generator,
    llm_note_enricher,
    llm_summarizer,
)
from .in_memory_store import InMemoryStore
from .policies import (
    CompactionPolicy,
    DefaultCompactionPolicy,
    DefaultRetrievalPolicy,
    RetrievalPolicy,
    Summarizer,
)
from .vector_store import VectorStore

__all__ = [
    "Chunk",
    "CompactionPolicy",
    "CompactionScope",
    "DefaultCompactionPolicy",
    "DefaultRetrievalPolicy",
    "Embedder",
    "EpisodicRecord",
    "EpisodicStore",
    "HashingEmbedder",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "RetrievalPolicy",
    "SentenceTransformersEmbedder",
    "Summarizer",
    "VectorStore",
    "chunk_text",
    "cosine_similarity",
    "llm_evolver",
    "llm_link_generator",
    "llm_note_enricher",
    "llm_summarizer",
    "memory_block_to_document",
    "memory_blocks_to_documents",
]
