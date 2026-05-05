"""Bridge between memory retrieval and the RAG citation intrinsics.

The RAG intrinsics in ``mellea.stdlib.components.intrinsic.rag``
(``find_citations``, ``flag_hallucinated_content``, ``check_context_relevance``)
expect ``Document`` components keyed by ``doc_id``. Memory retrieval emits
``MemoryBlock`` instances. The two shapes are near-identical — this module
does the trivial translation and spells out the contract:

* The ``Document.doc_id`` is set to the block's ``record_id``. Every
  citation the intrinsic returns can therefore be traced back to a specific
  ``MemoryStore`` record — calling ``store.get(doc_id)`` resolves it. This
  is also what makes ``store.forget(doc_id)`` remove a cited chunk later.
* ``Document.title`` is taken from ``meta["doc_title"]`` when present, so
  multi-chunk documents render with their original title.
* The block's ``meta["doc_id"]`` (the *source* document id, pre-chunking)
  is preserved on the returned ``Document.title`` prefix so
  ``find_citations`` reports readable source ids even when the record id is
  a UUID.

Keeping the translation narrow and explicit makes the attribution chain
auditable end-to-end: ``MemoryStore record_id → Document.doc_id →
citation.citation_doc_id`` is a single equality check, not a heuristic.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...core.base import MemoryBlock, MemorySource
from ..components.docs.document import Document


def memory_block_to_document(block: MemoryBlock) -> Document:
    """Convert a single ``MemoryBlock`` into a ``Document``.

    ``doc_id`` is the block's ``record_id`` so callers can pass the block
    directly into the citation intrinsics and resolve any returned
    ``citation_doc_id`` against the originating store.

    Args:
        block (MemoryBlock): The block to convert. Source tag is preserved
            in the returned ``Document.title`` prefix for readability.

    Returns:
        Document: A ``Document`` whose text, title, and doc_id trace back to
        the memory record.
    """
    source_doc_id = block._meta.get("doc_id")
    chunk_idx = block._meta.get("chunk_index")
    stored_title = block._meta.get("doc_title")

    # Build a readable title: prefer the stored title, fall back to source
    # doc id + chunk index so the model and the citation intrinsic can both
    # show something interpretable.
    title_parts: list[str] = []
    if stored_title:
        title_parts.append(str(stored_title))
    if source_doc_id:
        title_parts.append(f"source={source_doc_id}")
    if chunk_idx is not None:
        title_parts.append(f"chunk={chunk_idx}")
    title = " | ".join(title_parts) if title_parts else None

    return Document(text=block.value or "", title=title, doc_id=block.record_id)


def memory_blocks_to_documents(
    blocks: Iterable[MemoryBlock], *, only_external: bool = True
) -> list[Document]:
    """Convert an iterable of ``MemoryBlock`` into ``Document`` instances.

    Args:
        blocks (Iterable[MemoryBlock]): Retrieved memory blocks.
        only_external (bool): If ``True`` (default), skip blocks whose
            ``source`` is not ``EXTERNAL_DOC``. Citation intrinsics are
            designed around document-grounded RAG and give confusing
            results when fed episodic/summary blocks. Set to ``False`` if
            you really want to pass everything through.

    Returns:
        list[Document]: One ``Document`` per eligible block, order preserved.
    """
    out: list[Document] = []
    for block in blocks:
        if only_external and block.source is not MemorySource.EXTERNAL_DOC:
            continue
        out.append(memory_block_to_document(block))
    return out
