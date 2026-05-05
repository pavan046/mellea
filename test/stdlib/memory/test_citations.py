"""Tests for the MemoryBlock -> Document citation glue."""

from __future__ import annotations

import datetime

from mellea.core import MemoryBlock, MemorySource
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.memory.citations import (
    memory_block_to_document,
    memory_blocks_to_documents,
)


def _make_block(
    value: str = "passage",
    source: MemorySource = MemorySource.EXTERNAL_DOC,
    *,
    doc_id: str | None = "team",
    title: str | None = "Team",
    chunk_index: int | None = 0,
) -> MemoryBlock:
    meta: dict = {}
    if doc_id is not None:
        meta["doc_id"] = doc_id
    if title is not None:
        meta["doc_title"] = title
    if chunk_index is not None:
        meta["chunk_index"] = chunk_index
    return MemoryBlock(
        value,
        source=source,
        store_id="vs",
        record_id="rec-42",
        retrieved_at=datetime.datetime.now(datetime.UTC),
        score=0.5,
        meta=meta,
    )


def test_single_block_conversion_sets_doc_id_to_record_id() -> None:
    block = _make_block()
    doc = memory_block_to_document(block)
    assert isinstance(doc, Document)
    assert doc.doc_id == block.record_id
    assert doc.text == block.value
    # Title packs title + source doc id + chunk index for readability.
    assert "Team" in doc.title
    assert "source=team" in doc.title
    assert "chunk=0" in doc.title


def test_conversion_without_title_falls_back_to_source_and_index() -> None:
    block = _make_block(title=None)
    doc = memory_block_to_document(block)
    assert "Team" not in (doc.title or "")
    assert "source=team" in doc.title
    assert "chunk=0" in doc.title


def test_conversion_with_no_meta_returns_none_title() -> None:
    block = _make_block(doc_id=None, title=None, chunk_index=None)
    doc = memory_block_to_document(block)
    assert doc.title is None
    assert doc.doc_id == block.record_id


def test_batch_conversion_default_filters_non_external_blocks() -> None:
    external = _make_block(value="doc chunk", source=MemorySource.EXTERNAL_DOC)
    episodic = _make_block(value="episodic fact", source=MemorySource.EPISODIC)
    summary = _make_block(value="summary", source=MemorySource.SUMMARY)

    docs = memory_blocks_to_documents([external, episodic, summary])
    assert len(docs) == 1
    assert docs[0].text == "doc chunk"


def test_batch_conversion_only_external_false_includes_all() -> None:
    external = _make_block(value="doc chunk", source=MemorySource.EXTERNAL_DOC)
    episodic = _make_block(value="episodic fact", source=MemorySource.EPISODIC)

    docs = memory_blocks_to_documents([external, episodic], only_external=False)
    assert [d.text for d in docs] == ["doc chunk", "episodic fact"]


def test_batch_conversion_preserves_order() -> None:
    blocks = [_make_block(value=f"chunk {i}", chunk_index=i) for i in range(4)]
    docs = memory_blocks_to_documents(blocks)
    assert [d.text for d in docs] == [b.value for b in blocks]
