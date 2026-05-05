"""Tests for VectorStore — RAG-as-memory concrete MemoryStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mellea.core import CBlock, CompactionScope, ContextTurn, MemorySource
from mellea.stdlib.memory import (
    HashingEmbedder,
    VectorStore,
    memory_blocks_to_documents,
)

DOC_TEXT = (
    "Alice is a machine learning engineer. She works on large language models. "
    "Bob is a data scientist. He specializes in computer vision and image classification. "
    "Carol maintains the infrastructure. She keeps the GPU cluster running."
)


def _new_store(**kwargs) -> VectorStore:
    return VectorStore(
        "test", embedder=HashingEmbedder(dim=256), window_size=15, overlap=3, **kwargs
    )


def test_add_document_chunks_and_indexes() -> None:
    vs = _new_store()
    records = vs.add_document(DOC_TEXT, doc_id="team", title="Team Directory")
    assert len(records) >= 2
    assert vs.doc_ids() == ["team"]
    assert len(vs.chunks_for("team")) == len(records)
    for rec in records:
        assert rec.meta["doc_id"] == "team"
        assert rec.meta["doc_title"] == "Team Directory"
        assert isinstance(rec.meta["chunk_index"], int)
        assert rec.meta["start"] < rec.meta["end"]


def test_add_document_empty_text_noop() -> None:
    vs = _new_store()
    assert vs.add_document("") == []
    assert vs.add_document("   \n  ") == []


def test_retrieve_returns_memory_blocks_with_external_source() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")

    hits = vs.retrieve("computer vision", k=3)
    assert hits
    for h in hits:
        assert h.source is MemorySource.EXTERNAL_DOC
        assert h.store_id == "test"
        assert h._meta.get("doc_id") == "team"
        assert h.score is not None


def test_retrieve_ranks_relevant_chunk_first() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    hits = vs.retrieve("computer vision", k=1)
    assert len(hits) == 1
    assert "computer vision" in hits[0].value.lower()


def test_retrieve_empty_query_returns_nothing() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    assert vs.retrieve("", k=5) == []


def test_retrieve_filters_by_doc_id() -> None:
    vs = _new_store()
    vs.add_document("First note about hiking.", doc_id="a")
    vs.add_document("Second note about hiking.", doc_id="b")
    hits = vs.retrieve("hiking", k=5, filters={"doc_id": "b"})
    assert hits
    assert all(h._meta["doc_id"] == "b" for h in hits)


def test_forget_removes_individual_chunk() -> None:
    vs = _new_store()
    records = vs.add_document(DOC_TEXT, doc_id="team")
    target = records[0]
    vs.forget(target.record_id, reason="noise")
    remaining = [r.record_id for r in vs.chunks_for("team")]
    assert target.record_id not in remaining
    assert any(a["op"] == "forget" for a in vs.audit_log)


def test_forget_document_drops_all_chunks() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    removed = vs.forget_document("team")
    assert removed > 0
    assert "team" not in vs.doc_ids()
    assert vs.retrieve("computer vision", k=5) == []


def test_supersede_replaces_content_and_reembeds() -> None:
    vs = _new_store()
    records = vs.add_document(DOC_TEXT, doc_id="team")
    original = records[0]
    new = vs.supersede(original.record_id, "Updated chunk about deep learning.")
    assert new.record_id != original.record_id
    assert vs.get(original.record_id) is None
    assert new.links == [original.record_id]
    hits = vs.retrieve("deep learning", k=3)
    assert any(h.record_id == new.record_id for h in hits)


def test_supersede_unknown_id_raises() -> None:
    vs = _new_store()
    with pytest.raises(KeyError):
        vs.supersede("nope", "replacement")


def test_rewrite_re_embeds_existing_content() -> None:
    vs = _new_store()
    records = vs.add_document("Original content.", doc_id="d")
    refreshed = vs.rewrite(records[0].record_id)
    assert refreshed.record_id == records[0].record_id
    assert any(a["op"] == "rewrite" for a in vs.audit_log)


def test_compact_is_noop() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    assert vs.compact(CompactionScope(by="actor")) == []
    assert any(
        a["op"] == "compact" and a.get("note") == "vector_store_noop"
        for a in vs.audit_log
    )


def test_save_and_load_roundtrip_preserves_scores() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    expected = vs.retrieve("computer vision", k=3)

    with tempfile.TemporaryDirectory() as td:
        vs.save(td)
        loaded = VectorStore.load(td, embedder=HashingEmbedder(dim=256))
        actual = loaded.retrieve("computer vision", k=3)

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual, strict=True):
        assert e.record_id == a.record_id
        assert e.score == pytest.approx(a.score, abs=1e-6)


def test_load_embedder_dim_mismatch_raises() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team")
    with tempfile.TemporaryDirectory() as td:
        vs.save(td)
        with pytest.raises(ValueError, match="embedder dim"):
            VectorStore.load(td, embedder=HashingEmbedder(dim=128))


def test_write_abc_path_round_trips_a_turn_as_document() -> None:
    vs = _new_store()
    turn = ContextTurn(CBlock("A recorded turn about deployment pipelines."), None)
    records = vs.write(turn, actor="user")
    assert records
    # doc_id generated from the turn, discoverable via retrieve().
    hits = vs.retrieve("deployment", k=3)
    assert hits


def test_memory_blocks_to_documents_preserves_record_ids() -> None:
    vs = _new_store()
    vs.add_document(DOC_TEXT, doc_id="team", title="Team")
    hits = vs.retrieve("computer vision", k=3)
    docs = memory_blocks_to_documents(hits)
    assert len(docs) == len(hits)
    for block, doc in zip(hits, docs, strict=True):
        assert doc.doc_id == block.record_id
        assert doc.text == block.value
        # Title encodes the source doc id + chunk index for readability.
        assert "source=team" in doc.title
        assert "chunk=" in doc.title
