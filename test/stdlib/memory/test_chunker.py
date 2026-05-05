"""Tests for the sentence-aware chunker used by VectorStore ingestion."""

from __future__ import annotations

from itertools import pairwise

import pytest

from mellea.stdlib.memory.chunker import Chunk, chunk_text


def test_empty_and_whitespace_text_yield_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_document_produces_one_chunk() -> None:
    chunks = chunk_text("Hello world. This fits.", window_size=20, overlap=4)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].start == 0
    assert chunks[0].end == len("Hello world. This fits.")


def test_chunks_respect_window_size() -> None:
    doc = " ".join(f"Sentence {i} has five tokens." for i in range(20))
    chunks = chunk_text(doc, window_size=30, overlap=6)
    for c in chunks:
        # Token count roughly bounded by window (allow slop for the last chunk).
        assert len(c.text.split()) <= 35


def test_chunks_overlap_by_roughly_requested_tokens() -> None:
    doc = " ".join(f"Sentence {i} exists." for i in range(40))
    chunks = chunk_text(doc, window_size=20, overlap=6)
    assert len(chunks) >= 2
    # Adjacent chunks share content
    for a, b in pairwise(chunks):
        assert b.start < a.end, f"chunk {b.index} starts after {a.index} ended"


def test_overlap_must_be_less_than_window() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", window_size=10, overlap=10)


def test_chunks_are_returned_in_document_order_with_monotonic_offsets() -> None:
    doc = " ".join(f"Sentence {i} exists." for i in range(50))
    chunks = chunk_text(doc, window_size=30, overlap=8)
    for a, b in pairwise(chunks):
        assert a.index < b.index
        assert a.start <= b.start


def test_chunk_offsets_map_back_into_source() -> None:
    doc = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(doc, window_size=4, overlap=1)
    for c in chunks:
        # The chunk's text should appear in the source at the declared offset range.
        assert doc[c.start : c.end].strip() == c.text


def test_custom_sentence_splitter_is_respected() -> None:
    def always_one_span(text: str) -> list[tuple[int, int]]:
        return [(0, len(text))]

    doc = "Anything goes here."
    chunks = chunk_text(
        doc, window_size=2, overlap=1, sentence_splitter=always_one_span
    )
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == doc
