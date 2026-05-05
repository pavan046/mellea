"""Sentence-aware sliding-window chunker used by ``VectorStore`` ingestion.

Splits raw document text into overlapping chunks that respect sentence
boundaries when possible. Emits chunks with character offsets into the source
document so downstream citation intrinsics (``find_citations``,
``flag_hallucinated_content``) can map response spans back to exact document
spans.

No external dependencies. The sentence splitter is a regex-based heuristic
(``.!?`` followed by whitespace and a capital letter); swap it for ``nltk`` or
``spacy`` at call-time via the ``sentence_splitter`` parameter if you need
something smarter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """A chunked span of a source document.

    Args:
        text (str): The chunk content.
        start (int): Character offset of the chunk's first character in the
            source document (inclusive).
        end (int): Character offset of the chunk's last character plus one
            (exclusive, half-open interval matching Python slicing).
        index (int): Zero-based position of this chunk within its source.
    """

    text: str
    start: int
    end: int
    index: int


def _default_sentence_splitter(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets for each sentence in ``text``.

    Offsets cover the entire document (no gaps) so the chunker can always
    fall back to character offsets when sentence alignment fails.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        spans.append((cursor, end))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return spans


def _token_count(text: str) -> int:
    """Whitespace-delimited token count.

    Intentionally tokenizer-agnostic — the chunker's job is to bound chunks
    in a way that survives model-tokenizer choice. For strict token budgets
    callers can pre-tokenize and set ``window_size`` conservatively.
    """
    if not text.strip():
        return 0
    return len(_WHITESPACE.split(text.strip()))


def chunk_text(
    text: str,
    *,
    window_size: int = 200,
    overlap: int = 40,
    sentence_splitter: Callable[[str], list[tuple[int, int]]] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping, sentence-aligned chunks.

    The chunker greedily packs whole sentences into a chunk until adding the
    next sentence would exceed ``window_size`` whitespace tokens. Chunk
    boundaries are then pulled back by ``overlap`` tokens so consecutive
    chunks share tail/head context — this prevents a fact that straddles a
    boundary from becoming invisible to retrieval.

    Args:
        text (str): The source document content.
        window_size (int): Target chunk size in whitespace-delimited tokens.
            Defaults to 200.
        overlap (int): Token overlap between consecutive chunks. Must be
            strictly less than ``window_size``. Defaults to 40.
        sentence_splitter: Optional callable returning ``(start, end)``
            offsets for each sentence. Defaults to a regex heuristic.

    Returns:
        list[Chunk]: One or more chunks in document order. Empty input yields
        an empty list.

    Raises:
        ValueError: If ``overlap`` is not strictly less than ``window_size``.
    """
    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than window_size ({window_size})"
        )
    text = text or ""
    if not text.strip():
        return []

    splitter = sentence_splitter or _default_sentence_splitter
    sentence_spans = splitter(text)
    if not sentence_spans:
        return [Chunk(text=text, start=0, end=len(text), index=0)]

    chunks: list[Chunk] = []
    i = 0
    while i < len(sentence_spans):
        chunk_start = sentence_spans[i][0]
        chunk_end = sentence_spans[i][1]
        running_tokens = _token_count(text[chunk_start:chunk_end])
        j = i + 1
        while j < len(sentence_spans):
            s_start, s_end = sentence_spans[j]
            next_tokens = _token_count(text[s_start:s_end])
            if running_tokens + next_tokens > window_size and running_tokens > 0:
                break
            chunk_end = s_end
            running_tokens += next_tokens
            j += 1

        chunks.append(
            Chunk(
                text=text[chunk_start:chunk_end].strip(),
                start=chunk_start,
                end=chunk_end,
                index=len(chunks),
            )
        )

        if j >= len(sentence_spans):
            break

        # Slide forward, pulling back by overlap tokens.
        i = _next_index_with_overlap(sentence_spans, i, j, overlap, text)
        if i <= 0 or i >= len(sentence_spans):
            break

    return chunks


def _next_index_with_overlap(
    sentence_spans: list[tuple[int, int]],
    prev_start_idx: int,
    next_start_idx: int,
    overlap: int,
    text: str,
) -> int:
    """Walk backwards from ``next_start_idx`` until ``overlap`` tokens are covered."""
    if overlap <= 0:
        return next_start_idx
    k = next_start_idx - 1
    accumulated = 0
    while k > prev_start_idx:
        s_start, s_end = sentence_spans[k]
        accumulated += _token_count(text[s_start:s_end])
        if accumulated >= overlap:
            return k
        k -= 1
    return max(prev_start_idx + 1, next_start_idx)
