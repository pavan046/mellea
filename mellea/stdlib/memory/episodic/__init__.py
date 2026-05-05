"""Episodic memory: A-Mem-style note network with cross-session persistence.

Re-exports the public surface: ``EpisodicStore`` (the SQLite-backed store with
sync-on-write A-Mem three-phase writes), the phase callables
(``llm_note_enricher``, ``llm_link_generator``, ``llm_evolver``) for swapping
in your own prompts, and the LLM-backed summarizer used by compaction.
"""

from __future__ import annotations

from .amem import llm_evolver, llm_link_generator, llm_note_enricher, llm_summarizer
from .schema import EpisodicRecord
from .store import EpisodicStore

__all__ = [
    "EpisodicRecord",
    "EpisodicStore",
    "llm_evolver",
    "llm_link_generator",
    "llm_note_enricher",
    "llm_summarizer",
]
