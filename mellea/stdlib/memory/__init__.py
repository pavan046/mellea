"""Standard-library memory components built on ``mellea.core.memory``.

Re-exports the core memory contracts alongside the default concrete stores and
policies shipped with mellea: ``InMemoryStore`` (reference implementation for
tests and small scripts), ``DefaultRetrievalPolicy`` (top-k union across
stores with trailing splice), and ``DefaultCompactionPolicy`` (summarize the
oldest slice when the window budget is exceeded).

Start here when you want memory without picking a vector DB or graph backend.
"""

from __future__ import annotations

from ...core.memory import CompactionScope, MemoryRecord, MemoryStore
from .in_memory_store import InMemoryStore
from .policies import (
    CompactionPolicy,
    DefaultCompactionPolicy,
    DefaultRetrievalPolicy,
    RetrievalPolicy,
)

__all__ = [
    "CompactionPolicy",
    "CompactionScope",
    "DefaultCompactionPolicy",
    "DefaultRetrievalPolicy",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "RetrievalPolicy",
]
