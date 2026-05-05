"""First-class memory abstractions for mellea.

Defines the contracts every memory store must satisfy: ``MemoryRecord`` (the
stored unit, shaped after the Zettelkasten note in Xu et al. arXiv:2502.12110
§3.1), ``MemoryStore`` (abstract write/read/inhibit backend), and the auxiliary
``CompactionScope`` descriptor. Concrete stores (in-memory, vector, episodic,
graph) live elsewhere and inherit the verbs declared here.

The contracts here honour the write/read/inhibit-update causal chain of Zhang
et al. arXiv:2509.18868: every record has a stable id, every write is
auditable, and every retrieval produces ``MemoryBlock`` instances whose
provenance points back to the originating record.
"""

from __future__ import annotations

import abc
import datetime
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import ContextTurn, MemoryBlock


@dataclass
class MemoryRecord:
    """The stored unit inside a ``MemoryStore``.

    Shape follows A-Mem's Zettelkasten note (Xu et al. arXiv:2502.12110 §3.1
    eq. 1): raw content plus LLM-derived semantic components and links. Storing
    these fields explicitly (rather than a flat ``(id, text, vec)`` row) is
    what makes memory evolution tractable — the evolution pass rewrites
    ``keywords``, ``tags``, and ``contextual_description`` in place.

    Args:
        record_id (str): Stable identifier. Persistent across rewrites; changes
            only via ``supersede``.
        content (str): ``c_i`` — the original interaction content.
        timestamp (datetime.datetime): ``t_i`` — wall-clock time of the write.
        keywords (list[str]): ``K_i`` — LLM-extracted key concepts.
        tags (list[str]): ``G_i`` — LLM-assigned categorization.
        contextual_description (str): ``X_i`` — LLM-generated semantic gloss.
        embedding (list[float]): ``e_i`` — dense vector over ``concat(c, K, G, X)``.
            Stored as a list for serialization; stores may cast to numpy as needed.
        links (list[str]): ``L_i`` — record ids of linked notes.
        actor (str): Who wrote this record (user id, agent name, "system").
        meta (dict[str, Any]): Free-form metadata.
        superseded_by (str | None): If set, points to the record that replaced this
            one. Treated as a tombstone by retrieval.
    """

    record_id: str
    content: str
    timestamp: datetime.datetime
    actor: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    contextual_description: str = ""
    embedding: list[float] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    superseded_by: str | None = None


@dataclass
class CompactionScope:
    """Descriptor passed to ``MemoryStore.compact`` to bound a rollup pass.

    Compaction clusters related records and emits consolidated records that
    link back to their sources (§3.5 of the proposal). The scope controls how
    clusters are formed and what slice of the store is eligible.

    Args:
        by (Literal["entity", "time_bucket", "tag", "actor"]): Clustering dimension.
        threshold (int): Minimum cluster size before a consolidated record is emitted.
        max_age (datetime.timedelta | None): Only records older than this are eligible.
            ``None`` means no lower-bound age filter.
    """

    by: Literal["entity", "time_bucket", "tag", "actor"]
    threshold: int = 3
    max_age: datetime.timedelta | None = None


class MemoryStore(abc.ABC):
    """Abstract backend for persistent, addressable memory.

    Implementations honour the write → read → inhibit/update contract from
    Zhang et al. arXiv:2509.18868 §3 and the A-Mem three-phase write from Xu
    et al. arXiv:2502.12110 §3.1-§3.3. The ABC itself is carrier-agnostic:
    vector, episodic, and graph stores all implement the same verbs.

    Args:
        store_id (str): Stable identifier for this store instance. Embedded in
            every ``MemoryBlock`` emitted by ``retrieve`` so downstream
            consumers can distinguish stores.
        write_on_turn (bool): If ``True``, ``MelleaSession`` will call
            ``write()`` after every completed turn. Defaults to ``False``;
            callers with custom ingestion flows can leave this off.
    """

    def __init__(self, store_id: str, *, write_on_turn: bool = False):
        """Initialize MemoryStore with a stable id and optional auto-write hook."""
        self._store_id = store_id
        self._write_on_turn = write_on_turn

    @property
    def store_id(self) -> str:
        """Stable identifier embedded in every emitted ``MemoryBlock``."""
        return self._store_id

    @property
    def write_on_turn(self) -> bool:
        """Whether ``MelleaSession`` should auto-write after each completed turn."""
        return self._write_on_turn

    @abc.abstractmethod
    def write(
        self, turn: ContextTurn, *, actor: str, tags: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        """Ingest a turn and return the records produced.

        Implementations may produce zero, one, or many records per turn
        (e.g. an episodic store may extract multiple facts). The A-Mem
        three-phase write runs here:

        1. note construction: ``K, G, X ← LLM(c ‖ t ‖ P_s1)``  [A-Mem eq. 2]
        2. link generation:   ``L      ← LLM(m_n ‖ M_near ‖ P_s2)`` [A-Mem eq. 6]
        3. memory evolution:  for ``m_j ∈ M_near``, ``m_j* ← LLM(...)`` [A-Mem eq. 7]

        Phase 3 is sync-on-write by default; stores MAY expose an async mode,
        but retrieval consistency is the sync contract.

        Args:
            turn (ContextTurn): The completed model input/output pair.
            actor (str): Who produced this turn (user id, agent name).
            tags (dict[str, Any] | None): Optional tagging metadata passed through
                to the resulting records.

        Returns:
            list[MemoryRecord]: The records materialized during this write.
        """
        ...

    @abc.abstractmethod
    def retrieve(
        self, query: str, *, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[MemoryBlock]:
        """Retrieve the top-k records most relevant to ``query``.

        Emits ``MemoryBlock`` instances ready to splice into a context
        projection. Superseded records MUST be filtered out of the result.

        Args:
            query (str): The retrieval query (typically the current turn).
            k (int): Maximum number of blocks to return.
            filters (dict[str, Any] | None): Optional carrier-specific filters
                (e.g. actor, tag, time range).

        Returns:
            list[MemoryBlock]: Ordered by descending relevance score.
        """
        ...

    @abc.abstractmethod
    def forget(self, record_id: str, *, reason: str) -> None:
        """Mark a record as forgotten.

        The paper's inhibit verb. Implementations MUST persist enough state
        that a subsequent ``retrieve`` does not surface the record, and SHOULD
        emit an audit entry capturing ``reason``.

        Args:
            record_id (str): The record to inhibit.
            reason (str): Human-readable justification, recorded for audit.
        """
        ...

    @abc.abstractmethod
    def supersede(self, record_id: str, new_value: str) -> MemoryRecord:
        """Replace a record's content in place, preserving provenance.

        The old record is tombstoned via ``superseded_by``; the new record gets
        a fresh id. Callers that want to erase the old content entirely should
        follow up with ``forget``.

        Args:
            record_id (str): The record being replaced.
            new_value (str): The new content for the replacement record.

        Returns:
            MemoryRecord: The newly created replacement record.
        """
        ...

    @abc.abstractmethod
    def rewrite(self, record_id: str) -> MemoryRecord:
        """Re-run the A-Mem evolution pass against the record's current neighbors.

        Internal to ``write`` by default; exposed publicly so prompt changes or
        schema migrations can trigger a targeted re-evolution without a full
        re-ingest.

        Args:
            record_id (str): The record to re-evolve.

        Returns:
            MemoryRecord: The updated record (same ``record_id``).
        """
        ...

    @abc.abstractmethod
    def compact(self, scope: CompactionScope) -> list[MemoryRecord]:
        """Cluster and consolidate records matching ``scope``.

        Compaction produces new records that subsume many originals; the
        sources are NOT deleted, only demoted in retrieval. Consolidated
        records link back to sources via ``links`` so the audit trail stays
        intact and ``forget`` on a source is still meaningful.

        Args:
            scope (CompactionScope): Clustering dimension and thresholds.

        Returns:
            list[MemoryRecord]: The consolidated records that were emitted.
        """
        ...

    @abc.abstractmethod
    def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch a single record by id. Returns ``None`` if absent or forgotten."""
        ...


__all__ = ["CompactionScope", "MemoryRecord", "MemoryStore"]
