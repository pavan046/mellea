"""Retrieval and compaction policies consulted by ``MemoryContext``.

Two orthogonal contracts live here:

* ``RetrievalPolicy`` chooses *which* memory to recall and *where* in the
  generation view to splice it. Splicing position matters — Zhang et al.
  arXiv:2509.18868 §4.3 observes a mid-sequence drop in recall and attribution.
* ``CompactionPolicy`` decides when the projected view has grown past budget
  and how to fold older turns into a ``SUMMARY`` ``MemoryBlock`` backed by a
  real ``MemoryRecord`` so ``forget``/audit chains still resolve.

Default implementations are deliberately simple so they can be subclassed or
replaced wholesale; workstreams B/C/D will ship richer policies (hybrid
retrieval, entity-aware compaction).
"""

from __future__ import annotations

import abc
import datetime
from collections.abc import Callable

from ...core.base import (
    CBlock,
    Component,
    ContextTurn,
    MemoryBlock,
    MemorySource,
    ModelOutputThunk,
)
from ...core.memory import MemoryStore

Summarizer = Callable[[str], str]
"""Callable that turns folded history text into a short summary string.

``DefaultCompactionPolicy`` wraps this callable so workstreams can plug in any
backend-backed summarizer (LLM call, extractive summarizer, etc.) without the
policy having to know about backends.
"""


class RetrievalPolicy(abc.ABC):
    """Chooses which records to recall and where to splice them.

    Called by ``MemoryContext.view_for_generation`` once per turn. The policy
    owns three decisions: query derivation, cross-store fusion, and positional
    splicing. Implementations MUST preserve ``MemoryBlock`` identity (no
    re-wrapping as plain ``CBlock``) so downstream attribution stays intact.
    """

    @abc.abstractmethod
    def derive_query(self, history: list[Component | CBlock]) -> str:
        """Compute the retrieval query from the current projected history.

        Args:
            history (list[Component | CBlock]): The context projection as it
                would be sent to the backend, before memory has been spliced.

        Returns:
            str: The retrieval query. Empty string means "do not retrieve".
        """
        ...

    @abc.abstractmethod
    def retrieve(self, query: str, stores: list[MemoryStore]) -> list[MemoryBlock]:
        """Fuse recall across the attached stores.

        Args:
            query (str): The query produced by ``derive_query``.
            stores (list[MemoryStore]): Stores attached to the ``MemoryContext``.

        Returns:
            list[MemoryBlock]: The fused, ranked recall. May be empty.
        """
        ...

    @abc.abstractmethod
    def splice(
        self, history: list[Component | CBlock], recalled: list[MemoryBlock]
    ) -> list[Component | CBlock]:
        """Insert the recalled blocks into the history at the right position.

        Args:
            history (list[Component | CBlock]): The pre-splice projection.
            recalled (list[Component | CBlock]): Recalled blocks from ``retrieve``.

        Returns:
            list[Component | CBlock]: The spliced projection ready for generation.
        """
        ...


class DefaultRetrievalPolicy(RetrievalPolicy):
    """Top-k union across stores, spliced immediately before the last user turn.

    - Derives the query from the last non-assistant block's string form.
    - Queries every attached store with the same ``k``, concatenates results,
      and returns them sorted by score (``None`` scores sink to the bottom).
    - Splices recalled blocks before the final element of the history so the
      user's latest message stays closest to the generation boundary.

    Args:
        k (int): Per-store retrieval cap. Defaults to 5.
        filters (dict | None): Forwarded to every ``retrieve`` call.
    """

    def __init__(self, *, k: int = 5, filters: dict | None = None):
        """Initialize DefaultRetrievalPolicy with a per-store recall cap."""
        self._k = k
        self._filters = filters

    def derive_query(self, history: list[Component | CBlock]) -> str:
        """Last non-assistant block's string form, falling back to the tail block."""
        for item in reversed(history):
            if isinstance(item, ModelOutputThunk):
                continue
            if isinstance(item, CBlock):
                return item.value or ""
            # Components: use str() which invokes parts()/template rendering.
            return str(item)
        return str(history[-1]) if history else ""

    def retrieve(self, query: str, stores: list[MemoryStore]) -> list[MemoryBlock]:
        """Query every store and sort the union by descending score."""
        if not query:
            return []
        hits: list[MemoryBlock] = []
        for store in stores:
            hits.extend(store.retrieve(query, k=self._k, filters=self._filters))
        hits.sort(
            key=lambda b: b.score if b.score is not None else float("-inf"),
            reverse=True,
        )
        return hits

    def splice(
        self, history: list[Component | CBlock], recalled: list[MemoryBlock]
    ) -> list[Component | CBlock]:
        """Insert recalled blocks just before the final history element."""
        if not recalled:
            return list(history)
        if len(history) == 0:
            return list(recalled)
        return [*history[:-1], *recalled, history[-1]]


class CompactionPolicy(abc.ABC):
    """Window-side compaction consulted when the view exceeds budget.

    Folds the oldest slice of a projection into one or more ``SUMMARY``
    ``MemoryBlock`` entries so the projection stays under budget without
    losing the ability to ``forget`` the underlying turns later.
    """

    @abc.abstractmethod
    def should_compact(self, history: list[Component | CBlock], budget: int) -> bool:
        """Whether the projection needs compaction right now.

        Args:
            history (list[Component | CBlock]): The post-splice projection.
            budget (int): Maximum allowed component count.

        Returns:
            bool: ``True`` if ``compact`` should be called.
        """
        ...

    @abc.abstractmethod
    def compact(self, history: list[Component | CBlock]) -> list[Component | CBlock]:
        """Replace the oldest slice with ``SUMMARY`` ``MemoryBlock`` entries."""
        ...


class DefaultCompactionPolicy(CompactionPolicy):
    """Summarize the oldest slice of the history into a real ``MemoryRecord``.

    On ``compact``, folds the oldest ``fold_fraction`` of the projection into
    plain text, passes that text to ``summarizer`` to produce a genuine
    summary, and writes the summary to ``store`` as a ``MemoryRecord``. The
    returned projection replaces the folded slice with a single
    ``MemoryBlock(source=SUMMARY)`` whose ``store_id`` / ``record_id`` resolve
    to that freshly written record — so ``forget(record_id)`` has a real
    handle and the audit chain stays intact (§3.5 of the proposal).

    The fold slice preserves ``folded_record_ids`` in the new record's ``meta``
    (via ``tags``) so callers can still reach the underlying sources.

    If the slice stringifies to empty text, or ``summarizer`` returns empty,
    or ``store.write`` produces no records, ``compact`` returns ``history``
    unchanged rather than emitting unresolvable provenance.

    Args:
        summarizer (Summarizer): Callable turning folded text into a summary.
            Wrap a backend here (e.g. via ``mellea.stdlib.memory.llm_summarizer``).
        store (MemoryStore): Store to write the summary record into. Must be
            attached to the owning ``MemoryContext`` so the emitted
            ``MemoryBlock``'s ``store_id`` resolves.
        fold_fraction (float): Fraction of oldest history to compact.
            Defaults to 0.25.

    Attributes:
        store_id (str): The ``store_id`` stamped on emitted summary blocks.
            Derived from ``store`` so it always matches an attached store.
    """

    def __init__(
        self,
        summarizer: Summarizer,
        store: MemoryStore,
        *,
        fold_fraction: float = 0.25,
    ):
        """Initialize DefaultCompactionPolicy with a summarizer and target store."""
        assert 0.0 < fold_fraction < 1.0, "fold_fraction must be in (0, 1)"
        self._summarizer = summarizer
        self._store = store
        self._fold_fraction = fold_fraction

    @property
    def store_id(self) -> str:
        """``store_id`` of the attached store; stamped on emitted summary blocks."""
        return self._store.store_id

    def should_compact(self, history: list[Component | CBlock], budget: int) -> bool:
        """Trigger compaction once the history grows past ``budget``."""
        return len(history) > budget

    def compact(self, history: list[Component | CBlock]) -> list[Component | CBlock]:
        """Summarize the oldest slice and splice a resolvable ``SUMMARY`` block."""
        fold_count = max(1, int(len(history) * self._fold_fraction))
        to_fold = history[:fold_count]
        tail = history[fold_count:]

        folded_text = "\n".join(self._stringify(item) for item in to_fold).strip()
        if not folded_text:
            return list(history)

        folded_ids: list[str] = [
            item.record_id for item in to_fold if isinstance(item, MemoryBlock)
        ]

        summary_text = self._summarizer(folded_text).strip()
        if not summary_text:
            return list(history)

        records = self._store.write(
            ContextTurn(CBlock(summary_text), None),
            actor="compaction",
            tags={
                "op": "compact",
                "folded_record_ids": folded_ids,
                "folded_count": fold_count,
            },
        )
        if not records:
            return list(history)

        record = records[0]
        summary = MemoryBlock(
            summary_text,
            source=MemorySource.SUMMARY,
            store_id=self._store.store_id,
            record_id=record.record_id,
            retrieved_at=datetime.datetime.now(datetime.UTC),
            meta={"folded_record_ids": folded_ids, "folded_count": fold_count},
        )
        return [summary, *tail]

    @staticmethod
    def _stringify(item: Component | CBlock) -> str:
        if isinstance(item, CBlock):
            return item.value or ""
        return str(item)
