"""Retrieval and compaction policies consulted by ``MemoryContext``.

Two orthogonal contracts live here:

* ``RetrievalPolicy`` chooses *which* memory to recall and *where* in the
  generation view to splice it. Splicing position matters — Zhang et al.
  arXiv:2509.18868 §4.3 observes a mid-sequence drop in recall and attribution.
* ``CompactionPolicy`` decides when the projected view has grown past budget
  and how to fold older turns into a ``SUMMARY`` ``MemoryBlock`` without
  losing audit links.

Default implementations are deliberately simple so they can be subclassed or
replaced wholesale; workstreams B/C/D will ship richer policies (hybrid
retrieval, entity-aware compaction).
"""

from __future__ import annotations

import abc
import datetime

from ...core.base import CBlock, Component, MemoryBlock, MemorySource, ModelOutputThunk
from ...core.memory import MemoryStore


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
    """Summarize the oldest 25% of the history once the budget is exceeded.

    Emits a ``MemoryBlock(source=SUMMARY)`` whose value is a plain-text
    concatenation of the compacted entries. ``record_id`` captures the ids of
    the folded blocks so ``forget`` on a source still has a handle.

    Args:
        store_id (str): ``store_id`` to stamp on emitted summary blocks.
        fold_fraction (float): Fraction of oldest history to compact.
            Defaults to 0.25.
    """

    def __init__(self, *, store_id: str = "compaction", fold_fraction: float = 0.25):
        """Initialize DefaultCompactionPolicy with a summary store id and fold ratio."""
        assert 0.0 < fold_fraction < 1.0, "fold_fraction must be in (0, 1)"
        self._store_id = store_id
        self._fold_fraction = fold_fraction

    def should_compact(self, history: list[Component | CBlock], budget: int) -> bool:
        """Trigger compaction once the history grows past ``budget``."""
        return len(history) > budget

    def compact(self, history: list[Component | CBlock]) -> list[Component | CBlock]:
        """Fold the oldest slice into a single ``SUMMARY`` block."""
        fold_count = max(1, int(len(history) * self._fold_fraction))
        to_fold = history[:fold_count]
        tail = history[fold_count:]

        folded_text = "\n".join(self._stringify(item) for item in to_fold)
        folded_ids: list[str] = []
        for item in to_fold:
            if isinstance(item, MemoryBlock):
                folded_ids.append(item.record_id)

        summary = MemoryBlock(
            f"Summary of prior turns:\n{folded_text}",
            source=MemorySource.SUMMARY,
            store_id=self._store_id,
            record_id=f"summary:{hash(folded_text) & 0xFFFFFFFF:08x}",
            retrieved_at=datetime.datetime.now(datetime.UTC),
            meta={"folded_record_ids": folded_ids, "folded_count": fold_count},
        )
        return [summary, *tail]

    @staticmethod
    def _stringify(item: Component | CBlock) -> str:
        if isinstance(item, CBlock):
            return item.value or ""
        return str(item)
