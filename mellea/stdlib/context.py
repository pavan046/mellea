"""Concrete ``Context`` implementations for common conversation patterns.

Provides ``ChatContext``, which accumulates all turns in a sliding-window chat history
(configurable via ``window_size``), ``SimpleContext``, in which each interaction
is treated as a stateless single-turn exchange (no prior history is passed to the
model), and ``MemoryContext``, a memory-aware ``ChatContext`` subclass that splices
``MemoryBlock`` entries from attached ``MemoryStore`` instances on every
``view_for_generation`` call. Import ``ChatContext`` for multi-turn conversations,
``SimpleContext`` when you want each call to the model to be independent, and
``MemoryContext`` when you need cross-session memory or RAG-as-memory.
"""

from __future__ import annotations

# Leave unused `ContextTurn` import for import ergonomics.
from ..core import CBlock, Component, Context, ContextTurn
from ..core.memory import MemoryStore


class ChatContext(Context):
    """Initializes a chat context with unbounded window_size and is_chat=True by default.

    Args:
        window_size (int | None): Maximum number of context turns to include when
            calling ``view_for_generation``. ``None`` (the default) means the full
            history is always returned.
    """

    def __init__(self, *, window_size: int | None = None):
        """Initialize ChatContext with an optional sliding-window size."""
        super().__init__()
        self._window_size = window_size

    def add(self, c: Component | CBlock) -> ChatContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to append.

        Returns:
            ChatContext: A new ``ChatContext`` with the added entry, preserving the
            current ``window_size`` setting.
        """
        new = ChatContext.from_previous(self, c)
        new._window_size = self._window_size
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return the context entries to pass to the model, respecting the configured window.

        Uses the ``window_size`` set during initialisation to limit how many past
        turns are included.  ``None`` is returned when the underlying history is
        non-linear.

        Returns:
            list[Component | CBlock] | None: Ordered list of context entries up to
            ``window_size`` turns, or ``None`` if the history is non-linear.
        """
        return self.as_list(self._window_size)


class MemoryContext(ChatContext):
    """A ``ChatContext`` that consults ``MemoryStore`` instances on every view.

    On ``view_for_generation``, the context projects its linked-list history
    exactly like ``ChatContext`` (respecting ``window_size``), then hands that
    projection to a ``RetrievalPolicy`` which derives a query, fuses recall
    across the attached stores, and splices the resulting ``MemoryBlock``
    entries back in. If a ``CompactionPolicy`` is attached, it runs last and
    folds the oldest slice into a ``SUMMARY`` block when the projection
    exceeds ``turn_budget``.

    The stores are kept on ``self._stores`` and propagated across ``add``
    calls. They live *outside* the immutable linked list by design — stores
    outlive any single context instance.

    Args:
        stores (list[MemoryStore]): Stores to consult on each view. May be empty.
        retrieval_policy: Policy controlling query derivation, cross-store
            fusion, and splicing position. Defaults to ``DefaultRetrievalPolicy``.
        compaction_policy: Optional window-side compaction. Defaults to ``None``.
        turn_budget (int | None): Max components in the projected view before
            compaction is triggered. Ignored when ``compaction_policy`` is ``None``.
        window_size (int | None): Forwarded to ``ChatContext``.
    """

    def __init__(
        self,
        *,
        stores: list[MemoryStore] | None = None,
        retrieval_policy=None,  # type: ignore[assignment]
        compaction_policy=None,  # type: ignore[assignment]
        turn_budget: int | None = None,
        window_size: int | None = None,
    ):
        """Initialize MemoryContext with attached stores and policies."""
        super().__init__(window_size=window_size)
        from .memory.policies import DefaultRetrievalPolicy

        self._stores: list[MemoryStore] = list(stores or [])
        self._retrieval_policy = retrieval_policy or DefaultRetrievalPolicy()
        self._compaction_policy = compaction_policy
        self._turn_budget = turn_budget

    @property
    def stores(self) -> list[MemoryStore]:
        """The stores consulted on every ``view_for_generation`` call."""
        return list(self._stores)

    def add(self, c: Component | CBlock) -> MemoryContext:
        """Append ``c`` to the context, propagating stores and policies.

        Args:
            c (Component | CBlock): The component or content block to append.

        Returns:
            MemoryContext: A new ``MemoryContext`` sharing the same stores and
            policies, with ``c`` appended to the history.
        """
        new = MemoryContext.from_previous(self, c)
        new._window_size = self._window_size  # type: ignore[attr-defined]
        new._stores = self._stores
        new._retrieval_policy = self._retrieval_policy
        new._compaction_policy = self._compaction_policy
        new._turn_budget = self._turn_budget
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Project history, splice recalled memory, then optionally compact."""
        history = super().view_for_generation() or []
        if self._stores:
            query = self._retrieval_policy.derive_query(history)
            recalled = self._retrieval_policy.retrieve(query, self._stores)
            history = self._retrieval_policy.splice(history, recalled)

        if (
            self._compaction_policy is not None
            and self._turn_budget is not None
            and self._compaction_policy.should_compact(history, self._turn_budget)
        ):
            history = self._compaction_policy.compact(history)

        return history


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: Component | CBlock) -> SimpleContext:
        """Add a new component or CBlock to the context and return the updated context.

        Args:
            c (Component | CBlock): The component or content block to record.

        Returns:
            SimpleContext: A new ``SimpleContext`` containing only the added entry;
            prior history is not retained.
        """
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return an empty list, since ``SimpleContext`` does not pass history to the model.

        Each call to the model is treated as a stateless, independent exchange.
        No prior turns are forwarded.

        Returns:
            list[Component | CBlock] | None: Always an empty list.
        """
        return []
