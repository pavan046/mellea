"""Unit tests for MemoryContext and MelleaSession memory verbs."""

from __future__ import annotations

import pytest

from mellea.core import CBlock, MemoryBlock
from mellea.stdlib.context import MemoryContext
from mellea.stdlib.memory import (
    DefaultCompactionPolicy,
    DefaultRetrievalPolicy,
    InMemoryStore,
)
from mellea.stdlib.session import MelleaSession


class _StubBackend:
    """Minimal stand-in for Backend; memory tests never call generate()."""



# ---------------------------------------------------------------------------
# MemoryContext
# ---------------------------------------------------------------------------


def test_memory_context_is_immutable_add_returns_new_instance() -> None:
    ctx = MemoryContext()
    ctx2 = ctx.add(CBlock("hello"))
    assert ctx is not ctx2
    assert ctx.view_for_generation() == []
    assert len(ctx2.view_for_generation()) == 1


def test_memory_context_propagates_stores_across_adds() -> None:
    s = InMemoryStore("p")
    ctx = MemoryContext(stores=[s]).add(CBlock("first")).add(CBlock("second"))
    assert ctx.stores[0] is s


def test_memory_context_splice_injects_memory_blocks() -> None:
    s = InMemoryStore("p")
    from mellea.core import ContextTurn

    s.write(ContextTurn(CBlock("user enjoys hiking in the alps"), None), actor="user")
    ctx = MemoryContext(stores=[s]).add(CBlock("what do I enjoy hiking"))

    view = ctx.view_for_generation()
    assert view is not None
    assert any(isinstance(v, MemoryBlock) for v in view)
    # User's final turn stays at the end.
    assert isinstance(view[-1], CBlock)
    assert "enjoy hiking" in view[-1].value


def test_memory_context_no_stores_no_splice() -> None:
    ctx = MemoryContext().add(CBlock("standalone"))
    view = ctx.view_for_generation()
    assert view == [ctx.node_data]


def test_memory_context_compaction_triggered_over_budget() -> None:
    ctx = MemoryContext(
        compaction_policy=DefaultCompactionPolicy(fold_fraction=0.5), turn_budget=3
    )
    for i in range(5):
        ctx = ctx.add(CBlock(f"turn {i}"))
    view = ctx.view_for_generation()
    # Expect at least one summary block at the head.
    assert isinstance(view[0], MemoryBlock)
    assert view[0].value.startswith("Summary of prior turns")


# ---------------------------------------------------------------------------
# MelleaSession memory verbs
# ---------------------------------------------------------------------------


def _session_with_memory(
    store: InMemoryStore | None = None,
) -> tuple[MelleaSession, InMemoryStore]:
    s = store or InMemoryStore("test", write_on_turn=True)
    m = MelleaSession(backend=_StubBackend(), ctx=MemoryContext(stores=[s]))  # type: ignore[arg-type]
    return m, s


def test_remember_writes_to_attached_store() -> None:
    m, s = _session_with_memory()
    records = m.remember("user enjoys tea")
    assert len(records) == 1
    assert records[0].content == "user enjoys tea"
    assert s.get(records[0].record_id) is not None


def test_remember_with_no_memory_store_raises() -> None:
    m = MelleaSession(backend=_StubBackend())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="No MemoryStore attached"):
        m.remember("anything")


def test_remember_targets_specific_store_by_id() -> None:
    s1 = InMemoryStore("alpha")
    s2 = InMemoryStore("beta")
    m = MelleaSession(
        backend=_StubBackend(),  # type: ignore[arg-type]
        ctx=MemoryContext(stores=[s1, s2]),
    )
    m.remember("fact", store_id="beta")
    assert s1.all_records() == []
    assert len(s2.all_records()) == 1


def test_recall_returns_memory_blocks() -> None:
    m, _ = _session_with_memory()
    m.remember("user avoids dairy due to allergy")
    hits = m.recall("dairy")
    assert hits
    assert all(isinstance(h, MemoryBlock) for h in hits)


def test_forget_via_session_removes_record() -> None:
    m, s = _session_with_memory()
    rec = m.remember("ephemeral fact")[0]
    m.forget(rec.record_id, reason="user requested")
    assert s.get(rec.record_id) is None


def test_recall_with_unknown_store_id_returns_empty() -> None:
    m, _ = _session_with_memory()
    m.remember("some fact")
    assert m.recall("fact", store_id="nonexistent") == []
