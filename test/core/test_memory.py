"""Unit tests for mellea.core memory abstractions and stdlib reference store."""

from __future__ import annotations

import datetime

import pytest

from mellea.core import (
    CBlock,
    CompactionScope,
    ContextTurn,
    MemoryBlock,
    MemoryRecord,
    MemorySource,
)
from mellea.stdlib.memory import (
    DefaultCompactionPolicy,
    DefaultRetrievalPolicy,
    InMemoryStore,
)

# ---------------------------------------------------------------------------
# MemoryBlock
# ---------------------------------------------------------------------------


def test_memory_block_carries_provenance() -> None:
    now = datetime.datetime.now(datetime.UTC)
    block = MemoryBlock(
        "hi",
        source=MemorySource.EPISODIC,
        store_id="s1",
        record_id="r1",
        retrieved_at=now,
        score=0.75,
        meta={"extra": 1},
    )
    assert block.value == "hi"
    assert block.source is MemorySource.EPISODIC
    assert block.store_id == "s1"
    assert block.record_id == "r1"
    assert block.retrieved_at == now
    assert block.score == pytest.approx(0.75)
    assert block._meta["memory_source"] == "episodic"
    assert block._meta["memory_record_id"] == "r1"
    assert block._meta["extra"] == 1


def test_memory_block_is_cblock_subtype() -> None:
    block = MemoryBlock(
        "hi",
        source=MemorySource.EXTERNAL_DOC,
        store_id="s",
        record_id="r",
        retrieved_at=datetime.datetime.now(datetime.UTC),
    )
    assert isinstance(block, CBlock)
    # repr should surface provenance, not just the value.
    assert "source='external_doc'" in repr(block)


def test_memory_block_score_optional() -> None:
    block = MemoryBlock(
        "hi",
        source=MemorySource.SUMMARY,
        store_id="s",
        record_id="r",
        retrieved_at=datetime.datetime.now(datetime.UTC),
    )
    assert block.score is None
    assert "memory_score" not in block._meta


# ---------------------------------------------------------------------------
# InMemoryStore — MemoryStore ABC contract
# ---------------------------------------------------------------------------


def _write(store: InMemoryStore, text: str, actor: str = "user") -> MemoryRecord:
    records = store.write(ContextTurn(CBlock(text), None), actor=actor)
    assert len(records) == 1
    return records[0]


def test_write_then_retrieve_roundtrip() -> None:
    s = InMemoryStore("t")
    r = _write(s, "The user loves hiking in the Alps")
    hits = s.retrieve("hiking", k=5)
    assert len(hits) == 1
    assert hits[0].record_id == r.record_id
    assert hits[0].store_id == "t"
    assert hits[0].score is not None and hits[0].score > 0


def test_empty_content_write_returns_no_records() -> None:
    s = InMemoryStore("t")
    assert s.write(ContextTurn(None, None), actor="user") == []
    assert s.write(ContextTurn(CBlock("   "), None), actor="user") == []


def test_retrieve_respects_k_and_ranks_by_score() -> None:
    s = InMemoryStore("t")
    _write(s, "hiking hiking hiking")  # strong overlap
    _write(s, "a single mention of hiking")  # weak overlap
    _write(s, "totally unrelated content")  # no overlap

    hits = s.retrieve("hiking", k=1)
    assert len(hits) == 1
    assert hits[0].value == "hiking hiking hiking"


def test_retrieve_filters_by_actor() -> None:
    s = InMemoryStore("t")
    _write(s, "hiking matters", actor="alice")
    _write(s, "hiking matters too", actor="bob")
    hits = s.retrieve("hiking", k=5, filters={"actor": "alice"})
    assert len(hits) == 1
    assert hits[0]._meta["actor"] == "alice"


def test_forget_removes_record_from_retrieval() -> None:
    s = InMemoryStore("t")
    r = _write(s, "secret ingredient is saffron")
    s.forget(r.record_id, reason="user asked")
    assert s.retrieve("saffron", k=5) == []
    assert any(a["op"] == "forget" and a["reason"] == "user asked" for a in s.audit_log)


def test_forget_on_missing_id_is_noop_with_audit() -> None:
    s = InMemoryStore("t")
    s.forget("does-not-exist", reason="cleanup")
    entries = [a for a in s.audit_log if a["op"] == "forget"]
    assert len(entries) == 1
    assert entries[0]["was_present"] is False


def test_supersede_tombstones_old_and_links_new() -> None:
    s = InMemoryStore("t")
    old = _write(s, "favorite language is python")
    new = s.supersede(old.record_id, "favorite language is rust")

    assert new.record_id != old.record_id
    # old should no longer retrieve under its former content.
    assert s.retrieve("python", k=5) == []
    assert s.retrieve("rust", k=5)[0].record_id == new.record_id
    assert new.links == [old.record_id]
    # get() should hide tombstoned records.
    assert s.get(old.record_id) is None
    assert s.get(new.record_id) is not None


def test_supersede_unknown_id_raises() -> None:
    s = InMemoryStore("t")
    with pytest.raises(KeyError):
        s.supersede("nope", "replacement")


def test_rewrite_runs_enricher_and_audit() -> None:
    s = InMemoryStore("t")
    r = _write(s, "greek salad with feta and olives")
    rewritten = s.rewrite(r.record_id)
    assert rewritten.record_id == r.record_id
    assert any(a["op"] == "rewrite" for a in s.audit_log)


def test_rewrite_invokes_injected_enricher() -> None:
    calls: list[str] = []

    def enricher(text: str) -> tuple[list[str], list[str], str]:
        calls.append(text)
        return ["kw"], ["tg"], "desc"

    s = InMemoryStore("t", enricher=enricher)
    r = _write(s, "some content")
    assert calls  # enricher fired on write
    pre_rewrite_len = len(calls)
    s.rewrite(r.record_id)
    assert len(calls) == pre_rewrite_len + 1


def test_link_generator_and_evolver_are_invoked() -> None:
    generated_for: list[str] = []
    evolved_for: list[str] = []

    def link_gen(record: MemoryRecord, neighbors: list[MemoryRecord]) -> list[str]:
        generated_for.append(record.record_id)
        return [n.record_id for n in neighbors]

    def evolver(
        record: MemoryRecord, neighbors: list[MemoryRecord]
    ) -> list[MemoryRecord]:
        evolved_for.append(record.record_id)
        # Tag evolved neighbors so we can confirm persistence.
        for n in neighbors:
            n.tags = sorted(set(n.tags) | {"evolved"})
        return neighbors

    s = InMemoryStore("t", link_generator=link_gen, evolver=evolver)
    first = _write(s, "the user enjoys python programming")
    second = _write(s, "the user also enjoys python notebooks")

    # second write sees first as a neighbor
    assert first.record_id in generated_for or second.record_id in generated_for
    assert evolved_for
    assert "evolved" in s.get(first.record_id).tags


def test_compact_emits_consolidated_records_with_links() -> None:
    s = InMemoryStore("t")
    for i in range(4):
        _write(s, f"daily note {i} about backups", actor="alice")

    produced = s.compact(CompactionScope(by="actor", threshold=3))
    assert produced
    cons = produced[0]
    assert "compaction:actor" in cons.tags
    # 100% provenance: every consolidated record links to all sources.
    assert set(cons.links) == {
        r.record_id
        for r in s.all_records()
        if r.actor == "alice" and r.record_id != cons.record_id
    }


def test_compact_skips_clusters_below_threshold() -> None:
    s = InMemoryStore("t")
    _write(s, "only one alice note", actor="alice")
    produced = s.compact(CompactionScope(by="actor", threshold=5))
    assert produced == []


def test_compact_max_age_filter() -> None:
    s = InMemoryStore("t")
    # Two "recent" writes, threshold >= 2, but max_age=1h excludes them.
    _write(s, "alice note 1", actor="alice")
    _write(s, "alice note 2", actor="alice")
    assert (
        s.compact(
            CompactionScope(
                by="actor", threshold=2, max_age=datetime.timedelta(hours=1)
            )
        )
        == []
    )


def test_audit_log_is_snapshot() -> None:
    s = InMemoryStore("t")
    _write(s, "some content")
    log1 = s.audit_log
    _write(s, "more content")
    assert len(log1) == 1  # snapshot semantics: not mutated in place


def test_all_records_excludes_tombstones_by_default() -> None:
    s = InMemoryStore("t")
    r = _write(s, "to be replaced")
    s.supersede(r.record_id, "replacement")
    assert len(s.all_records()) == 1
    assert len(s.all_records(include_tombstoned=True)) == 2


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_default_retrieval_policy_splice_position() -> None:
    s = InMemoryStore("t")
    _write(s, "the user likes mangoes")

    policy = DefaultRetrievalPolicy(k=3)
    history: list = [
        CBlock("old"),
        CBlock("newer"),
        CBlock("the user likes what fruit"),
    ]
    query = policy.derive_query(history)
    recalled = policy.retrieve(query, [s])
    spliced = policy.splice(history, recalled)

    assert spliced[-1] is history[-1]
    # recalled block should sit just before the last entry.
    assert isinstance(spliced[-2], MemoryBlock)


def test_default_retrieval_policy_empty_query_returns_nothing() -> None:
    s = InMemoryStore("t")
    _write(s, "irrelevant")
    policy = DefaultRetrievalPolicy()
    assert policy.retrieve("", [s]) == []


def test_default_retrieval_policy_fuses_across_stores() -> None:
    s1 = InMemoryStore("store1")
    s2 = InMemoryStore("store2")
    _write(s1, "store1 knows about hiking")
    _write(s2, "store2 also knows about hiking")
    policy = DefaultRetrievalPolicy(k=5)
    hits = policy.retrieve("hiking", [s1, s2])
    ids = {h.store_id for h in hits}
    assert ids == {"store1", "store2"}


def test_default_compaction_policy_folds_oldest() -> None:
    store = InMemoryStore("compaction-test")
    policy = DefaultCompactionPolicy(
        summarizer=lambda text: f"SUMMARY[{len(text)}]", store=store, fold_fraction=0.5
    )
    history = [CBlock(f"turn {i}") for i in range(4)]
    assert policy.should_compact(history, budget=3)
    folded = policy.compact(history)
    assert isinstance(folded[0], MemoryBlock)
    assert folded[0].source is MemorySource.SUMMARY
    assert folded[0].store_id == "compaction-test"
    assert store.get(folded[0].record_id) is not None
    assert folded[0]._meta["folded_count"] == 2
    assert len(folded) == 3  # one summary + remaining tail


def test_default_compaction_policy_under_budget_noop() -> None:
    policy = DefaultCompactionPolicy(
        summarizer=lambda _text: "unused", store=InMemoryStore("compaction-test")
    )
    history = [CBlock("only")]
    assert not policy.should_compact(history, budget=5)


def test_default_compaction_policy_empty_summary_preserves_history() -> None:
    store = InMemoryStore("compaction-test")
    policy = DefaultCompactionPolicy(
        summarizer=lambda _text: "", store=store, fold_fraction=0.5
    )
    history = [CBlock(f"turn {i}") for i in range(4)]
    folded = policy.compact(history)
    assert folded == history
    assert store.all_records() == []
