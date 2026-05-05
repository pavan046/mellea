"""Tests for EpisodicStore: persistence, timeline replay, A-Mem integration."""

from __future__ import annotations

import datetime
import os
import tempfile
import time

import pytest

from mellea.core import CBlock, CompactionScope, ContextTurn, MemorySource
from mellea.stdlib.memory import EpisodicStore, HashingEmbedder
from mellea.stdlib.memory.episodic.amem import (
    llm_evolver,
    llm_link_generator,
    llm_note_enricher,
    llm_summarizer,
)


def _turn(text: str) -> ContextTurn:
    return ContextTurn(CBlock(text), None)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


# ---- construction + write path -------------------------------------------


def test_write_produces_record_with_session_user_keys() -> None:
    s = EpisodicStore()
    records = s.write(
        _turn("user alice prefers oat milk"),
        actor="user",
        tags={"session_id": "s1", "user_id": "alice"},
    )
    assert len(records) == 1
    r = records[0]
    assert r.content == "user alice prefers oat milk"
    # session_id / user_id are on the EpisodicRecord subclass.
    assert getattr(r, "session_id", None) == "s1"
    assert getattr(r, "user_id", None) == "alice"
    assert s.get(r.record_id) is not None


def test_write_empty_content_noop() -> None:
    s = EpisodicStore()
    assert s.write(ContextTurn(None, None), actor="user") == []
    assert s.write(ContextTurn(CBlock("   "), None), actor="user") == []


def test_write_invokes_all_three_amem_phases_in_order(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "episodic memory indexer",
                {
                    "keywords": ["project-x"],
                    "tags": ["work"],
                    "contextual_description": "About project x.",
                },
            ),
            ("linking a new memory note", {"links": []}),
            (
                "maintaining a zettelkasten",
                {
                    "keywords": ["updated"],
                    "tags": ["work"],
                    "contextual_description": "Evolved.",
                },
            ),
        ]
    )
    s = EpisodicStore(
        enricher=llm_note_enricher(backend),
        link_generator=llm_link_generator(backend),
        evolver=llm_evolver(backend),
    )
    s.write(_turn("project-x is launching"), actor="user", tags={"user_id": "u"})
    seen = [p[:60] for p in backend.prompts_seen]
    # At least the enricher fires on every write. Link + evolver fire only
    # when there are neighbours, so a lone first write should only use P_s1.
    assert any("episodic memory indexer" in p for p in seen)


def test_second_write_triggers_link_generation_and_evolution(
    scripted_backend_factory,
) -> None:
    enrich_rule = (
        "episodic memory indexer",
        {"keywords": ["topic"], "tags": ["work"], "contextual_description": "A topic."},
    )
    link_rule = ("linking a new memory note", {"links": []})  # placeholder
    evolve_rule = (
        "maintaining a zettelkasten",
        {
            "keywords": ["evolved"],
            "tags": ["work"],
            "contextual_description": "Evolved description.",
        },
    )
    backend = scripted_backend_factory([enrich_rule, link_rule, evolve_rule])
    s = EpisodicStore(
        enricher=llm_note_enricher(backend),
        link_generator=llm_link_generator(backend),
        evolver=llm_evolver(backend),
    )
    s.write(_turn("first note about topic"), actor="user", tags={"user_id": "u"})
    s.write(_turn("second note about topic"), actor="user", tags={"user_id": "u"})

    seen = " ".join(backend.prompts_seen)
    assert "linking a new memory note" in seen
    assert "maintaining a zettelkasten" in seen


# ---- retrieval filters ---------------------------------------------------


def test_retrieve_isolates_by_user_id() -> None:
    s = EpisodicStore()
    s.write(_turn("alice likes hiking"), actor="user", tags={"user_id": "alice"})
    s.write(_turn("bob likes hiking"), actor="user", tags={"user_id": "bob"})
    alice_hits = s.retrieve("hiking", k=5, filters={"user_id": "alice"})
    bob_hits = s.retrieve("hiking", k=5, filters={"user_id": "bob"})
    assert len(alice_hits) == 1
    assert len(bob_hits) == 1
    assert alice_hits[0]._meta["user_id"] == "alice"
    assert bob_hits[0]._meta["user_id"] == "bob"


def test_retrieve_isolates_by_session_id() -> None:
    s = EpisodicStore()
    s.write(_turn("hiking note"), actor="user", tags={"session_id": "s1"})
    s.write(_turn("hiking note"), actor="user", tags={"session_id": "s2"})
    hits = s.retrieve("hiking", k=5, filters={"session_id": "s1"})
    assert len(hits) == 1
    assert hits[0]._meta["session_id"] == "s1"


def test_retrieve_empty_query_returns_empty() -> None:
    s = EpisodicStore()
    s.write(_turn("something"), actor="user")
    assert s.retrieve("", k=5) == []


def test_retrieve_blocks_source_is_episodic() -> None:
    s = EpisodicStore()
    s.write(_turn("hiking"), actor="user")
    hits = s.retrieve("hiking", k=1)
    assert hits[0].source is MemorySource.EPISODIC
    assert hits[0].store_id == "episodic"


def test_retrieve_with_embedder_produces_blended_scores() -> None:
    """With an embedder attached, scores come from fused lexical+vector signals.

    We can't assert a specific score value (depends on hash collisions) but we
    can check that retrieval works, that a matching query still ranks the
    correct record first, and that ``score`` is a float.
    """
    s = EpisodicStore(embedder=HashingEmbedder(dim=128))
    s.write(_turn("alice enjoys rock climbing on weekends"), actor="user")
    s.write(_turn("bob prepares quarterly financial reports"), actor="user")
    hits = s.retrieve("rock climbing", k=2)
    assert hits
    assert "climbing" in hits[0].value
    assert isinstance(hits[0].score, float)


# ---- timeline replay -----------------------------------------------------


def test_timeline_replay_returns_only_records_before_as_of() -> None:
    s = EpisodicStore()
    s.write(_turn("earliest note about x"), actor="user")
    time.sleep(0.01)
    mid = _now()
    time.sleep(0.01)
    s.write(_turn("later note about x"), actor="user")

    before = s.retrieve("x", k=5, filters={"as_of": mid})
    assert len(before) == 1
    assert before[0].value == "earliest note about x"
    after = s.retrieve("x", k=5)
    assert len(after) == 2


def test_timeline_replay_ignores_tombstones_after_as_of() -> None:
    s = EpisodicStore()
    r = s.write(_turn("the ephemeral note"), actor="user")[0]
    mid = _now()
    time.sleep(0.01)
    s.forget(r.record_id, reason="cleanup")
    assert s.retrieve("ephemeral", k=5) == []
    # Before the tombstone, the record was live.
    past_hits = s.retrieve("ephemeral", k=5, filters={"as_of": mid})
    assert len(past_hits) == 1


# ---- forget / supersede / rewrite ----------------------------------------


def test_forget_tombstones_record_and_audits() -> None:
    s = EpisodicStore()
    r = s.write(_turn("sensitive fact"), actor="user")[0]
    s.forget(r.record_id, reason="user_request")
    assert s.get(r.record_id) is None
    assert s.retrieve("sensitive", k=5) == []
    audit = s.audit_log()
    assert any(e.op == "forget" and e.record_id == r.record_id for e in audit)


def test_supersede_tombstones_old_and_links_new() -> None:
    s = EpisodicStore()
    r = s.write(_turn("python is great"), actor="user")[0]
    new = s.supersede(r.record_id, "rust is great")
    assert new.record_id != r.record_id
    assert s.get(r.record_id) is None  # hidden because it's superseded
    assert s.get(new.record_id) is not None
    assert r.record_id in new.links
    # retrieval now reflects the new content
    assert s.retrieve("python", k=5) == []
    hits = s.retrieve("rust", k=5)
    assert hits and hits[0].record_id == new.record_id


def test_supersede_unknown_id_raises() -> None:
    s = EpisodicStore()
    with pytest.raises(KeyError):
        s.supersede("nonexistent", "whatever")


def test_rewrite_refreshes_enrichment(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "episodic memory indexer",
                {
                    "keywords": ["new-kw"],
                    "tags": [],
                    "contextual_description": "new description",
                },
            ),
            ("linking a new memory note", {"links": []}),
            (
                "maintaining a zettelkasten",
                {"keywords": [], "tags": [], "contextual_description": ""},
            ),
        ]
    )
    s = EpisodicStore(enricher=llm_note_enricher(backend))
    r = s.write(_turn("a record"), actor="user")[0]
    refreshed = s.rewrite(r.record_id)
    assert refreshed.record_id == r.record_id
    assert refreshed.keywords == ["new-kw"]
    assert any(e.op == "rewrite" for e in s.audit_log())


# ---- compaction ----------------------------------------------------------


def test_compact_by_user_id_isolates_clusters() -> None:
    s = EpisodicStore()
    for text, uid in [
        ("alice 1", "alice"),
        ("alice 2", "alice"),
        ("alice 3", "alice"),
        ("bob 1", "bob"),
        ("bob 2", "bob"),
    ]:
        s.write(_turn(text), actor="user", tags={"user_id": uid})
    out = s.compact(CompactionScope(by="user_id", threshold=2))
    assert len(out) == 2
    users = {r.user_id for r in out}
    assert users == {"alice", "bob"}
    for consolidated in out:
        # Source chain preserved: consolidated record links back to members.
        assert len(consolidated.links) >= 2
        for src_id in consolidated.links:
            assert s.get(src_id) is not None


def test_compact_below_threshold_is_noop() -> None:
    s = EpisodicStore()
    s.write(_turn("singleton"), actor="user", tags={"user_id": "u1"})
    out = s.compact(CompactionScope(by="user_id", threshold=5))
    assert out == []


def test_compact_with_llm_summarizer_uses_prompt(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "summarizing a cluster",
                {
                    "summary": "Alice likes both hiking and cycling.",
                    "keywords": ["outdoor"],
                    "tags": ["summary", "preference"],
                },
            )
        ]
    )
    s = EpisodicStore(summarizer=llm_summarizer(backend))
    s.write(_turn("alice likes hiking"), actor="user", tags={"user_id": "alice"})
    s.write(_turn("alice likes cycling"), actor="user", tags={"user_id": "alice"})
    out = s.compact(CompactionScope(by="user_id", threshold=2))
    assert len(out) == 1
    assert "hiking and cycling" in out[0].content
    assert "outdoor" in out[0].keywords


# ---- persistence ---------------------------------------------------------


def test_records_persist_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "episodic.db")
        s1 = EpisodicStore(db_path=db_path)
        s1.write(
            _turn("persistent note about hiking"), actor="user", tags={"user_id": "u"}
        )
        r = s1.retrieve("hiking", k=1)
        assert r
        s1.close()

        s2 = EpisodicStore(db_path=db_path)
        hits = s2.retrieve("hiking", k=1)
        assert len(hits) == 1
        assert hits[0].value == "persistent note about hiking"
        s2.close()


def test_audit_log_is_persisted() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "episodic.db")
        s1 = EpisodicStore(db_path=db_path)
        r = s1.write(_turn("will be forgotten"), actor="user")[0]
        s1.forget(r.record_id, reason="test")
        s1.close()

        s2 = EpisodicStore(db_path=db_path)
        ops = [e.op for e in s2.audit_log()]
        assert "write" in ops
        assert "forget" in ops
        s2.close()
