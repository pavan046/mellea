"""Tests for the LLM-backed A-Mem phase callables."""

from __future__ import annotations

import datetime

import pytest

from mellea.core.memory import MemoryRecord
from mellea.stdlib.memory.episodic.amem import (
    _parse_json_object,
    llm_evolver,
    llm_link_generator,
    llm_note_enricher,
    llm_summarizer,
)

# ---- parse_json_object ---------------------------------------------------


def test_parse_recovers_plain_json() -> None:
    out = _parse_json_object(
        '{"keywords": ["a"], "tags": []}', expected_keys=["keywords"]
    )
    assert out == {"keywords": ["a"], "tags": []}


def test_parse_recovers_fenced_json() -> None:
    fenced = '```json\n{"x": 1}\n```'
    assert _parse_json_object(fenced, expected_keys=["x"]) == {"x": 1}


def test_parse_recovers_json_from_chatter() -> None:
    msg = 'Sure! Here is the object: {"a": 2} hope that helps'
    assert _parse_json_object(msg, expected_keys=["a"]) == {"a": 2}


def test_parse_returns_empty_on_failure() -> None:
    assert _parse_json_object("no json here", expected_keys=["a"]) == {}
    assert _parse_json_object("", expected_keys=["a"]) == {}


# ---- note enricher -------------------------------------------------------


def test_note_enricher_invokes_prompt_and_parses(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "episodic memory indexer",
                {
                    "keywords": ["hiking", "alps"],
                    "tags": ["preference"],
                    "contextual_description": "User enjoys hiking in the Alps.",
                },
            )
        ]
    )
    enrich = llm_note_enricher(backend)
    kws, tags, desc = enrich(
        "I love hiking in the Alps",
        actor="alice",
        timestamp=datetime.datetime.now(datetime.UTC),
    )
    assert kws == ["hiking", "alps"]
    assert tags == ["preference"]
    assert desc == "User enjoys hiking in the Alps."
    assert backend.prompts_seen  # actually called


def test_note_enricher_falls_back_on_broken_json(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [("episodic memory indexer", "not even close to json")]
    )
    enrich = llm_note_enricher(backend)
    kws, tags, desc = enrich("a short input")
    assert kws == []
    assert tags == []
    assert desc.startswith("a short input")


# ---- link generator ------------------------------------------------------


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def test_link_generator_filters_hallucinated_ids(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [("linking a new memory note", {"links": ["n1", "n2", "bogus-id"]})]
    )
    gen = llm_link_generator(backend)
    neighbors = [
        MemoryRecord(record_id="n1", content="neighbor 1", timestamp=_now(), actor="u"),
        MemoryRecord(record_id="n2", content="neighbor 2", timestamp=_now(), actor="u"),
    ]
    new = MemoryRecord(record_id="new", content="new note", timestamp=_now(), actor="u")
    links = gen(new, neighbors)
    assert set(links) == {"n1", "n2"}


def test_link_generator_no_neighbors_short_circuits(scripted_backend_factory) -> None:
    backend = scripted_backend_factory([])
    gen = llm_link_generator(backend)
    new = MemoryRecord(record_id="new", content="x", timestamp=_now(), actor="u")
    assert gen(new, []) == []
    # The backend was never called, so an empty ruleset is safe.
    assert backend.prompts_seen == []


# ---- evolver -------------------------------------------------------------


def test_evolver_rewrites_neighbor_fields_in_place(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "maintaining a zettelkasten",
                {
                    "keywords": ["project-x", "escalation"],
                    "tags": ["work", "urgent"],
                    "contextual_description": "Updated to reflect project-x escalation.",
                },
            )
        ]
    )
    evolve = llm_evolver(backend)
    neighbor = MemoryRecord(
        record_id="n1",
        content="status update",
        timestamp=_now(),
        actor="u",
        keywords=["old"],
        tags=["work"],
        contextual_description="old description",
    )
    new = MemoryRecord(
        record_id="new", content="project-x is now urgent", timestamp=_now(), actor="u"
    )
    evolved = evolve(new, [neighbor])
    assert len(evolved) == 1
    assert "project-x" in evolved[0].keywords
    assert "urgent" in evolved[0].tags
    assert "project-x" in evolved[0].contextual_description


def test_evolver_preserves_fields_on_broken_response(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [("maintaining a zettelkasten", "garbage response")]
    )
    evolve = llm_evolver(backend)
    neighbor = MemoryRecord(
        record_id="n1",
        content="status",
        timestamp=_now(),
        actor="u",
        keywords=["original"],
        tags=["work"],
        contextual_description="unchanged",
    )
    new = MemoryRecord(record_id="new", content="new note", timestamp=_now(), actor="u")
    evolved = evolve(new, [neighbor])
    assert evolved[0].keywords == ["original"]
    assert evolved[0].tags == ["work"]
    assert evolved[0].contextual_description == "unchanged"


# ---- summarizer ----------------------------------------------------------


def test_summarizer_returns_summary_keywords_tags(scripted_backend_factory) -> None:
    backend = scripted_backend_factory(
        [
            (
                "summarizing a cluster",
                {
                    "summary": "Alice cares about accessibility.",
                    "keywords": ["accessibility"],
                    "tags": ["summary", "preference"],
                },
            )
        ]
    )
    summarize = llm_summarizer(backend)
    records = [
        MemoryRecord(
            record_id="r1",
            content="alice says a11y matters",
            timestamp=_now(),
            actor="u",
        ),
        MemoryRecord(
            record_id="r2",
            content="alice reviewed a11y audit",
            timestamp=_now(),
            actor="u",
        ),
    ]
    summary, keywords, tags = summarize(records)
    assert "accessibility" in summary.lower()
    assert keywords == ["accessibility"]
    assert "summary" in tags


def test_summarizer_falls_back_to_concatenation(scripted_backend_factory) -> None:
    backend = scripted_backend_factory([("summarizing a cluster", "no json here")])
    summarize = llm_summarizer(backend)
    r1 = MemoryRecord(record_id="r1", content="A", timestamp=_now(), actor="u")
    r2 = MemoryRecord(record_id="r2", content="B", timestamp=_now(), actor="u")
    summary, _, tags = summarize([r1, r2])
    assert "A" in summary and "B" in summary
    assert "summary" in tags
