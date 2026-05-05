"""Dict-backed reference implementation of ``MemoryStore``.

Ships zero external dependencies and no LLM calls in the default path. Evolution
and note-enrichment callbacks are injectable so workstream C can layer the
A-Mem three-phase write on top without subclassing. Scoring is a simple BM25-ish
lexical overlap — good enough for unit tests and small scripts, deliberately
replaced by the vector/graph stores later.

Intended use: tests, examples, and sketch scripts where persistence isn't
needed. Do not use in production — ``VectorStore`` / ``EpisodicStore`` /
``GraphStore`` are the durable carriers.
"""

from __future__ import annotations

import datetime
import math
import re
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from ...core.base import ContextTurn, MemoryBlock, MemorySource, ModelOutputThunk
from ...core.memory import CompactionScope, MemoryRecord, MemoryStore

NoteEnricher = Callable[[str], tuple[list[str], list[str], str]]
"""Takes raw content, returns ``(keywords, tags, contextual_description)``.

Injected by workstream C to run the A-Mem note-construction prompt (P_s1).
Default is a lexical heuristic so the store works without LLM access.
"""

LinkGenerator = Callable[[MemoryRecord, list[MemoryRecord]], list[str]]
"""Takes a new record and its nearest neighbors, returns the ``links`` field.

Injected by workstream C to run the A-Mem link-generation prompt (P_s2).
Default returns an empty list.
"""

Evolver = Callable[[MemoryRecord, list[MemoryRecord]], list[MemoryRecord]]
"""Takes a new record and its neighbors, returns the evolved neighbors.

Injected by workstream C to run the A-Mem evolution prompt (P_s3).
Default is a no-op.
"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _default_enricher(content: str) -> tuple[list[str], list[str], str]:
    """Lexical fallback for the A-Mem P_s1 note-construction prompt."""
    toks = _tokenize(content)
    counts = Counter(toks)
    keywords = [w for w, _ in counts.most_common(5)]
    return keywords, [], content[:160]


def _default_links(record: MemoryRecord, neighbors: list[MemoryRecord]) -> list[str]:
    return []


def _default_evolve(
    record: MemoryRecord, neighbors: list[MemoryRecord]
) -> list[MemoryRecord]:
    return neighbors


class InMemoryStore(MemoryStore):
    """Dict-backed reference ``MemoryStore`` with lexical scoring.

    Records live in ``self._records`` keyed by ``record_id``. Tombstoned
    records stay in the dict (so supersede pointers still resolve) but are
    filtered out of retrieval. Scoring is BM25-lite over tokenized content +
    keywords + contextual description.

    Args:
        store_id (str): Stable identifier for this store.
        write_on_turn (bool): Whether ``MelleaSession`` auto-writes after each
            completed turn. Defaults to ``True`` here since this is the
            reference store.
        enricher (NoteEnricher | None): Optional A-Mem P_s1 implementation.
            Default uses a lexical heuristic.
        link_generator (LinkGenerator | None): Optional A-Mem P_s2 implementation.
        evolver (Evolver | None): Optional A-Mem P_s3 implementation.
        neighbor_k (int): Neighbors considered for link + evolve. Defaults to 5.
    """

    def __init__(
        self,
        store_id: str = "in_memory",
        *,
        write_on_turn: bool = True,
        enricher: NoteEnricher | None = None,
        link_generator: LinkGenerator | None = None,
        evolver: Evolver | None = None,
        neighbor_k: int = 5,
    ):
        """Initialize InMemoryStore with optional A-Mem phase implementations."""
        super().__init__(store_id, write_on_turn=write_on_turn)
        self._records: dict[str, MemoryRecord] = {}
        self._audit: list[dict[str, Any]] = []
        self._enricher = enricher or _default_enricher
        self._link_generator = link_generator or _default_links
        self._evolver = evolver or _default_evolve
        self._neighbor_k = neighbor_k

    # ---- public inspection helpers (not part of the ABC) ----

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Audit entries for every write/forget/supersede/rewrite/compact."""
        return list(self._audit)

    def all_records(self, *, include_tombstoned: bool = False) -> list[MemoryRecord]:
        """Snapshot of stored records; excludes tombstones by default."""
        if include_tombstoned:
            return list(self._records.values())
        return [r for r in self._records.values() if r.superseded_by is None]

    # ---- MemoryStore ABC ----

    def write(
        self, turn: ContextTurn, *, actor: str, tags: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        """A-Mem three-phase write: note construction → link generation → evolution."""
        content = self._turn_to_text(turn)
        if not content.strip():
            return []

        keywords, auto_tags, desc = self._enricher(content)
        tag_list = sorted(set(auto_tags) | set((tags or {}).keys()))

        record = MemoryRecord(
            record_id=str(uuid.uuid4()),
            content=content,
            timestamp=datetime.datetime.now(datetime.UTC),
            actor=actor,
            keywords=keywords,
            tags=tag_list,
            contextual_description=desc,
            embedding=[],
            links=[],
            meta=dict(tags or {}),
        )

        neighbors = self._top_k_records(content, k=self._neighbor_k)
        record.links = self._link_generator(record, neighbors)
        self._records[record.record_id] = record

        evolved = self._evolver(record, neighbors)
        for m in evolved:
            if m.record_id in self._records:
                self._records[m.record_id] = m

        self._audit.append(
            {
                "op": "write",
                "record_id": record.record_id,
                "actor": actor,
                "timestamp": record.timestamp.isoformat(),
            }
        )
        return [record]

    def retrieve(
        self, query: str, *, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[MemoryBlock]:
        """Top-k lexical scoring, filters applied before ranking."""
        if not query.strip():
            return []
        candidates = [r for r in self._records.values() if self._passes(r, filters)]
        scored = [(self._score(query, r), r) for r in candidates]
        scored = [(s, r) for s, r in scored if s > 0.0]
        scored.sort(key=lambda sr: sr[0], reverse=True)
        top = scored[:k]

        now = datetime.datetime.now(datetime.UTC)
        return [
            MemoryBlock(
                record.content,
                source=self._source_for(record),
                store_id=self._store_id,
                record_id=record.record_id,
                retrieved_at=now,
                score=score,
                meta={
                    "keywords": list(record.keywords),
                    "tags": list(record.tags),
                    "actor": record.actor,
                    "timestamp": record.timestamp.isoformat(),
                },
            )
            for score, record in top
        ]

    def forget(self, record_id: str, *, reason: str) -> None:
        """Remove a record outright; audit the reason."""
        existing = self._records.pop(record_id, None)
        self._audit.append(
            {
                "op": "forget",
                "record_id": record_id,
                "reason": reason,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "was_present": existing is not None,
            }
        )

    def supersede(self, record_id: str, new_value: str) -> MemoryRecord:
        """Tombstone the old record, emit a fresh one, link them both ways."""
        if record_id not in self._records:
            raise KeyError(f"record_id {record_id!r} not found")
        old = self._records[record_id]
        keywords, auto_tags, desc = self._enricher(new_value)
        new = MemoryRecord(
            record_id=str(uuid.uuid4()),
            content=new_value,
            timestamp=datetime.datetime.now(datetime.UTC),
            actor=old.actor,
            keywords=keywords,
            tags=sorted(set(auto_tags) | set(old.tags)),
            contextual_description=desc,
            embedding=[],
            links=[record_id],
            meta={**old.meta, "supersedes": record_id},
        )
        old.superseded_by = new.record_id
        self._records[new.record_id] = new
        self._audit.append(
            {
                "op": "supersede",
                "old_record_id": record_id,
                "new_record_id": new.record_id,
                "timestamp": new.timestamp.isoformat(),
            }
        )
        return new

    def rewrite(self, record_id: str) -> MemoryRecord:
        """Re-enrich and re-evolve a record against current neighbors."""
        if record_id not in self._records:
            raise KeyError(f"record_id {record_id!r} not found")
        record = self._records[record_id]
        keywords, auto_tags, desc = self._enricher(record.content)
        record.keywords = keywords
        record.tags = sorted(set(auto_tags) | set(record.tags))
        record.contextual_description = desc

        neighbors = self._top_k_records(
            record.content, k=self._neighbor_k, exclude={record_id}
        )
        record.links = self._link_generator(record, neighbors)
        evolved = self._evolver(record, neighbors)
        for m in evolved:
            if m.record_id in self._records:
                self._records[m.record_id] = m

        self._audit.append(
            {
                "op": "rewrite",
                "record_id": record_id,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        return record

    def compact(self, scope: CompactionScope) -> list[MemoryRecord]:
        """Cluster by ``scope.by`` and emit one consolidated record per cluster.

        Sources are preserved (no deletions); the consolidated record's
        ``links`` field holds every source ``record_id``.
        """
        now = datetime.datetime.now(datetime.UTC)
        min_age = scope.max_age
        eligible = [
            r
            for r in self._records.values()
            if r.superseded_by is None
            and (min_age is None or (now - r.timestamp) >= min_age)
        ]

        clusters: dict[str, list[MemoryRecord]] = {}
        for r in eligible:
            for key in self._cluster_keys(r, scope.by):
                clusters.setdefault(key, []).append(r)

        produced: list[MemoryRecord] = []
        for key, members in clusters.items():
            if len(members) < scope.threshold:
                continue
            consolidated = MemoryRecord(
                record_id=str(uuid.uuid4()),
                content=self._summarize([m.content for m in members]),
                timestamp=now,
                actor="compaction",
                keywords=sorted({k for m in members for k in m.keywords}),
                tags=sorted(
                    {t for m in members for t in m.tags}
                    | {f"compaction:{scope.by}", f"cluster:{key}"}
                ),
                contextual_description=f"Consolidated {len(members)} records on {scope.by}={key!r}",
                embedding=[],
                links=[m.record_id for m in members],
                meta={
                    "compaction_scope": scope.by,
                    "compaction_key": key,
                    "source_count": len(members),
                },
            )
            self._records[consolidated.record_id] = consolidated
            produced.append(consolidated)
            self._audit.append(
                {
                    "op": "compact",
                    "record_id": consolidated.record_id,
                    "source_ids": list(consolidated.links),
                    "scope": scope.by,
                    "key": key,
                    "timestamp": now.isoformat(),
                }
            )
        return produced

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return the record, or ``None`` if missing or tombstoned."""
        record = self._records.get(record_id)
        if record is None or record.superseded_by is not None:
            return None
        return record

    # ---- internals ----

    @staticmethod
    def _turn_to_text(turn: ContextTurn) -> str:
        parts: list[str] = []
        if turn.model_input is not None:
            parts.append(str(turn.model_input))
        if turn.output is not None and isinstance(turn.output, ModelOutputThunk):
            if turn.output.value is not None:
                parts.append(str(turn.output.value))
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _source_for(record: MemoryRecord) -> MemorySource:
        """Pick a reasonable ``MemorySource`` tag for a retrieved record."""
        if "compaction_scope" in record.meta:
            return MemorySource.SUMMARY
        return MemorySource.EPISODIC

    @staticmethod
    def _passes(record: MemoryRecord, filters: dict[str, Any] | None) -> bool:
        if record.superseded_by is not None:
            return False
        if not filters:
            return True
        if "actor" in filters and record.actor != filters["actor"]:
            return False
        if "tag" in filters and filters["tag"] not in record.tags:
            return False
        if "since" in filters and record.timestamp < filters["since"]:
            return False
        if "until" in filters and record.timestamp > filters["until"]:
            return False
        return True

    def _score(self, query: str, record: MemoryRecord) -> float:
        """BM25-lite lexical overlap over content + keywords + description."""
        qtoks = _tokenize(query)
        if not qtoks:
            return 0.0
        doc_tokens = _tokenize(
            record.content
            + " "
            + " ".join(record.keywords)
            + " "
            + record.contextual_description
        )
        if not doc_tokens:
            return 0.0
        doc_counts = Counter(doc_tokens)
        score = 0.0
        doc_len = len(doc_tokens)
        avg_len = 50.0  # rough prior; bounded constant is fine for in-memory tests
        k1, b = 1.5, 0.75
        for q in set(qtoks):
            tf = doc_counts.get(q, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (len(self._records) + 1) / (self._doc_freq(q) + 0.5))
            norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += idf * norm
        return score

    def _doc_freq(self, term: str) -> int:
        return sum(
            1
            for r in self._records.values()
            if r.superseded_by is None and term in _tokenize(r.content)
        )

    def _top_k_records(
        self, query: str, *, k: int, exclude: set[str] | None = None
    ) -> list[MemoryRecord]:
        exclude = exclude or set()
        scored = [
            (self._score(query, r), r)
            for r in self._records.values()
            if r.superseded_by is None and r.record_id not in exclude
        ]
        scored = [(s, r) for s, r in scored if s > 0.0]
        scored.sort(key=lambda sr: sr[0], reverse=True)
        return [r for _, r in scored[:k]]

    @staticmethod
    def _cluster_keys(record: MemoryRecord, by: str) -> list[str]:
        if by == "actor":
            return [record.actor]
        if by == "tag":
            return list(record.tags) or ["__untagged__"]
        if by == "time_bucket":
            return [record.timestamp.strftime("%Y-%m-%d")]
        if by == "entity":
            # Workstream D will override with entity extraction; fall back to keywords.
            return list(record.keywords) or ["__no_entity__"]
        raise ValueError(f"unknown CompactionScope.by: {by!r}")

    @staticmethod
    def _summarize(contents: list[str]) -> str:
        """Trivial concatenation summary. Workstream C replaces this with LLM."""
        return "\n---\n".join(contents)
