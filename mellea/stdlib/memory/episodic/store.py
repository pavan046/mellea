"""EpisodicStore — A-Mem-style note network with SQLite persistence.

Implements the ``MemoryStore`` ABC per the proposal's §5.1:

* **Three-phase sync-on-write** (A-Mem §3.1-§3.3): ``write`` runs note
  construction, link generation, and memory evolution in sequence. A
  just-written note is retrievable only after its neighbours' keywords,
  tags, and contextual descriptions have been updated.
* **Cross-session persistence**: every record is keyed by
  ``(session_id, user_id)`` so one user's memory doesn't leak into
  another's, and scripts can filter ``retrieve(filters={"user_id": ...})``.
* **Timeline replay**: ``retrieve(filters={"as_of": T})`` returns only the
  notes that existed and were not forgotten or superseded before ``T``.
* **Optional embedder** (reuses B's ``Embedder``): when provided, retrieval
  fuses lexical (FTS5) and vector (cosine) signals; absent, it falls back
  to FTS5 alone.
* **Tombstone semantics**: ``forget`` sets ``forgotten_at`` on the row; the
  record stays in the DB so timeline replay still works.
* **LLM-backed compaction**: ``compact(scope)`` clusters by the scope
  dimension, calls the summarizer, and persists the consolidated record
  with ``links`` pointing at every source.

The store is thread-safe enough for a single writer and many readers via
SQLite's WAL mode; multi-writer concurrency would need a connection pool.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from ....core.base import ContextTurn, MemoryBlock, MemorySource, ModelOutputThunk
from ....core.memory import CompactionScope, MemoryRecord, MemoryStore
from ..embedders import Embedder
from .schema import (
    AuditEntry,
    EpisodicRecord,
    connect,
    record_to_row,
    row_embedding,
    row_to_audit,
    row_to_record,
)

NoteEnricher = Callable[..., tuple[list[str], list[str], str]]
"""``(content, *, actor=None, timestamp=None) -> (keywords, tags, contextual_description)``."""

LinkGenerator = Callable[[MemoryRecord, list[MemoryRecord]], list[str]]
"""``(new_record, neighbors) -> list[linked_record_ids]``."""

Evolver = Callable[[MemoryRecord, list[MemoryRecord]], list[MemoryRecord]]
"""``(new_record, neighbors) -> evolved_neighbors`` (may be mutated in place)."""

Summarizer = Callable[[list[MemoryRecord]], tuple[str, list[str], list[str]]]
"""``(cluster_records) -> (summary_text, keywords, tags)``."""


class EpisodicStore(MemoryStore):
    """SQLite-backed A-Mem episodic memory.

    Args:
        store_id (str): Stable identifier embedded in every emitted
            ``MemoryBlock``.
        db_path (str): Path to the SQLite database file. Created if missing.
            Use ``":memory:"`` for ephemeral stores in tests.
        enricher (NoteEnricher | None): Implements A-Mem P_s1 (note
            construction). Required for meaningful notes; fall back is a
            content-truncating heuristic.
        link_generator (LinkGenerator | None): Implements A-Mem P_s2. Default
            returns an empty list (no inter-note links).
        evolver (Evolver | None): Implements A-Mem P_s3. Default is a no-op.
        summarizer (Summarizer | None): Used by ``compact`` to consolidate
            clusters. Default concatenates member content.
        embedder (Embedder | None): Optional dense encoder. When provided,
            retrieval blends lexical and vector scores.
        neighbor_k (int): Number of nearest neighbours considered for link
            generation and evolution. Defaults to 5 per A-Mem.
        write_on_turn (bool): Auto-write every completed turn. Defaults to
            ``True`` — episodic memory is usually what users want captured.
    """

    def __init__(
        self,
        store_id: str = "episodic",
        *,
        db_path: str = ":memory:",
        enricher: NoteEnricher | None = None,
        link_generator: LinkGenerator | None = None,
        evolver: Evolver | None = None,
        summarizer: Summarizer | None = None,
        embedder: Embedder | None = None,
        neighbor_k: int = 5,
        write_on_turn: bool = True,
    ):
        """Initialize EpisodicStore with SQLite persistence and A-Mem callables."""
        super().__init__(store_id, write_on_turn=write_on_turn)
        self._conn: sqlite3.Connection = connect(db_path)
        self._db_path = db_path
        self._enricher = enricher or _default_enricher
        self._link_generator = link_generator or _default_links
        self._evolver = evolver or _default_evolver
        self._summarizer = summarizer or _default_summarizer
        self._embedder = embedder
        self._neighbor_k = neighbor_k

    # ---- inspection ------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Close the SQLite connection on GC."""
        try:
            self._conn.close()
        except Exception:
            pass

    def audit_log(self, *, limit: int | None = None) -> list[AuditEntry]:
        """Return audit entries, most recent first."""
        q = "SELECT rowid, op, record_id, timestamp, payload FROM audit ORDER BY rowid DESC"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        return [row_to_audit(r) for r in self._conn.execute(q)]

    def all_records(
        self, *, include_tombstoned: bool = False, include_superseded: bool = False
    ) -> list[EpisodicRecord]:
        """Snapshot of records; filters tombstoned and superseded by default."""
        clauses: list[str] = []
        if not include_tombstoned:
            clauses.append("forgotten_at IS NULL")
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [
            row_to_record(r)
            for r in self._conn.execute(
                f"SELECT * FROM records {where} ORDER BY timestamp"
            )
        ]

    # ---- MemoryStore ABC -------------------------------------------------

    def write(
        self, turn: ContextTurn, *, actor: str, tags: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        """A-Mem three-phase write (sync): construction -> linking -> evolution."""
        content = self._turn_to_text(turn)
        if not content.strip():
            return []
        tags = tags or {}
        now = datetime.datetime.now(datetime.UTC)
        session_id = tags.pop("session_id", None)
        user_id = tags.pop("user_id", None)

        # Phase 1: note construction.
        keywords, auto_tags, desc = self._enricher(content, actor=actor, timestamp=now)
        tag_list = sorted(set(auto_tags) | set(tags.keys()))

        embedding_vec = (
            self._embedder.embed(_enrichment_text(content, keywords, tag_list, desc))
            if self._embedder
            else np.zeros((0,), dtype=np.float32)
        )

        record = EpisodicRecord(
            record_id=str(uuid.uuid4()),
            content=content,
            timestamp=now,
            actor=actor,
            keywords=keywords,
            tags=tag_list,
            contextual_description=desc,
            embedding=embedding_vec.tolist(),
            links=[],
            meta=dict(tags),
            session_id=session_id,
            user_id=user_id,
        )

        # Phase 2: link generation against top-k neighbours.
        neighbors = self._top_k_neighbors(
            record, k=self._neighbor_k, session_id=session_id, user_id=user_id
        )
        record.links = self._link_generator(record, neighbors)

        # Persist the new record first so evolution sees consistent ids.
        self._upsert_record(record)

        # Phase 3: evolve neighbours in place, persist changes.
        evolved = self._evolver(record, neighbors)
        for m in evolved:
            self._upsert_record(m)

        self._append_audit(
            "write",
            record.record_id,
            {
                "actor": actor,
                "session_id": session_id,
                "user_id": user_id,
                "neighbor_count": len(neighbors),
                "link_count": len(record.links),
            },
        )
        return [record]

    def retrieve(
        self, query: str, *, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[MemoryBlock]:
        """Top-k hybrid lexical + vector retrieval with filter/timeline support."""
        if not query.strip():
            return []
        filters = filters or {}

        rows = self._candidate_rows(filters)
        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        fts_scores = self._fts_scores(query, [r["record_id"] for r in rows])

        q_vec: np.ndarray | None = None
        if self._embedder is not None:
            q_vec = self._embedder.embed(query)

        for row in rows:
            lex = fts_scores.get(row["record_id"], 0.0)
            score = lex
            if q_vec is not None and q_vec.size > 0:
                vec = row_embedding(row)
                if vec.size == q_vec.size and vec.size > 0:
                    cos = float(np.dot(vec, q_vec))
                    score = 0.5 * lex + 0.5 * cos
            if score <= 0.0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:k]

        retrieved_at = datetime.datetime.now(datetime.UTC)
        blocks: list[MemoryBlock] = []
        for score, row in top:
            record = row_to_record(row)
            blocks.append(
                MemoryBlock(
                    record.content,
                    source=_source_for(record),
                    store_id=self._store_id,
                    record_id=record.record_id,
                    retrieved_at=retrieved_at,
                    score=score,
                    meta={
                        "keywords": list(record.keywords),
                        "tags": list(record.tags),
                        "actor": record.actor,
                        "session_id": record.session_id,
                        "user_id": record.user_id,
                        "timestamp": record.timestamp.isoformat(),
                    },
                )
            )
        return blocks

    def forget(self, record_id: str, *, reason: str) -> None:
        """Tombstone a record. Still returned in timeline replay before its tombstone time.

        The FTS row is deliberately preserved — retrieval filters tombstones
        via the main ``records`` table's ``forgotten_at`` column, and keeping
        the FTS row is what lets ``retrieve(filters={"as_of": T})`` still find
        the record for times before it was forgotten.
        """
        now = datetime.datetime.now(datetime.UTC)
        cursor = self._conn.execute(
            "UPDATE records SET forgotten_at=? WHERE record_id=? AND forgotten_at IS NULL",
            (now.isoformat(), record_id),
        )
        was_live = cursor.rowcount > 0
        self._conn.commit()
        self._append_audit(
            "forget", record_id, {"reason": reason, "was_live": was_live}
        )

    def supersede(self, record_id: str, new_value: str) -> MemoryRecord:
        """Create a replacement record; tombstone the old one with a supersede link."""
        old_row = self._conn.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)
        ).fetchone()
        if old_row is None:
            raise KeyError(f"record_id {record_id!r} not found")
        old = row_to_record(old_row)

        now = datetime.datetime.now(datetime.UTC)
        keywords, auto_tags, desc = self._enricher(
            new_value, actor=old.actor, timestamp=now
        )
        new_tags = sorted(set(auto_tags) | set(old.tags))
        embedding_vec = (
            self._embedder.embed(_enrichment_text(new_value, keywords, new_tags, desc))
            if self._embedder
            else np.zeros((0,), dtype=np.float32)
        )

        new = EpisodicRecord(
            record_id=str(uuid.uuid4()),
            content=new_value,
            timestamp=now,
            actor=old.actor,
            keywords=keywords,
            tags=new_tags,
            contextual_description=desc,
            embedding=embedding_vec.tolist(),
            links=[record_id],
            meta={**old.meta, "supersedes": record_id},
            session_id=old.session_id,
            user_id=old.user_id,
        )
        self._conn.execute(
            "UPDATE records SET superseded_by=? WHERE record_id=?",
            (new.record_id, record_id),
        )
        self._upsert_record(new)

        self._append_audit("supersede", record_id, {"new_record_id": new.record_id})
        return new

    def rewrite(self, record_id: str) -> MemoryRecord:
        """Re-run the enrichment and evolution passes against current neighbours."""
        row = self._conn.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"record_id {record_id!r} not found")
        record = row_to_record(row)
        now = datetime.datetime.now(datetime.UTC)

        keywords, auto_tags, desc = self._enricher(
            record.content, actor=record.actor, timestamp=record.timestamp
        )
        record.keywords = keywords
        record.tags = sorted(set(auto_tags) | set(record.tags))
        record.contextual_description = desc

        if self._embedder is not None:
            record.embedding = self._embedder.embed(
                _enrichment_text(record.content, record.keywords, record.tags, desc)
            ).tolist()

        neighbors = self._top_k_neighbors(
            record,
            k=self._neighbor_k,
            session_id=record.session_id,
            user_id=record.user_id,
            exclude={record_id},
        )
        record.links = self._link_generator(record, neighbors)
        self._upsert_record(record)

        for m in self._evolver(record, neighbors):
            self._upsert_record(m)

        self._append_audit("rewrite", record_id, {"timestamp": now.isoformat()})
        return record

    def compact(self, scope: CompactionScope) -> list[MemoryRecord]:
        """Cluster by ``scope.by`` and emit one LLM-summarized consolidated record per cluster.

        Sources are preserved — not deleted, not tombstoned. Consolidated
        record links capture every source ``record_id`` so the audit chain
        stays intact and ``forget`` on a source is still effective.
        """
        now = datetime.datetime.now(datetime.UTC)
        clauses = ["forgotten_at IS NULL", "superseded_by IS NULL"]
        params: list[Any] = []
        if scope.max_age is not None:
            cutoff = (now - scope.max_age).isoformat()
            clauses.append("timestamp <= ?")
            params.append(cutoff)
        rows = self._conn.execute(
            f"SELECT * FROM records WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        if not rows:
            return []

        clusters: dict[str, list[EpisodicRecord]] = {}
        for row in rows:
            record = row_to_record(row)
            for key in _cluster_keys(record, scope.by):
                clusters.setdefault(key, []).append(record)

        produced: list[MemoryRecord] = []
        for key, members in clusters.items():
            if len(members) < scope.threshold:
                continue
            summary, keywords, tags = self._summarizer(list(members))
            consolidated = EpisodicRecord(
                record_id=str(uuid.uuid4()),
                content=summary,
                timestamp=now,
                actor="compaction",
                keywords=sorted(
                    set(keywords) | {kw for m in members for kw in m.keywords}
                ),
                tags=sorted(
                    set(tags) | {"summary", f"compaction:{scope.by}", f"cluster:{key}"}
                ),
                contextual_description=f"Consolidated {len(members)} records on {scope.by}={key!r}",
                embedding=(
                    self._embedder.embed(summary).tolist() if self._embedder else []
                ),
                links=[m.record_id for m in members],
                meta={
                    "compaction_scope": scope.by,
                    "compaction_key": key,
                    "source_count": len(members),
                },
                session_id=_common_or_none(m.session_id for m in members),
                user_id=_common_or_none(m.user_id for m in members),
            )
            self._upsert_record(consolidated)
            produced.append(consolidated)
            self._append_audit(
                "compact",
                consolidated.record_id,
                {
                    "scope": scope.by,
                    "key": key,
                    "source_ids": list(consolidated.links),
                    "source_count": len(members),
                },
            )
        return produced

    def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch a live (non-tombstoned, non-superseded) record by id."""
        row = self._conn.execute(
            "SELECT * FROM records WHERE record_id=? AND forgotten_at IS NULL AND superseded_by IS NULL",
            (record_id,),
        ).fetchone()
        return row_to_record(row) if row else None

    # ---- internals -------------------------------------------------------

    def _upsert_record(self, record: MemoryRecord) -> None:
        if isinstance(record, EpisodicRecord):
            episodic = record
        else:
            episodic = EpisodicRecord(
                record_id=record.record_id,
                content=record.content,
                timestamp=record.timestamp,
                actor=record.actor,
                keywords=record.keywords,
                tags=record.tags,
                contextual_description=record.contextual_description,
                embedding=record.embedding,
                links=record.links,
                meta=record.meta,
                superseded_by=record.superseded_by,
                session_id=None,
                user_id=None,
            )
        row = record_to_row(episodic)
        self._conn.execute(
            "INSERT OR REPLACE INTO records ("
            "record_id, content, timestamp, actor, session_id, user_id, keywords, tags, "
            "contextual_description, embedding, links, meta, superseded_by, forgotten_at) "
            "VALUES (:record_id, :content, :timestamp, :actor, :session_id, :user_id, "
            ":keywords, :tags, :contextual_description, :embedding, :links, :meta, "
            ":superseded_by, :forgotten_at)",
            row,
        )
        # Reinsert into FTS (delete-then-insert handles updates).
        self._conn.execute(
            "DELETE FROM records_fts WHERE rowid=(SELECT rowid FROM records WHERE record_id=?)",
            (episodic.record_id,),
        )
        rowid = self._conn.execute(
            "SELECT rowid FROM records WHERE record_id=?", (episodic.record_id,)
        ).fetchone()["rowid"]
        self._conn.execute(
            "INSERT INTO records_fts(rowid, content, keywords, tags, contextual_description) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                rowid,
                episodic.content,
                " ".join(episodic.keywords),
                " ".join(episodic.tags),
                episodic.contextual_description,
            ),
        )
        self._conn.commit()

    def _append_audit(
        self, op: str, record_id: str | None, payload: dict[str, Any]
    ) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        self._conn.execute(
            "INSERT INTO audit(op, record_id, timestamp, payload) VALUES (?, ?, ?, ?)",
            (op, record_id, now, json.dumps(payload)),
        )
        self._conn.commit()

    def _candidate_rows(self, filters: dict[str, Any]) -> list[sqlite3.Row]:
        """Rows eligible for retrieval given filter dict."""
        clauses: list[str] = ["superseded_by IS NULL"]
        params: list[Any] = []

        as_of = filters.get("as_of")
        if as_of is not None:
            iso = (
                as_of.isoformat()
                if isinstance(as_of, datetime.datetime)
                else str(as_of)
            )
            clauses.append("timestamp <= ?")
            params.append(iso)
            # Timeline replay: tombstones AFTER as_of don't count.
            clauses.append("(forgotten_at IS NULL OR forgotten_at > ?)")
            params.append(iso)
        else:
            clauses.append("forgotten_at IS NULL")

        for key, col in (
            ("actor", "actor"),
            ("session_id", "session_id"),
            ("user_id", "user_id"),
        ):
            if key in filters and filters[key] is not None:
                clauses.append(f"{col} = ?")
                params.append(filters[key])

        if "since" in filters and filters["since"] is not None:
            since = filters["since"]
            clauses.append("timestamp >= ?")
            params.append(
                since.isoformat()
                if isinstance(since, datetime.datetime)
                else str(since)
            )

        return list(
            self._conn.execute(
                f"SELECT * FROM records WHERE {' AND '.join(clauses)}", params
            )
        )

    def _fts_scores(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        """Run an FTS MATCH and return ``{record_id: score}`` for the candidates."""
        if not candidate_ids:
            return {}
        escaped = _escape_fts_query(query)
        if not escaped:
            return {}
        rows = list(
            self._conn.execute(
                "SELECT records.record_id AS record_id, bm25(records_fts) AS rank "
                "FROM records_fts JOIN records ON records_fts.rowid = records.rowid "
                "WHERE records_fts MATCH ?",
                (escaped,),
            )
        )
        allowed = set(candidate_ids)
        scores: dict[str, float] = {}
        for r in rows:
            rid = r["record_id"]
            if rid not in allowed:
                continue
            raw = r["rank"]
            # bm25() is negative; smaller (more negative) = better. Convert to
            # a non-negative relevance score.
            scores[rid] = max(0.0, -float(raw))
        return scores

    def _top_k_neighbors(
        self,
        record: MemoryRecord,
        *,
        k: int,
        session_id: str | None,
        user_id: str | None,
        exclude: set[str] | None = None,
    ) -> list[MemoryRecord]:
        exclude = exclude or set()
        filters: dict[str, Any] = {}
        if session_id is not None:
            filters["session_id"] = session_id
        if user_id is not None:
            filters["user_id"] = user_id
        rows = self._candidate_rows(filters)
        rows = [r for r in rows if r["record_id"] not in exclude]
        if not rows:
            return []

        query_text = _enrichment_text(
            record.content, record.keywords, record.tags, record.contextual_description
        )
        fts = self._fts_scores(query_text, [r["record_id"] for r in rows])

        q_vec: np.ndarray | None = None
        if self._embedder is not None and record.embedding:
            q_vec = np.asarray(record.embedding, dtype=np.float32)

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = fts.get(row["record_id"], 0.0)
            if q_vec is not None:
                vec = row_embedding(row)
                if vec.size == q_vec.size and vec.size > 0:
                    score = 0.5 * score + 0.5 * float(np.dot(q_vec, vec))
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row_to_record(r) for _, r in scored[:k]]

    @staticmethod
    def _turn_to_text(turn: ContextTurn) -> str:
        parts: list[str] = []
        if turn.model_input is not None:
            parts.append(str(turn.model_input))
        if isinstance(turn.output, ModelOutputThunk) and turn.output.value is not None:
            parts.append(str(turn.output.value))
        return "\n".join(p for p in parts if p)


# ---- default phase callables (zero-dep fallbacks) -------------------------


def _default_enricher(
    content: str,
    *,
    actor: str | None = None,
    timestamp: datetime.datetime | None = None,
) -> tuple[list[str], list[str], str]:
    """Heuristic enricher used when no LLM-backed enricher is injected."""
    import re
    from collections import Counter

    toks = [t.lower() for t in re.findall(r"\w+", content or "") if len(t) > 2]
    counts = Counter(toks)
    return [w for w, _ in counts.most_common(5)], [], (content or "")[:160]


def _default_links(record: MemoryRecord, neighbors: list[MemoryRecord]) -> list[str]:
    return []


def _default_evolver(
    record: MemoryRecord, neighbors: list[MemoryRecord]
) -> list[MemoryRecord]:
    return neighbors


def _default_summarizer(
    records: list[MemoryRecord],
) -> tuple[str, list[str], list[str]]:
    return (
        "\n---\n".join(r.content for r in records),
        sorted({kw for r in records for kw in r.keywords}),
        ["summary"],
    )


# ---- utilities -----------------------------------------------------------


def _enrichment_text(
    content: str, keywords: list[str], tags: list[str], description: str
) -> str:
    return " ".join(
        part
        for part in [content, " ".join(keywords), " ".join(tags), description]
        if part
    )


def _escape_fts_query(query: str) -> str:
    """Quote each whitespace-separated token to keep FTS5 safe from operators."""
    import re

    tokens = re.findall(r"\w+", query)
    if not tokens:
        return ""
    # FTS5 OR between quoted tokens so any token can match.
    return " OR ".join(f'"{t}"' for t in tokens)


def _cluster_keys(record: EpisodicRecord, by: str) -> list[str]:
    """Return cluster keys for a record under a given ``CompactionScope.by``.

    ``user_id`` and ``session_id`` aren't formally enumerated in ``CompactionScope``
    (its ``Literal`` type predates these being first-class columns), but they
    are the natural clustering dimensions for an episodic store so we accept
    them here. Records without the keyed attribute are bucketed under an
    ``__unknown__`` sentinel rather than dropped, so callers see and can debug
    unlabeled records.
    """
    if by == "actor":
        return [record.actor]
    if by == "tag":
        return list(record.tags) or ["__untagged__"]
    if by == "time_bucket":
        return [record.timestamp.strftime("%Y-%m-%d")]
    if by == "entity":
        return list(record.keywords) or ["__no_entity__"]
    if by == "user_id":
        return [record.user_id or "__no_user__"]
    if by == "session_id":
        return [record.session_id or "__no_session__"]
    raise ValueError(f"unknown CompactionScope.by: {by!r}")


def _source_for(record: MemoryRecord) -> MemorySource:
    if "compaction_scope" in record.meta:
        return MemorySource.SUMMARY
    return MemorySource.EPISODIC


def _common_or_none(values):
    """Return the single common value across an iterable, or ``None``."""
    unique = {v for v in values if v is not None}
    if len(unique) == 1:
        return next(iter(unique))
    return None
