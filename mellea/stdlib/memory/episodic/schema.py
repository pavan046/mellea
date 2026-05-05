"""SQLite schema and (de)serialization helpers for ``EpisodicStore``.

The store's durable layout is a single SQLite database per store instance, so
cross-session replay and timeline queries become index scans instead of
linear walks. Three tables:

* ``records`` — one row per episodic note. Columns mirror ``MemoryRecord`` plus
  ``session_id``/``user_id`` for cross-session keying. The A-Mem Zettelkasten
  fields (``keywords``, ``tags``, ``contextual_description``, ``links``) are
  stored as JSON blobs; ``embedding`` as a raw ``float32`` byte buffer.
* ``audit`` — append-only log of write/forget/supersede/rewrite/compact ops.
* ``records_fts`` — FTS5 virtual table over ``content || keywords || tags ||
  contextual_description`` so retrieval has a cheap lexical baseline even when
  no ``Embedder`` is configured.

Zero external dependencies beyond Python's stdlib ``sqlite3``.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ....core.memory import MemoryRecord


@dataclass
class EpisodicRecord(MemoryRecord):
    """A ``MemoryRecord`` keyed by session and user for cross-session replay.

    Two persisted fields beyond the base record:

    Attributes:
        session_id (str | None): The session in which this note was captured.
            ``None`` for system-authored notes (e.g., manual ``remember`` calls
            outside an active session).
        user_id (str | None): The end-user the note is about. Used to isolate
            per-user memory so one user's preferences don't leak into
            another's retrieval.
    """

    session_id: str | None = None
    user_id: str | None = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    session_id TEXT,
    user_id TEXT,
    keywords TEXT NOT NULL,                -- JSON array
    tags TEXT NOT NULL,                    -- JSON array
    contextual_description TEXT NOT NULL,
    embedding BLOB,                        -- float32 bytes, NULL when unset
    links TEXT NOT NULL,                   -- JSON array
    meta TEXT NOT NULL,                    -- JSON object
    superseded_by TEXT,
    forgotten_at TEXT                      -- tombstone; NULL means live
);

CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp);
CREATE INDEX IF NOT EXISTS idx_records_actor ON records(actor);

CREATE TABLE IF NOT EXISTS audit (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    op TEXT NOT NULL,
    record_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL                   -- JSON object
);
CREATE INDEX IF NOT EXISTS idx_audit_record ON audit(record_id);

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    content,
    keywords,
    tags,
    contextual_description
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes idempotently."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def connect(db_path: str) -> sqlite3.Connection:
    """Open a tuned SQLite connection for the episodic store.

    Enables ``WAL`` for better concurrent read/write ergonomics and
    ``foreign_keys`` (defensive; the schema doesn't use them yet but will
    under D's graph layer).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    init_schema(conn)
    return conn


def _encode_vec(vec: np.ndarray | None) -> bytes | None:
    if vec is None or len(vec) == 0:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _decode_vec(buf: bytes | None) -> np.ndarray:
    if not buf:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(buf, dtype=np.float32).copy()


def record_to_row(record: EpisodicRecord) -> dict[str, Any]:
    """Flatten a record into the dict passed to an INSERT/REPLACE."""
    return {
        "record_id": record.record_id,
        "content": record.content,
        "timestamp": record.timestamp.isoformat(),
        "actor": record.actor,
        "session_id": record.session_id,
        "user_id": record.user_id,
        "keywords": json.dumps(record.keywords),
        "tags": json.dumps(record.tags),
        "contextual_description": record.contextual_description,
        "embedding": _encode_vec(
            np.asarray(record.embedding, dtype=np.float32) if record.embedding else None
        ),
        "links": json.dumps(record.links),
        "meta": json.dumps(record.meta),
        "superseded_by": record.superseded_by,
        "forgotten_at": None,
    }


def row_to_record(row: sqlite3.Row) -> EpisodicRecord:
    """Inflate a SELECT row back into an EpisodicRecord."""
    return EpisodicRecord(
        record_id=row["record_id"],
        content=row["content"],
        timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
        actor=row["actor"],
        keywords=list(json.loads(row["keywords"])),
        tags=list(json.loads(row["tags"])),
        contextual_description=row["contextual_description"],
        embedding=_decode_vec(row["embedding"]).tolist(),
        links=list(json.loads(row["links"])),
        meta=dict(json.loads(row["meta"])),
        superseded_by=row["superseded_by"],
        session_id=row["session_id"],
        user_id=row["user_id"],
    )


def row_embedding(row: sqlite3.Row) -> np.ndarray:
    """Return the embedding from a row without inflating the whole record."""
    return _decode_vec(row["embedding"])


@dataclass
class AuditEntry:
    """A single entry in the audit log table.

    Attributes:
        rowid (int): Monotonically increasing row id assigned by SQLite.
        op (str): Operation name (``write``, ``forget``, ``supersede``,
            ``rewrite``, ``compact``).
        record_id (str | None): Target record id, or ``None`` for ops that
            don't target a specific record.
        timestamp (datetime.datetime): When the op was recorded.
        payload (dict[str, Any]): Op-specific structured payload.
    """

    rowid: int
    op: str
    record_id: str | None
    timestamp: datetime.datetime
    payload: dict[str, Any] = field(default_factory=dict)


def row_to_audit(row: sqlite3.Row) -> AuditEntry:
    """Inflate an audit-table row into an ``AuditEntry``."""
    return AuditEntry(
        rowid=row["rowid"],
        op=row["op"],
        record_id=row["record_id"],
        timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
        payload=dict(json.loads(row["payload"])),
    )
