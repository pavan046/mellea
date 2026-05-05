"""Vector-backed external memory store.

``VectorStore`` is the RAG-as-memory concrete ``MemoryStore``: it ingests raw
documents (via ``chunker.chunk_text``), embeds each chunk (via any
``Embedder`` protocol implementation), and retrieves by cosine similarity.
Each emitted ``MemoryBlock`` carries ``source=EXTERNAL_DOC`` plus enough
metadata (``doc_id``, ``chunk_index``, ``start``, ``end``) for the existing
RAG intrinsics (``find_citations``, ``flag_hallucinated_content``) to cite
back into the source document.

This is workstream B's primary deliverable. Workstream C's ``EpisodicStore``
will reuse the same embedding plumbing for semantic similarity over
episodic notes; workstream D's ``GraphStore`` will add relational queries.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ...core.base import ContextTurn, MemoryBlock, MemorySource, ModelOutputThunk
from ...core.memory import CompactionScope, MemoryRecord, MemoryStore
from .chunker import Chunk, chunk_text
from .embedders import Embedder, HashingEmbedder


class VectorStore(MemoryStore):
    """External-memory store backed by a dense vector index.

    Ingests raw documents via ``add_document`` (or from-turn writes via the
    ``write`` ABC verb), splits them into overlapping chunks, embeds each
    chunk, and retrieves by cosine similarity. Retrieval emits
    ``MemoryBlock`` instances with ``source=EXTERNAL_DOC`` and ``meta``
    fields (``doc_id``, ``doc_title``, ``chunk_index``, ``start``, ``end``,
    ``char_len``) so downstream citation intrinsics can map back to the
    source document.

    ``forget`` drops an individual chunk; ``forget_document`` drops every
    chunk belonging to a ``doc_id``. ``supersede`` replaces a chunk's text
    and re-embeds it. ``rewrite`` re-embeds without content change.
    ``compact`` is a no-op for the vector store (external documents don't
    benefit from the entity/tag clustering that episodic stores do); callers
    who want coarser granularity should adjust ``window_size`` at ingest.

    Args:
        store_id (str): Stable identifier embedded in every emitted block.
        embedder (Embedder | None): Embedding backend. Defaults to
            ``HashingEmbedder()`` so tests work out of the box.
        window_size (int): Chunk size in whitespace tokens. Defaults to 200.
        overlap (int): Chunk overlap in whitespace tokens. Defaults to 40.
        write_on_turn (bool): Whether ``MelleaSession`` should auto-write
            every turn into this store. Defaults to ``False`` — document
            ingestion is typically a separate pipeline.
    """

    def __init__(
        self,
        store_id: str = "vector",
        *,
        embedder: Embedder | None = None,
        window_size: int = 200,
        overlap: int = 40,
        write_on_turn: bool = False,
    ):
        """Initialize VectorStore with an embedder and chunker configuration."""
        super().__init__(store_id, write_on_turn=write_on_turn)
        self._embedder: Embedder = embedder or HashingEmbedder()
        self._window_size = window_size
        self._overlap = overlap
        self._records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._doc_index: dict[str, list[str]] = {}  # doc_id -> record_ids (in order)
        self._audit: list[dict[str, Any]] = []

    # ---- public ingestion helpers ----

    def add_document(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        actor: str = "ingest",
        tags: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Chunk, embed, and persist a document.

        Args:
            text (str): Raw document text.
            doc_id (str | None): Stable id for the source document. Defaults to
                a generated UUID. Reusing an existing ``doc_id`` appends chunks;
                use ``forget_document`` first if you need a clean replace.
            title (str | None): Optional human-readable title stored in meta.
            actor (str): Who is ingesting this document. Defaults to ``"ingest"``.
            tags (dict[str, Any] | None): Metadata attached to every chunk.

        Returns:
            list[MemoryRecord]: One record per emitted chunk, in document order.
        """
        if not text.strip():
            return []
        if doc_id is None:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"

        chunks = chunk_text(text, window_size=self._window_size, overlap=self._overlap)
        if not chunks:
            return []

        vectors = self._embedder.embed_batch([c.text for c in chunks])
        now = datetime.datetime.now(datetime.UTC)
        produced: list[MemoryRecord] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            record = self._persist_chunk(
                chunk=chunk,
                vector=vec,
                doc_id=doc_id,
                title=title,
                actor=actor,
                tags=tags,
                timestamp=now,
            )
            produced.append(record)

        self._audit.append(
            {
                "op": "add_document",
                "doc_id": doc_id,
                "chunk_count": len(produced),
                "timestamp": now.isoformat(),
            }
        )
        return produced

    def forget_document(self, doc_id: str, *, reason: str = "doc_removed") -> int:
        """Remove every chunk belonging to ``doc_id``. Returns count removed."""
        chunk_ids = list(self._doc_index.pop(doc_id, []))
        for cid in chunk_ids:
            self._records.pop(cid, None)
            self._vectors.pop(cid, None)
        self._audit.append(
            {
                "op": "forget_document",
                "doc_id": doc_id,
                "chunk_count": len(chunk_ids),
                "reason": reason,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        return len(chunk_ids)

    # ---- MemoryStore ABC ----

    def write(
        self, turn: ContextTurn, *, actor: str, tags: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        """Write a conversational turn as if it were a document.

        Rarely used directly — the normal RAG path ingests documents via
        ``add_document``. Provided so the ``VectorStore`` satisfies the
        ``MemoryStore`` ABC and can serve as a quick-and-dirty episodic
        store for scripts that don't need A-Mem evolution.
        """
        content = self._turn_to_text(turn)
        if not content.strip():
            return []
        doc_id = (tags or {}).get("doc_id") or f"turn-{uuid.uuid4().hex[:8]}"
        return self.add_document(
            content,
            doc_id=doc_id,
            title=(tags or {}).get("title"),
            actor=actor,
            tags=tags,
        )

    def retrieve(
        self, query: str, *, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[MemoryBlock]:
        """Top-k cosine similarity over the embedded chunks."""
        if not query.strip() or not self._vectors:
            return []
        q_vec = self._embedder.embed(query)

        eligible_ids = [
            rid
            for rid, rec in self._records.items()
            if rec.superseded_by is None and self._passes(rec, filters)
        ]
        if not eligible_ids:
            return []

        matrix = np.stack([self._vectors[rid] for rid in eligible_ids], axis=0)
        scores = matrix @ q_vec  # both pre-normalized → dot product == cosine

        top_k = min(k, len(eligible_ids))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        now = datetime.datetime.now(datetime.UTC)
        blocks: list[MemoryBlock] = []
        for idx in top_idx:
            rid = eligible_ids[int(idx)]
            rec = self._records[rid]
            score = float(scores[int(idx)])
            if score <= 0.0:
                continue
            blocks.append(
                MemoryBlock(
                    rec.content,
                    source=MemorySource.EXTERNAL_DOC,
                    store_id=self._store_id,
                    record_id=rec.record_id,
                    retrieved_at=now,
                    score=score,
                    meta={
                        "doc_id": rec.meta.get("doc_id"),
                        "doc_title": rec.meta.get("doc_title"),
                        "chunk_index": rec.meta.get("chunk_index"),
                        "start": rec.meta.get("start"),
                        "end": rec.meta.get("end"),
                        "actor": rec.actor,
                    },
                )
            )
        return blocks

    def forget(self, record_id: str, *, reason: str) -> None:
        """Drop a single chunk from the index and re-compact the doc_index."""
        rec = self._records.pop(record_id, None)
        self._vectors.pop(record_id, None)
        if rec is not None:
            doc_id = rec.meta.get("doc_id")
            if doc_id and doc_id in self._doc_index:
                self._doc_index[doc_id] = [
                    rid for rid in self._doc_index[doc_id] if rid != record_id
                ]
                if not self._doc_index[doc_id]:
                    self._doc_index.pop(doc_id, None)
        self._audit.append(
            {
                "op": "forget",
                "record_id": record_id,
                "reason": reason,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "was_present": rec is not None,
            }
        )

    def supersede(self, record_id: str, new_value: str) -> MemoryRecord:
        """Replace a chunk's content and re-embed it."""
        if record_id not in self._records:
            raise KeyError(f"record_id {record_id!r} not found")
        old = self._records[record_id]
        vec = self._embedder.embed(new_value)
        new = MemoryRecord(
            record_id=str(uuid.uuid4()),
            content=new_value,
            timestamp=datetime.datetime.now(datetime.UTC),
            actor=old.actor,
            keywords=list(old.keywords),
            tags=list(old.tags),
            contextual_description=old.contextual_description,
            embedding=vec.tolist(),
            links=[record_id],
            meta={**old.meta, "supersedes": record_id},
        )
        old.superseded_by = new.record_id
        self._records[new.record_id] = new
        self._vectors[new.record_id] = vec
        doc_id = new.meta.get("doc_id")
        if doc_id and doc_id in self._doc_index:
            self._doc_index[doc_id].append(new.record_id)
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
        """Re-embed a record against the current embedder state."""
        if record_id not in self._records:
            raise KeyError(f"record_id {record_id!r} not found")
        rec = self._records[record_id]
        vec = self._embedder.embed(rec.content)
        rec.embedding = vec.tolist()
        self._vectors[record_id] = vec
        self._audit.append(
            {
                "op": "rewrite",
                "record_id": record_id,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        return rec

    def compact(self, scope: CompactionScope) -> list[MemoryRecord]:
        """No-op for vector stores; returns ``[]``.

        Re-chunking at a coarser granularity is done at ingest-time by
        tuning ``window_size``, not at rollup time. Episodic/graph stores
        own meaningful compaction.
        """
        self._audit.append(
            {
                "op": "compact",
                "scope": scope.by,
                "produced": 0,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "note": "vector_store_noop",
            }
        )
        return []

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return a record by id, or ``None`` if tombstoned/missing."""
        rec = self._records.get(record_id)
        if rec is None or rec.superseded_by is not None:
            return None
        return rec

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        """Persist the store to a directory: ``records.json`` + ``vectors.npz``.

        The embedder itself is NOT persisted; callers MUST pass the same
        ``Embedder`` (or one produced by the same model) to ``load`` or the
        re-loaded vectors will be meaningless.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        records_payload = {
            rid: {
                "record_id": rec.record_id,
                "content": rec.content,
                "timestamp": rec.timestamp.isoformat(),
                "actor": rec.actor,
                "keywords": rec.keywords,
                "tags": rec.tags,
                "contextual_description": rec.contextual_description,
                "links": rec.links,
                "meta": rec.meta,
                "superseded_by": rec.superseded_by,
            }
            for rid, rec in self._records.items()
        }
        (target / "records.json").write_text(
            json.dumps(
                {
                    "store_id": self._store_id,
                    "records": records_payload,
                    "doc_index": self._doc_index,
                    "embedder_dim": self._embedder.dim,
                    "window_size": self._window_size,
                    "overlap": self._overlap,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if self._vectors:
            np.savez_compressed(
                target / "vectors.npz",
                ids=np.array(list(self._vectors.keys()), dtype=object),
                matrix=np.stack(list(self._vectors.values()), axis=0),
            )

    @classmethod
    def load(cls, path: str | Path, *, embedder: Embedder) -> VectorStore:
        """Load a store previously saved with ``save``.

        Args:
            path (str | Path): Directory produced by ``save``.
            embedder (Embedder): Must be compatible with the embedder used at
                save time; ``dim`` must match. A mismatch raises ``ValueError``.

        Returns:
            VectorStore: A fresh instance populated from disk.
        """
        source = Path(path)
        payload = json.loads((source / "records.json").read_text(encoding="utf-8"))
        if embedder.dim != payload["embedder_dim"]:
            raise ValueError(
                f"embedder dim {embedder.dim} does not match saved dim "
                f"{payload['embedder_dim']}"
            )

        store = cls(
            store_id=payload["store_id"],
            embedder=embedder,
            window_size=payload["window_size"],
            overlap=payload["overlap"],
        )
        for rid, raw in payload["records"].items():
            store._records[rid] = MemoryRecord(
                record_id=raw["record_id"],
                content=raw["content"],
                timestamp=datetime.datetime.fromisoformat(raw["timestamp"]),
                actor=raw["actor"],
                keywords=list(raw["keywords"]),
                tags=list(raw["tags"]),
                contextual_description=raw["contextual_description"],
                embedding=[],
                links=list(raw["links"]),
                meta=dict(raw["meta"]),
                superseded_by=raw.get("superseded_by"),
            )
        store._doc_index = {k: list(v) for k, v in payload["doc_index"].items()}

        vec_path = source / "vectors.npz"
        if vec_path.exists():
            data = np.load(vec_path, allow_pickle=True)
            ids = list(data["ids"])
            matrix = data["matrix"]
            for i, rid in enumerate(ids):
                store._vectors[str(rid)] = matrix[i].astype(np.float32)
        return store

    # ---- inspection ----

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Snapshot of audit entries."""
        return list(self._audit)

    def doc_ids(self) -> list[str]:
        """Stable list of ingested document ids."""
        return list(self._doc_index.keys())

    def chunks_for(self, doc_id: str) -> list[MemoryRecord]:
        """All non-tombstoned chunks belonging to ``doc_id``, in document order."""
        return [
            self._records[rid]
            for rid in self._doc_index.get(doc_id, [])
            if rid in self._records and self._records[rid].superseded_by is None
        ]

    # ---- internals ----

    def _persist_chunk(
        self,
        *,
        chunk: Chunk,
        vector: np.ndarray,
        doc_id: str,
        title: str | None,
        actor: str,
        tags: dict[str, Any] | None,
        timestamp: datetime.datetime,
    ) -> MemoryRecord:
        record = MemoryRecord(
            record_id=str(uuid.uuid4()),
            content=chunk.text,
            timestamp=timestamp,
            actor=actor,
            keywords=[],
            tags=sorted((tags or {}).keys()),
            contextual_description=(title or "")[:160],
            embedding=vector.tolist(),
            links=[],
            meta={
                **(tags or {}),
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "char_len": chunk.end - chunk.start,
            },
        )
        self._records[record.record_id] = record
        self._vectors[record.record_id] = vector.astype(np.float32)
        self._doc_index.setdefault(doc_id, []).append(record.record_id)
        return record

    @staticmethod
    def _turn_to_text(turn: ContextTurn) -> str:
        parts: list[str] = []
        if turn.model_input is not None:
            parts.append(str(turn.model_input))
        if isinstance(turn.output, ModelOutputThunk) and turn.output.value is not None:
            parts.append(str(turn.output.value))
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _passes(record: MemoryRecord, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        if "actor" in filters and record.actor != filters["actor"]:
            return False
        if "doc_id" in filters and record.meta.get("doc_id") != filters["doc_id"]:
            return False
        if "tag" in filters and filters["tag"] not in record.tags:
            return False
        return True
