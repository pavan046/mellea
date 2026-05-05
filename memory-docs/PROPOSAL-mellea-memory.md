# Proposal: First-Class Memory for Mellea

**Status:** Draft v0.1
**Team size:** 4 engineers (one per workstream)
**Duration (estimate):** 12 weeks to GA-quality v1
**Primary integration points:** `mellea.core.base.Context`, `mellea.core.base.CBlock`, `mellea.stdlib.context`, `mellea.stdlib.session`
**Grounding papers:**
- Zhang et al., *Memory in Large Language Models: Mechanisms, Evaluation and Evolution*, arXiv:2509.18868 (the "memory quadruple" taxonomy)
- Xu et al., *A-Mem: Agentic Memory for LLM Agents*, arXiv:2502.12110 (Zettelkasten-style notes, link generation, memory evolution)
- Mem0 (`mem0ai/mem0`) — reference for the knowledge-graph memory layer

---

## 1. Motivation

Mellea today has exactly one durable state-carrier: `Context`, an immutable linked list of `CBlock`/`Component` nodes threaded through a `MelleaSession`. That structure is sufficient for a single conversation — `ChatContext` slides a window, `SimpleContext` stays stateless — but it collapses three very different notions of state into one wire:

1. the **KV-shaped "what the model can currently see"** (contextual memory),
2. the **"what this user/agent has learned across sessions"** (episodic + semantic memory), and
3. the **"what tools, retrieval indices, and documents back this generation"** (external memory).

Zhang et al. (2509.18868) show that conflating these is not merely inelegant — it produces the *dual-identity problem* in RAG (documents returned by a retriever are serialized into the context window and thereby *become* contextual memory), it makes auditability impossible ("did this answer come from weights, from history, or from a retrieved document?"), and it blocks the write/read/inhibit governance chain that the paper argues is the correct operational contract.

Mellea is in an unusually good position to fix this cleanly. `Context` is already abstract and immutable, `CBlock` is already a tagged value with `meta`, and `MelleaSession` already owns the read path into the backend. We can introduce a principled memory layer without breaking a single existing caller.

## 2. What kind of memory are we actually building?

The paper defines LLM memory as **"persistent, addressable state that can be written during pretraining, fine-tuning, or inference, can be later read, and stably influences outputs."** It decomposes this into a *memory quadruple* `(storage location, persistence, access path, controllability)` and four carrier types:

| Carrier | Storage | Persistence | Where it lives in mellea today |
| --- | --- | --- | --- |
| **Parametric** | model weights | long-term | backend weights — **out of scope** for this proposal (model editing is its own project) |
| **Contextual** | KV cache / prompt | transient, per-call | `Context` + `CBlock`s → `view_for_generation()` |
| **External (non-parametric)** | retrieval index, doc/vector store, tools | updatable, auditable | currently ad-hoc — lives in user code or `stdlib/components` |
| **Procedural / episodic** | event store | session- and cross-session level | **does not exist** |

**Explicit non-goal:** we are not doing parametric memory (ROME/MEND/MEMIT-style in-weight edits). The paper treats this as a separate governance track, and so do we.

**Explicit targets:**

1. **External memory** — give mellea a first-class retrieval + attribution surface so RAG is not glued onto the prompt at the user's discretion.
2. **Procedural/episodic memory** — give mellea cross-session persistence: user preferences, agent observations, task timelines. This is the piece that is completely missing today, and it is exactly what Mem0 solves in its own ecosystem.
3. **Contextual memory as the *interface*** — `Context` stays the thing the backend sees, but it is now the *projection* of the memory subsystem onto the current turn, not the substrate itself.

This matches the paper's `write → read → inhibit/update` causal chain: memory is *written* (new episodic fact, new document), *read* (retrieved into the context projection), and *inhibited/updated* (forget, rewrite, supersede). We will build all three verbs.

## 3. Design sketch — how it bolts onto `CBlock` and `Context`

### 3.1 `MemoryBlock` — a `CBlock` that remembers where it came from

A new `CBlock` subclass (peer of the existing `ImageBlock`):

```python
class MemoryBlock(CBlock):
    """A CBlock that carries provenance and a pointer back to the memory store
    that materialized it. Used when memory content enters the context window."""

    def __init__(
        self,
        value: str,
        *,
        source: MemorySource,     # enum: EPISODIC | SEMANTIC | EXTERNAL_DOC | GRAPH_FACT | SUMMARY
        store_id: str,            # which MemoryStore produced this
        record_id: str,           # stable id inside that store — enables inhibit/update
        score: float | None = None,
        retrieved_at: datetime,
        meta: dict[str, Any] | None = None,
    ):
        ...
```

This is deliberately narrow: a `MemoryBlock` is still just a string as far as the backend is concerned, but it is *attributable*. The `meta` dict already supported by `CBlock` carries the structured fields, and `__repr__` surfaces them for tracing. This directly answers the paper's Layer-2 *attribution/faithfulness* metric (§4.3 of 2509.18868) — every token that came from memory is traceable to a record id.

### 3.2 `MemoryStore` — the abstract backend

Mirrors how mellea already treats backends: one protocol, several implementations.

```python
@dataclass
class MemoryRecord:
    """The stored unit. Structure follows A-Mem's Zettelkasten note
    (Xu et al., 2502.12110 §3.1 eq. 1): content + LLM-derived semantic
    components + links. Richer than a flat (id, text, vec) row so that
    memory evolution has structured fields to rewrite."""

    record_id: str
    content: str                         # c_i — original interaction content
    timestamp: datetime                  # t_i
    keywords: list[str]                  # K_i — LLM-extracted key concepts
    tags: list[str]                      # G_i — LLM-assigned categorization
    contextual_description: str          # X_i — LLM-generated semantic gloss
    embedding: np.ndarray                # e_i — over concat(c, K, G, X)
    links: list[str]                     # L_i — record_ids of linked notes
    actor: str
    meta: dict[str, Any]
    superseded_by: str | None = None     # tombstone pointer, if any


class MemoryStore(abc.ABC):
    # Write — three-phase per A-Mem §3.1–§3.3:
    #   (1) note construction:  K, G, X ← LLM(c ‖ t ‖ P_s1)        [eq. 2]
    #   (2) link generation:    L      ← LLM(m_n ‖ M_near ‖ P_s2)  [eq. 6]
    #   (3) memory evolution:   for m_j ∈ M_near, m_j* ← LLM(...)  [eq. 7]
    # Phase (3) is sync-on-write by default (see §5.1); stores MAY offer
    # an async mode, but retrieval consistency is the sync contract.
    @abc.abstractmethod
    def write(self, turn: ContextTurn, *, actor: str, tags: dict) -> list[MemoryRecord]: ...

    # Read — returns MemoryBlocks ready to splice into the context view
    @abc.abstractmethod
    def retrieve(self, query: str, *, k: int, filters: dict) -> list[MemoryBlock]: ...

    # Inhibit / update — the forget verb the paper insists on
    @abc.abstractmethod
    def forget(self, record_id: str, *, reason: str) -> None: ...
    @abc.abstractmethod
    def supersede(self, record_id: str, new_value: str) -> MemoryRecord: ...

    # Re-evolve a record against its current neighbors. Internal to write()
    # by default; exposed publicly so prompt changes or schema migrations
    # can trigger batch re-runs without a full re-ingest.
    @abc.abstractmethod
    def rewrite(self, record_id: str) -> MemoryRecord: ...

    # Consolidation — see §3.5. Clusters related records in `scope` and
    # emits summary records with provenance links back to the originals.
    # Distinct from evolution: evolution refines individual notes,
    # compaction produces new notes that subsume many.
    @abc.abstractmethod
    def compact(self, scope: CompactionScope) -> list[MemoryRecord]: ...
```

Concrete stores shipped in v1:
- `InMemoryStore` — reference impl, for tests and small scripts.
- `VectorStore(backend=...)` — semantic retrieval; thin adapter on top of whichever vector DB (we copy mem0's adapter list rather than re-litigating).
- `EpisodicStore` — LLM-extraction-driven fact writer (turn → structured facts), the bread and butter of Mem0; see §5.
- `GraphStore` — see §6 (knowledge-graph memory).

### 3.3 `MemoryContext` — a `Context` that consults memory on each view

```python
class MemoryContext(ChatContext):
    """A ChatContext that, on view_for_generation(), splices in MemoryBlocks
    retrieved from one or more MemoryStores using the current turn as the query."""

    def __init__(self, *, stores: list[MemoryStore], policy: RetrievalPolicy,
                 window_size: int | None = None):
        ...

    def view_for_generation(self) -> list[Component | CBlock] | None:
        history = super().view_for_generation()         # contextual memory
        query = self._derive_query(history)
        recalled = self._policy.retrieve(query, self._stores)   # external + episodic
        return self._policy.splice(history, recalled)          # merged projection
```

This is the single cleanest integration point in the whole proposal: **`MemoryContext` is a drop-in replacement for `ChatContext`**. Any existing mellea program becomes memory-aware by swapping one class. The immutability contract of `Context` is preserved (each `.add()` still returns a new node); the stores live *outside* the linked list, which is correct because they outlive it.

`RetrievalPolicy` encapsulates the paper's §4.3 layering: (a) retrieval-quality selection (Recall@k, nDCG), (b) attribution-aware splicing (MemoryBlocks are marked, not laundered into plain text), (c) positional placement choices (the paper's "mid-sequence drop" — *where* in the window you put the recall matters).

### 3.4 Session-level hooks

`MelleaSession` already owns the turn lifecycle. We add two hooks:
- **after-turn write:** when a turn completes, the session calls `store.write(turn, ...)` for every attached store with `write_on_turn=True`. This is how episodic facts get captured without user bookkeeping.
- **explicit verbs on the session:** `m.remember(fact)`, `m.forget(record_id)`, `m.recall(query)` — thin sugar over the stores, useful for agent-authored memory (mem0's v3 "agent-generated facts are first-class" insight applies directly).

### 3.5 Compaction & consolidation — the verb A-Mem doesn't have

A-Mem evolves individual notes but never condenses many into fewer or drops stale ones. That gap matters for two reasons: the contextual window is finite (short-term pressure), and an episodic store that only grows becomes retrieval-dilute over months (long-term pressure). We add a dedicated compaction path, orthogonal to evolution.

Three regimes, one verb:

1. **Short-term compaction (window-side).** When `MemoryContext.view_for_generation()` would exceed a turn budget, older turns are folded into a synthetic `MemoryBlock(source=SUMMARY)` rather than dropped. The summary carries `record_id`s of the turns it subsumes so the projection stays auditable. A `CompactionPolicy` is the peer of `RetrievalPolicy` on `MemoryContext`; defaults to "summarize the oldest 25% of turns once the budget is exceeded, cap summaries at N tokens."

2. **Long-term rollup (store-side).** `MemoryStore.compact(scope)` clusters related records in `scope` (by entity, by time bucket, by tag) and emits consolidated records. Each consolidated record links back to its sources — the originals are *not* deleted, only demoted in retrieval ranking. This preserves the audit trail the paper (2509.18868 §4.3) demands and lets `forget()` still operate on the primary records. Consolidated records can themselves be evolved (§5.1) since they are just notes with richer provenance.

3. **Decay.** An age-weighted down-ranking in `RetrievalPolicy`, not a deletion pass. Stale records stay recallable but drop in the retrieval score unless reinforced by access. The paper's inhibit verb is `forget()`; decay is the softer counterpart.

```python
@dataclass
class CompactionScope:
    by: Literal["entity", "time_bucket", "tag", "actor"]
    threshold: int              # min cluster size before we bother
    max_age: timedelta | None   # only compact records older than this

class CompactionPolicy(abc.ABC):
    """Window-side compaction, consulted by MemoryContext.view_for_generation()."""
    @abc.abstractmethod
    def should_compact(self, history: list[CBlock], budget: int) -> bool: ...
    @abc.abstractmethod
    def compact(self, history: list[CBlock]) -> list[CBlock]: ...  # yields SUMMARY blocks
```

Compaction and evolution compose but do not replace each other: evolution rewrites a note's `K, G, X` in light of new neighbors; compaction emits a new note that subsumes many. A compaction pass over a freshly-evolved store is strictly more informative than over an un-evolved one, so the default pipeline order is write → evolve → (later, scheduled) compact.

## 4. Team decomposition — one workstream per engineer

The design decomposes along clean interface lines, so four people can work in parallel from week 2 onward. Week 1 is a joint spike to lock the interfaces below.

### Workstream A — Core contracts (`mellea.core.memory`)
**Owner:** 1 engineer.
**Deliverables:** `MemoryBlock`, `MemoryStore` ABC, `MemoryRecord`, `MemoryContext`, `RetrievalPolicy`, `InMemoryStore` reference impl, full type coverage, unit tests. Touches `core/base.py` to add `MemoryBlock` alongside `ImageBlock`, and `stdlib/context.py` to add `MemoryContext` alongside `ChatContext`/`SimpleContext`.
**Why isolatable:** pure library code, no backend dependencies. Everyone else depends on this interface, nothing else depends on them.

### Workstream B — External memory (RAG-as-memory)
**Owner:** 1 engineer.
**Deliverables:** `VectorStore` with pluggable embedding backends (reuse mellea's backend config where possible), ingestion CLI (`m memory ingest`), document chunker (already exists in `stdlib/chunking.py`), attribution glue that marks each retrieved chunk with a citation id the IVR pipeline can surface. Explicit design target: pass the paper's Layer-1 (Recall@k) and Layer-2 (Citation Precision/Recall) metrics on a held-out eval. Integrate with the existing RAG intrinsics in `mellea/stdlib/components/intrinsic/rag.py` (check_context_relevance, find_citations, flag_hallucinated_content) — memory now *produces* the context those intrinsics *validate*.
**Why isolatable:** depends only on the A interfaces.

### Workstream C — Episodic / procedural memory + compaction
**Owner:** 1 engineer.
**Deliverables:** `EpisodicStore` implementing the A-Mem three-phase write (note construction → link generation → evolution, §3.2 and §5.1): LLM-extraction over each `ContextTurn` produces a structured note `(c, t, K, G, X, e, L)` per A-Mem §3.1, links are generated against the top-k neighbors (§3.2), and evolution rewrites those neighbors' `K, G, X` fields in place (§3.3). Forget/supersede backed by tombstones; cross-session persistence (session_id + user_id keys); timeline replay for `E-MARS+`. Compaction belongs here too: `CompactionPolicy` for the window side and `compact(scope)` for store-side rollup (§3.5). This is where mellea gains Mem0's *"remembers user preferences, adapts over time"* capability — and, via A-Mem evolution, exceeds Mem0's static-after-write model.
**Why isolatable:** depends on A; orthogonal to B (different store, different retrieval path).

### Workstream D — Knowledge-graph memory + governance
**Owner:** 1 engineer.
**Deliverables:** `GraphStore` — subclasses `EpisodicStore` and projects its `links` field as graph edges, pluggable graph backend (Neo4j / Kuzu / Neptune — cite Mem0's adapter set). Governance surface: audit log for every write/forget/supersede/rewrite/compact, the paper's "edit/forgetting certificate" (2509.18868 §5), temporal queries ("what did the system believe about X as of date D?"). Evaluation harness: wire up LoCoMo and LongMemEval so we can report both Mem0-comparable and A-Mem-comparable numbers, including A-Mem's LG/ME ablation (Table 3) to verify our evolution pass is actually load-bearing.
**Why isolatable:** depends on C's `EpisodicStore` and its evolved `links` field, but the graph layer and governance are otherwise standalone.

### Shared scaffolding (week 1, everyone)
- Lock the `MemoryStore` / `MemoryBlock` / `MemoryRecord` schemas (note the A-Mem-shaped record: `c, t, K, G, X, e, L`).
- Agree on the `RetrievalPolicy.splice()` and `CompactionPolicy.compact()` contracts.
- Pick the three A-Mem prompts (`P_s1` note construction, `P_s2` link generation, `P_s3` evolution) and their Granite adapters.
- Decide on the session hook API and the sync-vs-async evolution default (per §5.1, lean sync).

## 5. Knowledge-graph memory — the icing

The paper positions procedural/episodic memory as the carrier that "emphasizes temporal structure and re-playability and often shares infrastructure with external memory in practice." A knowledge graph is the natural shared substrate: entities (people, tasks, artifacts) as nodes, observed relations (preferences, ownership, dependencies, supersessions) as edges, each edge timestamped and reversible.

**We adopt Mem0's architectural pattern here and cite it explicitly.** Mem0 ships a `graph_store` abstraction in `mem0/mem0/memory/` with backends for Neo4j, Kuzu, and AWS Neptune (see `examples/misc/strands_agent_aws_elasticache_neptune.py` for the reference usage, and the `kuzu`/`graph_store` feature gating in `mem0/exceptions.py:396`). Mem0's v3 algorithm makes entities first-class — "entities are extracted, embedded, and linked across memories for retrieval boosting" — and reports +53.6 points on assistant memory recall relative to its v2.

Mellea's `GraphStore` mirrors this pattern but integrates with `MemoryBlock` and `Context` rather than Mem0's `Memory` class:

```python
class GraphStore(MemoryStore):
    """A graph-structured episodic memory. Nodes are entities; edges are
    timestamped, reversible relations. Retrieval fuses entity match + vector
    similarity + BM25, following Mem0's multi-signal fusion."""

    def write(self, turn: ContextTurn, ...):
        facts = self._extract(turn)               # LLM extraction → (subj, rel, obj, t)
        for f in facts:
            self._link_entities(f)                # entity resolution against existing nodes
            self._upsert_edge(f)                  # idempotent, append-only with supersession

    def retrieve(self, query: str, *, k: int, ...) -> list[MemoryBlock]:
        entities = self._extract_entities(query)
        hits = self._fuse(
            self._entity_walk(entities, depth=2),
            self._vector_search(query, k=k),
            self._bm25(query, k=k),
        )
        return [self._to_memory_block(h) for h in hits]
```

Three properties fall out of the graph formulation that pure vector memory does not give us:

1. **Relational queries the paper's procedural-memory evaluation requires** — "what has the user said about X over time?" becomes a bounded entity walk, not a nearest-neighbor approximation.
2. **Auditable supersession** — the paper's inhibit/update verb maps to edge tombstones with a replacement pointer; `m.forget()` and `m.supersede()` become graph operations, not store-specific hacks.
3. **Mem0-parity benchmarks** — we can compare head-to-head on LoCoMo / LongMemEval, which is the only credible way to claim the feature works.

### 5.1 Memory evolution — the A-Mem pass

A-Mem (Xu et al., 2502.12110) introduces the observation that *writing a new memory should trigger updates to related old memories*, not merely forward links. The arrival of note `m_n` can refine the contextual description, keywords, and tags of existing neighbors `m_j ∈ M_near`:

```
m_j* ← LLM(m_n ‖ (M_near \ m_j) ‖ m_j ‖ P_s3)    [A-Mem eq. 7]
```

The evolved `m_j*` replaces `m_j` in place. Over many turns, this yields a self-organizing note network: new experiences don't just accrete; they back-propagate into the organization of what's already there. A-Mem's ablation (Table 3) shows this module is load-bearing — link generation alone recovers most of the win, but evolution on top adds a further material gain on multi-hop and temporal tasks.

**Default: sync on write.** A-Mem's design assumes evolution runs inside the `write()` path, and the cost data supports it (~1,200 tokens/op, ~5 s on GPT-4o-mini, ~1.1 s on a local Llama 3.2 1B). Making evolution async would let a just-written note be retrieved before its neighbors have been updated, breaking the invariant the paper is built around. We default to sync-on-write, with an opt-in async/batched mode exposed via store configuration for throughput-sensitive deployments (bulk ingest, migrations).

**Scope: the nearest-neighbor set, not the whole store.** Evolution operates on the top-k neighbors returned by the link-generation step (A-Mem uses k=10 by default; we reuse that as our default). The store stays O(k) LLM calls per write, not O(N).

**Relationship to the graph layer (§5).** Evolution and the graph are complementary. The graph emerges from `L_i` link sets — edges are an observable projection of evolved notes. Evolution rewrites *node content*; the graph observes *edges between nodes*. In the implementation, `GraphStore` subclasses `EpisodicStore` and inherits the evolution pass; the graph projection is rebuilt from `links` after each write.

**Relationship to compaction (§3.5).** Orthogonal. Evolution refines a single note in light of its neighbors. Compaction emits a new note that subsumes many. A compaction pass over an evolved store is strictly richer than over a flat one.

**Public verb.** Evolution is internal to `write()` by default, but `MemoryStore.rewrite(record_id)` re-triggers it manually — needed for batch re-runs when the `P_s3` prompt changes or when the note schema migrates.

## 6. What this does *not* touch

- **Parametric memory / model editing.** The paper treats `ROME/MEND/MEMIT` as a separate governance track; so do we. If mellea later wants this, it plugs in *under* `MemoryStore` as a fifth store type with its own write path, but that is a separate proposal.
- **KV-cache persistence.** Chapter 4 of the mellea manifesto (cache-materialized training) already owns this; we do not re-open it.
- **Existing `ChatContext` / `SimpleContext` callers.** Zero breaking changes. Memory is opt-in by construction: you get it by instantiating `MemoryContext` instead of `ChatContext`.

## 7. Evaluation

Per the paper's §4 layered evaluation framework, we commit to the following gates for v1:

| Memory type | Metric | Target |
| --- | --- | --- |
| External | Recall@5, nDCG@10 on a held-out RAG eval | parity with mellea's existing RAG intrinsics |
| External | Citation Precision/Recall via `find_citations` intrinsic | ≥ 0.8 precision |
| Episodic / graph | LoCoMo | ≥ 85 (Mem0 v3 reports 91.6) |
| Episodic / graph | LongMemEval | ≥ 85 (Mem0 v3 reports 93.4) |
| Evolution | A-Mem ablation (LG+ME > LG-only > neither) reproduced on LoCoMo Multi-Hop and Temporal | ME delta ≥ the delta A-Mem reports (Table 3) within noise |
| Compaction | After window-side compaction, answerable-question rate on LongMemEval retained | ≥ 95% of the uncompacted baseline at equal token budget |
| Compaction | Long-term rollup preserves source provenance | 100% of consolidated records link back to source record_ids; `forget()` on a source is reflected in retrieval |
| Governance | 100% of writes produce an audit record; `forget()` is verified by a replay test | hard gate |

## 8. Open questions for the kickoff

1. Episodic extraction: single Granite adapter call, or can we reuse an existing intrinsic in `mellea/stdlib/components/intrinsic/`?
2. Default retrieval policy — hybrid (vector + BM25 + graph walk) at v1, or ship vector-only and layer the rest in v2?
3. Do we publish `MemoryBlock` in `mellea.core` (alongside `ImageBlock`) or in `mellea.stdlib.memory`? The paper's argument that memory is a first-class addressable state — and the fact that every backend already knows how to handle a `CBlock` — pushes us toward `core`.
4. How much of Mem0 do we depend on vs. re-implement? Current lean: re-implement in Mellea idiom, cite Mem0 as the reference design, stay API-compatible enough that someone moving from Mem0 recognizes the verbs.

---

*References: Zhang, D. et al. "Memory in Large Language Models: Mechanisms, Evaluation and Evolution." arXiv:2509.18868, 2025. Mem0 — mem0ai/mem0, [mem0.ai/research](https://mem0.ai/research).*
