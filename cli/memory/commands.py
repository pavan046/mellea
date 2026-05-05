"""Typer commands for ``m memory`` — external memory ingestion and inspection.

Keeps the CLI surface narrow: ``m memory ingest`` chunks and embeds a file (or
directory of files) into a ``VectorStore`` on disk, ``m memory list`` summarises
a saved store, and ``m memory recall`` runs an ad-hoc retrieval against a saved
store. Heavier imports (``numpy``, embedding backends) are deferred to command
bodies so ``m memory --help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path

import typer

memory_app = typer.Typer(name="memory", help="Manage mellea's external memory stores.")


def _iter_text_files(source: Path) -> list[Path]:
    """Yield readable text files under ``source`` in a deterministic order."""
    if source.is_file():
        return [source]
    if source.is_dir():
        exts = {".txt", ".md", ".rst", ".json", ".log", ".py"}
        return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix in exts)
    raise typer.BadParameter(f"{source} is neither a file nor a directory")


@memory_app.command("ingest")
def ingest(
    source: Path = typer.Argument(
        ..., exists=True, readable=True, help="File or directory to ingest."
    ),
    store_path: Path = typer.Option(
        ..., "--store", "-s", help="Directory where the VectorStore will be saved."
    ),
    store_id: str = typer.Option(
        "vector", "--store-id", help="Stable id for the store."
    ),
    window_size: int = typer.Option(
        200, help="Chunk window size in whitespace tokens."
    ),
    overlap: int = typer.Option(40, help="Chunk overlap in whitespace tokens."),
    embedder: str = typer.Option(
        "hashing",
        "--embedder",
        help="Embedder backend: 'hashing' (dependency-free) or 'sentence-transformers'.",
        case_sensitive=False,
    ),
    embedder_dim: int = typer.Option(
        256, help="Dimensionality for the 'hashing' embedder. Ignored otherwise."
    ),
    model_id: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        help="HF model id for the 'sentence-transformers' embedder.",
    ),
) -> None:
    """Chunk, embed, and persist a document (or directory of documents).

    Prerequisites:
        Mellea installed. The 'sentence-transformers' embedder requires the
        extra install: ``pip install 'mellea[memory]'`` (or
        ``pip install sentence-transformers``).

    Examples:
        m memory ingest ./docs/handbook.md --store ./memory.db
        m memory ingest ./docs/ --store ./memory.db --embedder sentence-transformers

    See Also:
        guide: memory/external
    """
    from mellea.stdlib.memory import (
        HashingEmbedder,
        SentenceTransformersEmbedder,
        VectorStore,
    )

    chosen = embedder.lower().replace("_", "-")
    if chosen == "hashing":
        emb = HashingEmbedder(dim=embedder_dim)
    elif chosen in {"sentence-transformers", "st"}:
        emb = SentenceTransformersEmbedder(model_id=model_id)
    else:
        raise typer.BadParameter(
            f"unknown embedder {embedder!r}; expected 'hashing' or 'sentence-transformers'"
        )

    if store_path.exists() and any(store_path.iterdir()):
        store = VectorStore.load(store_path, embedder=emb)
        typer.echo(
            f"Loaded existing store from {store_path} ({len(store.doc_ids())} docs)"
        )
    else:
        store = VectorStore(
            store_id=store_id, embedder=emb, window_size=window_size, overlap=overlap
        )
        typer.echo(f"Created new store at {store_path}")

    files = _iter_text_files(source)
    if not files:
        typer.echo("No ingestible files found under source.", err=True)
        raise typer.Exit(code=1)

    total_chunks = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            typer.echo(f"skip (non-utf8): {f}", err=True)
            continue
        doc_id = str(f.resolve())
        records = store.add_document(text, doc_id=doc_id, title=f.name)
        total_chunks += len(records)
        typer.echo(f"  {f}: {len(records)} chunks")

    store.save(store_path)
    typer.echo(
        f"Ingested {len(files)} file(s) -> {total_chunks} chunk(s). "
        f"Store saved to {store_path}."
    )


@memory_app.command("list")
def list_docs(
    store_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to saved VectorStore.",
    ),
    embedder: str = typer.Option(
        "hashing", help="Embedder used when the store was created. Must match."
    ),
    embedder_dim: int = typer.Option(256, help="Dimensionality for 'hashing'."),
    model_id: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
) -> None:
    """Summarize the documents and chunk counts in a saved store.

    Examples:
        m memory list ./memory.db

    See Also:
        guide: memory/external
    """
    from mellea.stdlib.memory import (
        HashingEmbedder,
        SentenceTransformersEmbedder,
        VectorStore,
    )

    chosen = embedder.lower().replace("_", "-")
    emb = (
        HashingEmbedder(dim=embedder_dim)
        if chosen == "hashing"
        else SentenceTransformersEmbedder(model_id=model_id)
    )
    store = VectorStore.load(store_path, embedder=emb)
    typer.echo(f"store_id={store._store_id!r} docs={len(store.doc_ids())}")
    for doc_id in store.doc_ids():
        chunks = store.chunks_for(doc_id)
        title = next(
            (c.meta.get("doc_title") for c in chunks if c.meta.get("doc_title")), None
        )
        typer.echo(f"  {doc_id} | title={title!r} | chunks={len(chunks)}")


@memory_app.command("recall")
def recall(
    query: str = typer.Argument(..., help="Retrieval query."),
    store_path: Path = typer.Option(
        ..., "--store", "-s", exists=True, file_okay=False, dir_okay=True
    ),
    k: int = typer.Option(5, "--k", "-k", help="Number of results."),
    embedder: str = typer.Option("hashing"),
    embedder_dim: int = typer.Option(256),
    model_id: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
) -> None:
    """Run an ad-hoc retrieval against a saved store.

    Examples:
        m memory recall 'computer vision' --store ./memory.db -k 3

    See Also:
        guide: memory/external
    """
    from mellea.stdlib.memory import (
        HashingEmbedder,
        SentenceTransformersEmbedder,
        VectorStore,
    )

    chosen = embedder.lower().replace("_", "-")
    emb = (
        HashingEmbedder(dim=embedder_dim)
        if chosen == "hashing"
        else SentenceTransformersEmbedder(model_id=model_id)
    )
    store = VectorStore.load(store_path, embedder=emb)
    hits = store.retrieve(query, k=k)
    if not hits:
        typer.echo("No matches.")
        return
    for h in hits:
        typer.echo(
            f"score={h.score:.3f} doc_id={h._meta.get('doc_id')!r} "
            f"chunk={h._meta.get('chunk_index')} title={h._meta.get('doc_title')!r}"
        )
        typer.echo(f"  {h.value[:200]}")
