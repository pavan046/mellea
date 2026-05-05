"""LLM-backed implementations of the A-Mem phase callables.

A-Mem (Xu et al. arXiv:2502.12110) decomposes the write path into three LLM
passes:

* **P_s1 — note construction** (§3.1 eq. 2): derive ``keywords``, ``tags``,
  and ``contextual_description`` from the raw note content.
* **P_s2 — link generation** (§3.2 eq. 6): pick which existing neighbours
  should be linked from the new note.
* **P_s3 — memory evolution** (§3.3 eq. 7): rewrite each related neighbour's
  ``keywords``/``tags``/``contextual_description`` in light of the new note.

Each phase is a free function returning a value compatible with
``InMemoryStore``'s injected-callable signatures so ``EpisodicStore`` (and
testers) can swap heuristics for LLM calls and back.

The LLM is invoked through the ``Backend.generate_from_raw`` interface — no
chat template, no hidden history, deterministic by default (``temperature=0``).
Each phase has a ``sync`` wrapper; internally it awaits a single thunk via
``asyncio.run`` when outside an event loop, or ``run_until_complete`` when
nested.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ....core.backend import Backend
from ....core.base import CBlock, ModelOutputThunk
from ....core.memory import MemoryRecord
from ...context import SimpleContext
from .schema import EpisodicRecord

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "templates" / "prompts" / "amem"
_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPT_DIR)),
    autoescape=select_autoescape(default=False, default_for_string=False),
    keep_trailing_newline=True,
)

_LOG = logging.getLogger(__name__)

_AMEM_MODEL_OPTIONS: dict[str, Any] = {"temperature": 0.0}


def _render(template_name: str, **ctx: Any) -> str:
    return _ENV.get_template(template_name).render(**ctx)


def _run_coro(coro: Any) -> Any:
    """Run ``coro`` to completion whether or not we're in an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # We're inside a running loop (e.g., already-async caller). Schedule the
    # coroutine on a background thread so the caller isn't forced to be async.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _raw_call(backend: Backend, prompt: str) -> str:
    """Single ``generate_from_raw`` round-trip. Returns the raw string output."""
    thunks: list[ModelOutputThunk] = await backend.generate_from_raw(
        [CBlock(prompt)], SimpleContext(), model_options=_AMEM_MODEL_OPTIONS
    )
    value = await thunks[0].avalue()
    return value or ""


def _parse_json_object(raw: str, *, expected_keys: list[str]) -> dict[str, Any]:
    """Best-effort JSON parse that tolerates leading/trailing chatter.

    Returns an empty dict with default values when parsing fails so the store
    can fall back to heuristic defaults without blowing up the write path.
    """
    if not raw:
        return {}
    # Strip common wrappers: code fences, leading prose before '{'.
    text = raw.strip()
    if text.startswith("```"):
        # Remove triple-backtick fences with optional lang tag.
        fence_end = text.find("\n")
        if fence_end != -1:
            text = text[fence_end + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        _LOG.warning("A-Mem output did not contain a JSON object: %r", raw[:200])
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        _LOG.warning("A-Mem JSON parse failed (%s): %r", e, raw[:200])
        return {}
    if not isinstance(obj, dict):
        return {}
    missing = [k for k in expected_keys if k not in obj]
    if missing:
        _LOG.warning("A-Mem output missing keys %s in %r", missing, obj)
    return obj


# ---- P_s1: note construction ---------------------------------------------


def llm_note_enricher(backend: Backend):
    """Build an enricher callable that runs the P_s1 prompt on a backend.

    Returned callable matches the ``NoteEnricher`` signature used by
    ``InMemoryStore`` and ``EpisodicStore``: takes a content string and
    returns ``(keywords, tags, contextual_description)``.

    Args:
        backend (Backend): Any concrete ``Backend`` instance.

    Returns:
        A callable ``(content: str) -> tuple[list[str], list[str], str]``.
    """

    def _enrich(
        content: str,
        *,
        actor: str | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> tuple[list[str], list[str], str]:
        prompt = _render(
            "note_construction.jinja2",
            content=content,
            actor=actor,
            timestamp=timestamp.isoformat() if timestamp else None,
        )
        raw = _run_coro(_raw_call(backend, prompt))
        obj = _parse_json_object(
            raw, expected_keys=["keywords", "tags", "contextual_description"]
        )
        return (
            _as_str_list(obj.get("keywords")),
            _as_str_list(obj.get("tags")),
            str(obj.get("contextual_description") or content[:160]),
        )

    return _enrich


# ---- P_s2: link generation ------------------------------------------------


def llm_link_generator(backend: Backend):
    """Build a link-generator callable that runs the P_s2 prompt on a backend.

    Returned callable takes ``(new_record, neighbors)`` and returns the list
    of ``record_id``s the new note should link to. Never invents ids — any id
    not present in ``neighbors`` is dropped from the output.
    """

    def _links(new_record: MemoryRecord, neighbors: list[MemoryRecord]) -> list[str]:
        if not neighbors:
            return []
        prompt = _render(
            "link_generation.jinja2", new_record=new_record, neighbors=neighbors
        )
        raw = _run_coro(_raw_call(backend, prompt))
        obj = _parse_json_object(raw, expected_keys=["links"])
        raw_links = _as_str_list(obj.get("links"))
        valid = {n.record_id for n in neighbors}
        return [r for r in raw_links if r in valid]

    return _links


# ---- P_s3: memory evolution ----------------------------------------------


def llm_evolver(backend: Backend):
    """Build an evolver callable that runs the P_s3 prompt per neighbour.

    Runs one LLM call per neighbour; returns the rewritten neighbour list.
    Neighbours whose fields did not meaningfully change are returned
    unchanged (the caller may still persist them, which is a no-op at the
    SQL layer since values match).
    """

    def _evolve(
        new_record: MemoryRecord, neighbors: list[MemoryRecord]
    ) -> list[MemoryRecord]:
        if not neighbors:
            return []
        evolved: list[MemoryRecord] = []
        for neighbor in neighbors:
            others = [n for n in neighbors if n.record_id != neighbor.record_id]
            prompt = _render(
                "evolution.jinja2",
                new_record=new_record,
                neighbor=neighbor,
                other_neighbors=others,
            )
            raw = _run_coro(_raw_call(backend, prompt))
            obj = _parse_json_object(
                raw, expected_keys=["keywords", "tags", "contextual_description"]
            )
            if obj:
                neighbor.keywords = _as_str_list(obj.get("keywords")) or list(
                    neighbor.keywords
                )
                neighbor.tags = _as_str_list(obj.get("tags")) or list(neighbor.tags)
                new_desc = obj.get("contextual_description")
                if isinstance(new_desc, str) and new_desc.strip():
                    neighbor.contextual_description = new_desc[:200]
            evolved.append(neighbor)
        return evolved

    return _evolve


# ---- Compaction summarizer -----------------------------------------------


def llm_summarizer(backend: Backend):
    """Build a summarizer used by compaction to consolidate a cluster.

    Returned callable takes ``list[MemoryRecord]`` (members of a cluster)
    and returns ``(summary_text, keywords, tags)``. Falls back to a simple
    concatenation when the JSON response is unparseable.
    """

    def _summarize(records: list[MemoryRecord]) -> tuple[str, list[str], list[str]]:
        if not records:
            return "", [], []
        prompt = _render("summarize.jinja2", records=records)
        raw = _run_coro(_raw_call(backend, prompt))
        obj = _parse_json_object(raw, expected_keys=["summary", "keywords", "tags"])
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "\n---\n".join(r.content for r in records)
        return (
            summary,
            _as_str_list(obj.get("keywords")),
            _as_str_list(obj.get("tags")) or ["summary"],
        )

    return _summarize


# ---- helpers --------------------------------------------------------------


def _as_str_list(value: Any) -> list[str]:
    """Coerce LLM output to ``list[str]``; drops non-strings defensively."""
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int, float))]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


__all__ = [
    "EpisodicRecord",
    "llm_evolver",
    "llm_link_generator",
    "llm_note_enricher",
    "llm_summarizer",
]
