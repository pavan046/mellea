"""Shared fixtures and the ``ScriptedBackend`` used to stub LLM calls."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Component, Context, ModelOutputThunk


class ScriptedBackend(Backend):
    """A ``Backend`` that serves pre-programmed JSON responses by prompt keyword.

    Each ``rule`` is a ``(substring, response)`` pair — the first rule whose
    substring appears in the prompt wins. Unmatched prompts raise so tests
    fail loudly instead of silently serving garbage.
    """

    def __init__(self, rules: list[tuple[str, Any]]):
        """Initialize ScriptedBackend with ordered (substring, response) rules."""
        self._rules = [(s.lower(), r) for s, r in rules]
        self.prompts_seen: list[str] = []

    async def _generate_from_context(self, action, ctx, **kw):  # pragma: no cover
        raise NotImplementedError("ScriptedBackend only supports generate_from_raw")

    async def generate_from_raw(
        self,
        actions: Sequence[Component | CBlock],
        ctx: Context,
        *,
        format: Any = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Return one thunk per action; each thunk carries a canned JSON string."""
        outputs: list[ModelOutputThunk] = []
        for action in actions:
            prompt = (
                str(action).lower()
                if not isinstance(action, CBlock)
                else (action.value or "").lower()
            )
            self.prompts_seen.append(prompt)
            response = self._match(prompt)
            if isinstance(response, dict):
                response = json.dumps(response)
            elif callable(response):
                response = response(prompt)
            outputs.append(ModelOutputThunk(response))
        return outputs

    def _match(self, prompt: str) -> Any:
        for key, value in self._rules:
            if key in prompt:
                return value
        raise AssertionError(
            f"ScriptedBackend got an unexpected prompt; add a matching rule.\n"
            f"Seen rules: {[k for k, _ in self._rules]}\n"
            f"Prompt head: {prompt[:300]!r}"
        )


@pytest.fixture
def scripted_backend_factory() -> Callable[..., ScriptedBackend]:
    """Factory fixture: call with ``rules`` to get a ``ScriptedBackend``."""

    def _make(rules: list[tuple[str, Any]]) -> ScriptedBackend:
        return ScriptedBackend(rules)

    return _make
