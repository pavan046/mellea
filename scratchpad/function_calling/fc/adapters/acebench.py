# pytest: skip_always
"""ACEBench ↔ OpenAI protocol translation.

ACEBench uses the same /chat request/response format as BFCL: OpenAI-format
messages and tools on the way in, ChatResponse (type/tool_calls/content) on
the way out. Translation is identical — this module re-exports the BFCL
functions under ACEBench-named aliases so call sites stay readable.
"""

from __future__ import annotations

from fc.adapters.bfcl import (
    bfcl_request_to_openai as acebench_request_to_openai,
    openai_response_to_bfcl as openai_response_to_acebench,
)

__all__ = ["acebench_request_to_openai", "openai_response_to_acebench"]
