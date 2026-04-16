# pytest: skip_always
"""BenchmarkServer — FastAPI server wrapping FunctionCallingPipeline.

Exposes three endpoints:

    POST /chat                  BFCL and ACEBench (ChatRequest/ChatResponse format)
    POST /v1/chat/completions   TAU2 (OpenAI-compatible)
    GET  /health                startup health poll

All protocol translation between benchmark wire formats and the OpenAI format
that FCPipeline expects lives in fc/adapters/. This file contains only HTTP
plumbing and startup logic.

Usage:
    # Hub model:
    uv run python scratchpad/function_calling/fc/server.py \\
        --model ibm-granite/granite-4.0-micro \\
        --adapters /path/to/fc-system \\
        --port 8080

    # Local model path (directory name used as model ID for template resolution):
    uv run python scratchpad/function_calling/fc/server.py \\
        --model-path /checkpoints/granite-4.1-3b \\
        --adapters /path/to/fc-system \\
        --port 8080

    # Local model path with explicit model name for template resolution:
    uv run python scratchpad/function_calling/fc/server.py \\
        --model-path /checkpoints/my-finetune \\
        --model-name ibm-granite/granite-4.1-3b \\
        --adapters /path/to/fc-system \\
        --port 8080

    # Override a single adapter:
    uv run python scratchpad/function_calling/fc/server.py \\
        --model ibm-granite/granite-4.0-micro \\
        --adapters /path/to/fc-system \\
        --adapter-override fc_router=/experiments/my_router \\
        --port 8080

Test with curl:
    curl http://localhost:8080/health

    curl -X POST http://localhost:8080/chat \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
        "tools": [{
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
              "type": "object",
              "properties": {"city": {"type": "string"}},
              "required": ["city"]
            }
          }
        }]
      }'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

# Make the fc package importable when running as a script or with -m from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from fc.adapters.bfcl import bfcl_request_to_openai, openai_response_to_bfcl
from fc.adapters.tau2 import openai_response_to_tau2, tau2_request_to_openai

logger = logging.getLogger("fc.server")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """BFCL /chat request. Messages and tools in OpenAI format."""

    id: Optional[str] = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = []


class ToolCall(BaseModel):
    """A single tool call in a ChatResponse."""

    name: str
    arguments: dict[str, Any]


class ChatResponse(BaseModel):
    """BFCL /chat response."""

    type: Literal["tool_calls", "response", "error"]
    tool_calls: Optional[list[ToolCall]] = None
    content: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request (TAU2 / v1/chat/completions).

    Only the fields FCPipeline needs are declared. Extra fields sent by the
    OpenAI SDK (stream, logprobs, etc.) are accepted and silently ignored via
    model_config extra="ignore".
    """

    model_config = ConfigDict(extra="ignore")

    model: str = "fc-agent"
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FC BenchmarkServer",
    description="FunctionCallingPipeline exposed for BFCL, TAU2, and ACEBench.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check for benchmark startup polls."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """BFCL and ACEBench endpoint.

    Accepts messages and tools in OpenAI format, calls FCPipeline, and returns
    a ChatResponse with type "tool_calls" or "response".
    """

    pipeline = app.state.pipeline
    messages, tools = bfcl_request_to_openai(req.model_dump())
    assistant_msg = await run_in_threadpool(
        pipeline.run, messages=messages, tools=tools
    )
    bfcl_resp = openai_response_to_bfcl(assistant_msg)

    tool_calls = None
    if bfcl_resp["type"] == "tool_calls":
        tool_calls = [ToolCall(**tc) for tc in bfcl_resp["tool_calls"]]

    return ChatResponse(
        type=bfcl_resp["type"], tool_calls=tool_calls, content=bfcl_resp.get("content")
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest) -> JSONResponse:
    """TAU2 endpoint (OpenAI-compatible).

    Accepts a standard OpenAI ChatCompletion request and returns a standard
    OpenAI ChatCompletion response. TAU2 uses the OpenAI Python SDK, so
    the response must be a valid ChatCompletion object.
    """
    pipeline = app.state.pipeline
    messages, tools = tau2_request_to_openai(req.model_dump(exclude_none=True))
    assistant_msg = await run_in_threadpool(
        pipeline.run, messages=messages, tools=tools
    )
    response = openai_response_to_tau2(assistant_msg, model=req.model)
    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Startup and CLI
# ---------------------------------------------------------------------------


def _build_pipeline(args: argparse.Namespace):
    """Load model and construct FunctionCallingPipeline."""
    # Deferred imports: heavy deps (torch, transformers) only loaded at server start.
    from fc.pipeline import FunctionCallingPipeline
    from mellea.backends.huggingface import LocalHFBackend
    from mellea.formatters.template_formatter import TemplateFormatter

    overrides: dict[str, str] = {}
    for item in args.adapter_override or []:
        if "=" not in item:
            logger.error("--adapter-override must be NAME=PATH, got: %r", item)
            sys.exit(1)
        name, path = item.split("=", 1)
        overrides[name] = path

    if args.model:
        logger.info("Loading model from Hub: %s", args.model)
        backend = LocalHFBackend(model_id=args.model)
    else:
        model_name = args.model_name or Path(args.model_path).name
        logger.info(
            "Loading model from local path: %s (name: %s)", args.model_path, model_name
        )
        formatter = TemplateFormatter(model_id=model_name)
        backend = LocalHFBackend(model_id=args.model_path, formatter=formatter)

    pipeline = FunctionCallingPipeline(
        backend=backend, adapters_dir=args.adapters, adapter_overrides=overrides or None
    )
    return pipeline


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FC BenchmarkServer — serves FunctionCallingPipeline over HTTP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", help="HuggingFace model ID (e.g. ibm-granite/granite-4.0-micro)."
    )
    model_group.add_argument("--model-path", help="Local path to a model directory.")
    parser.add_argument(
        "--model-name",
        help="Model name for template resolution when using --model-path "
        "(e.g. ibm-granite/granite-4.1-3b). Defaults to the directory name "
        "when not provided.",
    )
    parser.add_argument(
        "--adapters",
        required=True,
        help="Root directory containing adapter subdirectories (router, "
        "parallel_tool_calling, multi_step_tool_calling, conversational_detection).",
    )
    parser.add_argument(
        "--adapter-override",
        action="append",
        metavar="NAME=PATH",
        default=[],
        help="Override a single adapter path. May be repeated. "
        "Example: --adapter-override fc_router=/experiments/my_router",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to bind (default: 8080)."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    logger.info("Building pipeline...")
    app.state.pipeline = _build_pipeline(args)
    logger.info("Pipeline ready. Starting server on %s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
