# Benchmarking a Function-Calling Pipeline

This tutorial covers how to wrap a function-calling pipeline in an OpenAI-compatible
FastAPI server and evaluate it against BFCL v3 and v4.

The server pattern is benchmark-agnostic. Any tool that speaks the OpenAI chat
completions API can be pointed at the same endpoint.

---

## Part 1: Starting the Server

`fc/server.py` wraps `FunctionCallingPipeline` in a FastAPI application and exposes
a `/v1/chat/completions` endpoint. Run it from the repo root with `uv`.

```bash
# Hub model
uv run python scratchpad/function_calling/fc/server.py \
    --model ibm-granite/granite-4.0-micro \
    --adapters /path/to/fc-system \
    --port 8080

# Local model path (directory name used as model ID for template resolution)
uv run python scratchpad/function_calling/fc/server.py \
    --model-path /checkpoints/granite-4.1-3b \
    --adapters /path/to/fc-system \
    --port 8080

# Local model path with explicit model name for template resolution
uv run python scratchpad/function_calling/fc/server.py \
    --model-path /checkpoints/my-finetune \
    --model-name ibm-granite/granite-4.1-3b \
    --adapters /path/to/fc-system \
    --port 8080
```

The server exposes:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /chat` | BFCL/ACEBench native format |
| `GET /health` | Startup health check |

Verify the server is up before running any benchmark:

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

---

## Part 2: BFCL Setup

BFCL v3 and v4 are separate snapshots of the same repository, each installed in its
own virtual environment. Both are still actively cited in academic literature.

```bash
# BFCL v4 (current HEAD)
git clone https://github.com/ShishirPatil/gorilla.git bfcl-v4
cd bfcl-v4/berkeley-function-call-leaderboard
python -m venv .venv && source .venv/bin/activate
pip install -e .
deactivate && cd ../..

# BFCL v3 (pinned commit)
git clone https://github.com/ShishirPatil/gorilla.git bfcl-v3
cd bfcl-v3
git checkout 1bb65c95e6dd49286eb427b308e6c91001e98ae9
cd berkeley-function-call-leaderboard
python -m venv .venv && source .venv/bin/activate
pip install -e .
deactivate && cd ../..
```

Always activate the correct venv before running BFCL commands.

---

## Part 3: Registering a Custom Handler

BFCL routes each model through a handler class. The `custom_openai.py` handler
provided here is a thin subclass that redirects the OpenAI client to your server
instead of `api.openai.com`. It is already committed to both BFCL checkouts.

The handler reads `CUSTOM_OPENAI_BASE_URL` from the environment and raises a clear
error if it is not set.

**Handler location (both versions):**
```
bfcl-v{3,4}/berkeley-function-call-leaderboard/
    bfcl_eval/model_handler/api_inference/custom_openai.py
```

**v3** (`custom_openai.py`):
```python
import os
from openai import OpenAI
from bfcl_eval.model_handler.api_inference.openai import OpenAIHandler
from bfcl_eval.model_handler.model_style import ModelStyle

class CustomOpenAIHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        if "CUSTOM_OPENAI_BASE_URL" not in os.environ:
            raise EnvironmentError(
                "CUSTOM_OPENAI_BASE_URL must be set (e.g. http://localhost:8080/v1)"
            )
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.client = OpenAI(base_url=os.environ["CUSTOM_OPENAI_BASE_URL"], api_key="EMPTY")
```

Note: `OPENAI_API_KEY` must be set before `super().__init__()` is called because the
base `OpenAIHandler` constructs an OpenAI client immediately. `setdefault` ensures
any real key already in the environment is not overwritten.

**v4** (`custom_openai.py`):
```python
import os
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

class CustomOpenAIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        if "CUSTOM_OPENAI_BASE_URL" not in os.environ:
            raise EnvironmentError(
                "CUSTOM_OPENAI_BASE_URL must be set (e.g. http://localhost:8080/v1)"
            )
        os.environ["OPENAI_BASE_URL"] = os.environ["CUSTOM_OPENAI_BASE_URL"]
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
```

The differences between the two are forced by BFCL internals: the base class was
renamed and moved, `ModelStyle` moved to a different module, and the `__init__`
signature gained two new parameters in v4. The logic is identical.

The handler is already registered as `"custom-openai"` in `model_config.py` in both
checkouts. No additional registration is needed.

---

## Part 4: Running BFCL

### BFCL v3

```bash
cd bfcl-v3/berkeley-function-call-leaderboard
source .venv/bin/activate

export CUSTOM_OPENAI_BASE_URL=http://localhost:8080/v1

# Generate model responses
bfcl generate \
    --model custom-openai \
    --test-category simple

# Evaluate
bfcl evaluate \
    --model custom-openai \
    --test-category simple
```

### BFCL v4

```bash
cd bfcl-v4/berkeley-function-call-leaderboard
source .venv/bin/activate

export CUSTOM_OPENAI_BASE_URL=http://localhost:8080/v1

# Generate model responses
bfcl generate \
    --model custom-openai \
    --test-category simple

# Evaluate
bfcl evaluate \
    --model custom-openai \
    --test-category simple
```

### Test categories

Pass `--test-category simple` as a smoke test. Substitute any category from the
table below, or omit the flag (equivalent to `all`) for a full dataset run.

| Category | v3 | v4 | Notes |
|---|---|---|---|
| `simple` | yes | yes | Single-turn, one tool call |
| `parallel` | yes | yes | Single-turn, parallel tool calls |
| `multiple` | yes | yes | Single-turn, multiple candidate tools |
| `parallel_multiple` | yes | yes | Parallel and multiple combined |
| `irrelevance` | yes | yes | No tool call should be made |
| `java`, `javascript` | yes | yes | Non-Python argument types |
| `live_simple` and variants | yes | yes | Crowd-sourced live queries |
| `multi_turn_base` and variants | yes | yes | Multi-turn conversations |
| `web_search_base`, `memory_kv`, etc. | no | yes | Agentic categories, v4 only |
| `all` | yes | yes | Full dataset run |

```bash
# Full dataset run
bfcl generate --model custom-openai --test-category all
bfcl evaluate --model custom-openai --test-category all
```

---

## Key Differences Between v3 and v4

| | v3 | v4 |
|---|---|---|
| Handler `__init__` | `(model_name, temperature)` | `(model_name, temperature, registry_name, is_fc_model, **kwargs)` |
| OpenAI handler base class | `OpenAIHandler` in `api_inference/openai.py` | `OpenAICompletionsHandler` in `api_inference/openai_completion.py` |
| `ModelStyle` import path | `model_handler.model_style` | `constants.enums` |
| Default backend for OSS models | `vllm` | `sglang` |
| Agentic test categories | no | yes (`memory_*`, `web_search_*`) |

---

## Note on Future Migration

`FunctionCallingPipeline` currently lives in `scratchpad/function_calling/fc/`.
When it is promoted to a first-class Mellea component the server startup arguments
will change, but the BFCL integration steps (handler file, `model_config.py` entry,
`CUSTOM_OPENAI_BASE_URL`, CLI commands) will remain the same.
