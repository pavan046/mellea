"""Download all assets required by test_rag.py and query_clarification_rag.py.

Run once to pre-populate the cache:
    uv run python scratchpad/download_rag_assets.py
"""

import os
from pathlib import Path

DOWNLOAD_DIR = Path("/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Must be set before any huggingface_hub calls so all downloads land here.
os.environ["HF_HOME"] = str(DOWNLOAD_DIR)

import huggingface_hub  # noqa: E402

from mellea.formatters.granite.intrinsics import obtain_lora  # noqa: E402

MODEL_ID = "ibm-granite/granite-4.0-micro"
RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
BASE_MODEL_NAME = "granite-4.0-micro"

print(f"Downloading base model {MODEL_ID} -> {DOWNLOAD_DIR}/hub/ ...")
huggingface_hub.snapshot_download(MODEL_ID)
print("Base model done.")

ADAPTERS = [
    "answerability",
    "query_rewrite",
    "query_clarification",
    "hallucination_detection",
    "citations",
]

for adapter in ADAPTERS:
    print(f"Downloading {adapter} LoRA adapter from {RAG_REPO} ...")
    obtain_lora(adapter, BASE_MODEL_NAME, RAG_REPO, alora=False)
    print(f"  {adapter} done.")

print(f"\nAll assets saved to {DOWNLOAD_DIR}")
