import os
from pathlib import Path

MODEL_ID = "ibm-granite/granite-4.0-micro"
HF_HOME = Path("/proj/dmfexp/tool_reasoning_code/kapanipa/intrinsics")
model_cache_dir = HF_HOME / "hub" / ("models--" + MODEL_ID.replace("/", "--"))

if model_cache_dir.exists():
    print(f"Model found at {model_cache_dir}, loading locally...")
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    raise RuntimeError(
        f"Model not found at {model_cache_dir}. "
        "Run 'uv run python scratchpad/download_rag_assets.py' first."
    )

from mellea.backends.huggingface import LocalHFBackend  # noqa: E402
from mellea.stdlib.components import Document, Message  # noqa: E402
from mellea.stdlib.components.intrinsic import rag  # noqa: E402
from mellea.stdlib.context import ChatContext  # noqa: E402

backend = LocalHFBackend(model_id=MODEL_ID)
context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))
question = "What is the square root of 4?"

docs_answerable = [Document("The square root of 4 is 2.")]
docs_not_answerable = [Document("The square root of 8 is approximately 2.83.")]

score1 = rag.check_answerability(question, docs_answerable, context, backend)
score2 = rag.check_answerability(question, docs_not_answerable, context, backend)

print(f"Answerable docs score:     {score1:.4f}  (expect high, ~1.0)")
print(f"Non-answerable docs score: {score2:.4f}  (expect low, ~0.0)")
