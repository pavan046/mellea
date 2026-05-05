"""Embedding backends for ``VectorStore``.

Defines the ``Embedder`` protocol and ships two implementations:

* ``HashingEmbedder`` — feature-hashed bag-of-words baseline. Deterministic,
  pure-python, no external dependencies. Good for tests and small scripts;
  not competitive with modern neural embeddings on real corpora.
* ``SentenceTransformersEmbedder`` — thin wrapper over the
  ``sentence-transformers`` library. Lazy import; a missing dependency
  surfaces a friendly install hint per mellea's friendly-dependency-errors
  convention.

Additional adapters (``OpenAIEmbedder``, watsonx, etc.) plug into the same
``Embedder`` protocol.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@runtime_checkable
class Embedder(Protocol):
    """Embedder protocol: text in, L2-normalized vector out.

    Implementations MUST return vectors with the same ``dim`` for every
    input. Output vectors SHOULD be L2-normalized so downstream cosine
    similarity can be computed as a plain dot product.
    """

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a ``float32`` array of shape ``(dim,)``."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed many strings. Returns a ``float32`` array of shape ``(N, dim)``."""
        ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder using feature hashing.

    Each token is hashed into one of ``dim`` buckets with a signed weight
    derived from its tf and a simple length prior. The resulting vector is
    L2-normalized. Collisions are a feature, not a bug — they bound the
    vector size regardless of vocabulary.

    Not competitive with neural embeddings on semantic similarity, but
    perfect for unit tests and reproducible demos.

    Args:
        dim (int): Output dimensionality. Defaults to 256. Powers of two give
            the most uniform hash distribution but any positive integer works.
        seed (int): Salt mixed into the hash to make test runs reproducible
            even across Python's randomized hash seed.
    """

    def __init__(self, *, dim: int = 256, seed: int = 0):
        """Initialize HashingEmbedder with fixed dimensionality and hash seed."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        """Output vector dimensionality."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a unit-norm ``float32`` vector."""
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            idx, sign = self._hash(tok)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings. Loops over ``embed`` — this is the cheap path."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts], axis=0)

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(
            f"{self._seed}:{token}".encode(), usedforsecurity=False
        ).digest()
        idx = int.from_bytes(digest[:4], "little") % self._dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign


class SentenceTransformersEmbedder:
    """Real embedder backed by the ``sentence-transformers`` library.

    Lazy-imports the library so mellea installs without pulling torch for
    users who only want the in-memory store. If the dependency is missing,
    construction raises ``ImportError`` with an install hint.

    Args:
        model_id (str): Hugging Face model id. Defaults to the widely-used
            ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim).
        device (str | None): Optional torch device string. Forwarded to
            ``SentenceTransformer``. ``None`` lets the library auto-pick.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ):
        """Initialize SentenceTransformersEmbedder, lazy-importing the library."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised via install docs
            raise ImportError(
                "SentenceTransformersEmbedder requires extra dependencies. "
                'Please install them with: pip install "mellea[memory]" '
                "or `pip install sentence-transformers`."
            ) from e

        self._model = SentenceTransformer(model_id, device=device)
        test = self._model.encode("probe", normalize_embeddings=True)
        self._dim = int(test.shape[-1])
        self._model_id = model_id

    @property
    def dim(self) -> int:
        """Output vector dimensionality inferred from the model."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns an L2-normalized ``float32`` vector."""
        vec = self._model.encode(text or "", normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch. Falls through to ``SentenceTransformer.encode``."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return np.asarray(vecs, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors.

    Safe for non-normalized inputs; falls back to 0.0 for zero vectors.
    Kept as a free function so callers can compose it with their own ranking.
    """
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0 or not math.isfinite(denom):
        return 0.0
    return float(np.dot(a, b) / denom)
