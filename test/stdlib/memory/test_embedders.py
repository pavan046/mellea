"""Tests for the embedder protocol and HashingEmbedder fallback."""

from __future__ import annotations

import numpy as np
import pytest

from mellea.stdlib.memory.embedders import Embedder, HashingEmbedder, cosine_similarity


def test_hashing_embedder_conforms_to_protocol() -> None:
    emb = HashingEmbedder(dim=64)
    assert isinstance(emb, Embedder)
    assert emb.dim == 64


def test_hashing_embedder_output_shape_and_norm() -> None:
    emb = HashingEmbedder(dim=128)
    vec = emb.embed("the quick brown fox jumps")
    assert vec.shape == (128,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)


def test_hashing_embedder_deterministic_for_same_input() -> None:
    emb = HashingEmbedder(dim=64, seed=7)
    v1 = emb.embed("repeatable input")
    v2 = emb.embed("repeatable input")
    assert np.allclose(v1, v2)


def test_hashing_embedder_empty_input_returns_zero_vector() -> None:
    emb = HashingEmbedder(dim=32)
    vec = emb.embed("")
    assert vec.shape == (32,)
    assert not np.any(vec)


def test_hashing_embedder_batch_returns_2d_matrix() -> None:
    emb = HashingEmbedder(dim=32)
    batch = emb.embed_batch(["one", "two", "three"])
    assert batch.shape == (3, 32)
    assert batch.dtype == np.float32


def test_hashing_embedder_empty_batch() -> None:
    emb = HashingEmbedder(dim=16)
    assert emb.embed_batch([]).shape == (0, 16)


def test_hashing_embedder_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_cosine_similarity_bounds() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    d = np.zeros(3, dtype=np.float32)

    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(-1.0)
    assert cosine_similarity(a, d) == 0.0  # zero-vec safe


def test_identical_text_has_higher_similarity_than_disjoint() -> None:
    emb = HashingEmbedder(dim=256)
    a = emb.embed("machine learning for natural language processing")
    b = emb.embed("machine learning for natural language processing")
    c = emb.embed("the history of medieval pottery glazes")
    assert cosine_similarity(a, b) > cosine_similarity(a, c)
