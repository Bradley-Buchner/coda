"""Tests for the shared sentence-embedding helpers (coda.embeddings)."""

import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from coda import embeddings
from coda.kg.embed_nodes import EMBEDDING_COL, embed_nodes


class _FakeSentenceTransformer:
    """Embeds each text as [len(text), 1, 0], normalized when asked."""
    loaded = []

    def __init__(self, model_name):
        _FakeSentenceTransformer.loaded.append(model_name)

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        vectors = np.array([[len(text), 1.0, 0.0] for text in texts])
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


@pytest.fixture
def fake_model(monkeypatch):
    _FakeSentenceTransformer.loaded = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embeddings, "_models", {})
    monkeypatch.setattr(embeddings, "_encode_locks", {})
    return _FakeSentenceTransformer


def test_import_does_not_load_sentence_transformers():
    # torch should only load when something is first embedded
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import coda.embeddings, coda.kg.embed_nodes; "
                "assert 'sentence_transformers' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_model_loaded_once_and_reused(fake_model):
    model = embeddings.get_embedding_model()
    assert embeddings.get_embedding_model() is model
    assert fake_model.loaded == [embeddings.DEFAULT_EMBEDDING_MODEL]


def test_embed_texts_returns_unit_float32_rows(fake_model):
    vectors = embeddings.embed_texts(["fever", "cough for three weeks"])
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_embed_texts_empty_input(fake_model):
    vectors = embeddings.embed_texts([])
    assert vectors.shape == (0, 3)


def test_embed_nodes_uses_shared_encoder(fake_model, tmp_path):
    nodes_file = tmp_path / "nodes.tsv.gz"
    pd.DataFrame({"id:ID": ["x:1", "x:2"], "name": ["Fever", None]}).to_csv(
        nodes_file, sep="\t", index=False)

    embed_nodes(nodes_file)

    df = pd.read_csv(nodes_file, sep="\t")
    vectors = np.array([row.split(";") for row in df[EMBEDDING_COL]],
                       dtype=float)
    assert vectors.shape == (2, 3)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert fake_model.loaded == [embeddings.DEFAULT_EMBEDDING_MODEL]
