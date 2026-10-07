"""Shared sentence-embedding helpers that let CODA load a SentenceTransformer model
and produce embeddings for various text inputs (KG nodes, dialogue, questionnaires).
"""
import threading
from typing import Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_models = {}
_encode_locks = {}
_load_lock = threading.Lock()


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return the SentenceTransformer for ``model_name``, loading it once per process."""
    model = _models.get(model_name)
    if model is None:
        with _load_lock:
            model = _models.get(model_name)
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                _encode_locks[model_name] = threading.Lock()
                _models[model_name] = model
    return model


def embed_texts(texts: Sequence[str], model_name: str = DEFAULT_EMBEDDING_MODEL,
                show_progress_bar: bool = False) -> np.ndarray:
    """Embed texts as an ``(n, d)`` float32 array with unit-length rows."""
    model = get_embedding_model(model_name)
    texts = list(texts)
    if not texts:
        # Newer sentence-transformers renamed the dimension getter.
        get_dim = getattr(model, "get_embedding_dimension", None) \
            or model.get_sentence_embedding_dimension
        return np.zeros((0, get_dim()), dtype=np.float32)
    with _encode_locks[model_name]:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
    return np.asarray(embeddings, dtype=np.float32)
