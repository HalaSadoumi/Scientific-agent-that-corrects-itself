"""Free local astronomy embeddings and optional OpenAI-compatible API embeddings."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingServiceError(RuntimeError):
    """Raised when the configured embedding provider cannot produce vectors."""


@lru_cache(maxsize=2)
def _local_model(model_name: str):
    """Load the local astronomy model once per process, only when requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingServiceError(
            "Local embeddings need the optional 'astro-embeddings' dependencies; "
            "install them with `uv sync --extra astro-embeddings`."
        ) from exc
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        raise EmbeddingServiceError(
            "Could not load the local embedding model. Check model name and network access."
        ) from exc


def embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    """Create normalized vectors locally; text does not leave this computer."""
    if not texts:
        return []
    model = _local_model(model_name)
    try:
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [[float(value) for value in row] for row in vectors]
    except Exception as exc:
        raise EmbeddingServiceError("The local embedding model could not encode the text") from exc


def embed_texts(texts: list[str], api_base: str, api_key: str, model: str) -> list[list[float]]:
    if not texts:
        return []
    endpoint = api_base.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EmbeddingServiceError("The embedding service request failed") from exc
    try:
        rows = sorted(body["data"], key=lambda row: row["index"])
        vectors = [[float(value) for value in row["embedding"]] for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingServiceError("The embedding service returned an invalid response") from exc
    if len(vectors) != len(texts) or not vectors or not vectors[0]:
        raise EmbeddingServiceError("The embedding service returned the wrong number of vectors")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise EmbeddingServiceError("The embedding service returned inconsistent vector sizes")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise EmbeddingServiceError("The embedding service returned non-finite values")
    return vectors


def selected_backend(settings: object) -> str:
    backend = getattr(settings, "embedding_backend", "auto")
    if backend == "auto":
        return "api" if getattr(settings, "embedding_api_base", None) else "none"
    return backend


def create_embeddings(texts: list[str], settings: object) -> list[list[float]] | None:
    """Use the explicitly selected backend; 'none' means keyword-only retrieval."""
    backend = selected_backend(settings)
    if backend == "none":
        return None
    if backend == "local":
        return embed_local(
            texts, getattr(settings, "local_embedding_model", "NeuML/astrobert-small-embeddings")
        )
    if backend == "api":
        api_base = getattr(settings, "embedding_api_base", None)
        if not api_base:
            raise EmbeddingServiceError(
                "EMBEDDING_API_BASE is required for the API embedding backend"
            )
        return embed_texts(
            texts,
            api_base,
            getattr(settings, "embedding_api_key", None) or "",
            getattr(settings, "embedding_model", "text-embedding-3-small"),
        )
    raise EmbeddingServiceError(f"Unsupported embedding backend: {backend}")
