"""Optional client for OpenAI-compatible text embedding services."""

from __future__ import annotations

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingServiceError(RuntimeError):
    """Raised when the configured embedding provider cannot produce vectors."""


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
