"""Optional OpenAI-compatible answer generation and claim verification client."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelServiceError(RuntimeError):
    """Raised when a configured language model service fails or responds badly."""


def chat_json(
    *, api_base: str, api_key: str | None, model: str, system: str, user: str
) -> dict[str, Any]:
    """Ask an OpenAI-compatible chat endpoint for one JSON object."""
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            body = json.loads(response.read())
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise ModelServiceError(
            "The language-model request failed or returned invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise ModelServiceError("The language model did not return a JSON object")
    return result
