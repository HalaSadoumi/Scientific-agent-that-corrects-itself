"""Optional OpenAI-compatible answer generation and claim verification client."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ModelServiceError(RuntimeError):
    """Raised when a configured language model service fails or responds badly."""


def chat_json(
    *, api_base: str, api_key: str | None, model: str, system: str, user: str
) -> dict[str, Any]:
    """Ask Gemini's native API or another OpenAI-compatible endpoint for JSON."""
    headers = {"Content-Type": "application/json"}
    if "generativelanguage.googleapis.com" in (urlparse(api_base).hostname or ""):
        if not api_key:
            raise ModelServiceError("LLM_API_KEY is required for Gemini requests")
        endpoint = api_base.rstrip("/")
        if endpoint.endswith("/openai"):
            endpoint = endpoint.removesuffix("/openai")
        endpoint += f"/models/{model}:generateContent"
        headers["x-goog-api-key"] = api_key
        payload = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            }
        ).encode("utf-8")
    else:
        endpoint = api_base.rstrip("/") + "/chat/completions"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
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
    request = Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            body = json.loads(response.read())
        if "generativelanguage.googleapis.com" in (urlparse(api_base).hostname or ""):
            parts = body["candidates"][0]["content"]["parts"]
            content = "".join(part.get("text", "") for part in parts)
        else:
            content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            response_data = json.loads(body)
        except json.JSONDecodeError:
            response_data = {}
        provider_error = response_data.get("error") if isinstance(response_data, dict) else None
        message = provider_error.get("message") if isinstance(provider_error, dict) else None
        detail = str(message or body or exc.reason)[:500]
        raise ModelServiceError(f"Model provider returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        detail = str(exc.reason) if isinstance(exc, URLError) else "request timed out"
        raise ModelServiceError(f"Could not reach the model provider: {detail}") from exc
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ModelServiceError(
            "The language-model request failed or returned invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise ModelServiceError("The language model did not return a JSON object")
    return result
