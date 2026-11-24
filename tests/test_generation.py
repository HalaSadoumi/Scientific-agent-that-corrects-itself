from io import BytesIO
from urllib.error import HTTPError

import pytest

from agent import generation
from agent.generation import ModelServiceError


def test_chat_json_reports_provider_http_error(monkeypatch) -> None:
    def reject_request(*args, **kwargs):
        raise HTTPError(
            "https://generativelanguage.googleapis.com/v1beta",
            429,
            "Too Many Requests",
            {},
            BytesIO(b'{"error":{"message":"Quota exceeded"}}'),
        )

    monkeypatch.setattr(generation, "urlopen", reject_request)
    with pytest.raises(ModelServiceError, match="HTTP 429: Quota exceeded"):
        generation.chat_json(
            api_base="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="test-model",
            system="system",
            user="user",
        )
