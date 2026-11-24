from fastapi.testclient import TestClient

from agent.api.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_contains_answer_generation_panel() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Draft and verify" in response.text
    assert 'id="answer-results"' in response.text
