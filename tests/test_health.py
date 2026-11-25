from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from agent.api import main
from agent.api.main import app
from agent.config import PROJECT_ROOT, Settings
from agent.storage.catalog import Catalog

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_settings_paths_are_project_rooted(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_DIR", raising=False)

    settings = Settings(_env_file=None, data_dir="custom-data")

    assert Path(settings.data_dir) == PROJECT_ROOT / "custom-data"
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_frontend_contains_answer_generation_panel() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Astronomy evidence assistant" in response.text
    assert 'data-view="graph"' in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="view-policy"' in response.text
    assert 'id="graph-evidence-results"' in response.text


def test_dashboard_snapshot_is_session_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(data_dir=str(tmp_path), storage_backend="sqlite"),
    )
    session = uuid4()
    Catalog(tmp_path / "catalog.sqlite3").add_document(
        {
            "id": str(uuid4()), "session_id": str(session), "filename": "astro.txt",
            "mime_type": "text/plain", "sha256": "a" * 64, "storage_key": "astro.txt",
            "parser": "utf8-text", "page_count": 1,
            "created_at": "2026-11-24T00:00:00+00:00", "source_tier": "t0_user",
        },
        [{"page": 1, "order": 0, "text": "NGC 123 was observed by Chandra."}],
    )

    result = client.get(f"/sessions/{session}/snapshot").json()
    isolated = client.get(f"/sessions/{uuid4()}/snapshot").json()
    assert result["document_count"] == 1
    assert result["chunk_count"] == 1
    assert result["word_count"] == 6
    assert isolated["document_count"] == 0


def test_graph_endpoint_returns_explicit_disabled_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(neo4j_uri=None))
    session = uuid4()
    result = client.get(f"/sessions/{session}/graph").json()
    assert result["enabled"] is False
    assert result["entities"] == []
    graph_hits = client.get(f"/sessions/{session}/graph/search", params={"q": "NGC 123"})
    assert graph_hits.status_code == 200
    assert graph_hits.json()["hits"] == []
