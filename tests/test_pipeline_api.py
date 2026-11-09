from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from agent.api import main
from agent.api.main import app
from agent.ingestion.pipeline import ExtractedPage, chunk_pages


def test_chunking_preserves_page_and_overlap() -> None:
    source = "word " * 120
    chunks = chunk_pages((ExtractedPage(4, source),), size=200, overlap=30)
    assert len(chunks) > 1
    assert all(page == 4 for page, _ in chunks)
    assert chunks[0][1][-20:] in chunks[1][1]


def test_upload_is_indexed_and_retrieval_is_session_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        data_dir=str(tmp_path), max_upload_bytes=1024 * 1024,
        chunk_size_chars=300, chunk_overlap_chars=30,
    ))
    client = TestClient(app)
    session = uuid4()
    response = client.post(
        f"/sessions/{session}/inputs",
        data={"question": "What is the stellar mass?"},
        files=[
            ("files", ("paper.txt", b"The stellar mass is about 1.4 solar masses.", "text/plain"))
        ],
    )
    assert response.status_code == 200, response.text
    uploaded = response.json()["documents"][0]
    assert uploaded["chunk_count"] == 1
    assert uploaded["sha256"]

    found = client.get(f"/sessions/{session}/retrieve", params={"q": "stellar mass"})
    assert found.status_code == 200
    assert "1.4 solar masses" in found.json()["hits"][0]["text"]
    assert found.json()["hits"][0]["page"] == 1

    isolated = client.get(f"/sessions/{uuid4()}/retrieve", params={"q": "stellar mass"})
    assert isolated.status_code == 200
    assert isolated.json()["hits"] == []


def test_image_is_stored_without_claiming_ocr(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        data_dir=str(tmp_path), max_upload_bytes=1024 * 1024,
        chunk_size_chars=300, chunk_overlap_chars=30,
    ))
    client = TestClient(app)
    response = client.post(
        f"/sessions/{uuid4()}/inputs",
        files=[("files", ("chart.png", b"\x89PNG\r\n\x1a\n", "image/png"))],
    )
    assert response.status_code == 200
    assert response.json()["documents"][0]["text_extracted"] is False
    assert response.json()["documents"][0]["chunk_count"] == 0
