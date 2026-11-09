from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from agent.api import main
from agent.api.main import app
from agent.ingestion.pipeline import ExtractedPage, chunk_pages
from agent.storage.catalog import Catalog


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
        ocr_enabled=True, ocr_language="eng", embedding_api_base=None,
        embedding_api_key=None, embedding_model="test", semantic_weight=0.5,
        object_storage_backend="local", s3_endpoint="127.0.0.1:3900",
        s3_bucket="uploads", s3_access_key="key", s3_secret_key="secret", s3_region="garage",
        llm_api_base=None, llm_api_key=None, llm_model=None, verification_model=None,
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
        ocr_enabled=True, ocr_language="eng", embedding_api_base=None,
        embedding_api_key=None, embedding_model="test", semantic_weight=0.5,
        object_storage_backend="local", s3_endpoint="127.0.0.1:3900",
        s3_bucket="uploads", s3_access_key="key", s3_secret_key="secret", s3_region="garage",
        llm_api_base=None, llm_api_key=None, llm_model=None, verification_model=None,
    ))
    client = TestClient(app)
    response = client.post(
        f"/sessions/{uuid4()}/inputs",
        files=[("files", ("chart.png", b"\x89PNG\r\n\x1a\n", "image/png"))],
    )
    assert response.status_code == 200
    assert response.json()["documents"][0]["text_extracted"] is False
    assert response.json()["documents"][0]["ocr_status"] == "unavailable"
    assert response.json()["documents"][0]["chunk_count"] == 0


def test_embedding_search_finds_passage_without_shared_words(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        data_dir=str(tmp_path), max_upload_bytes=1024 * 1024,
        chunk_size_chars=300, chunk_overlap_chars=30,
        ocr_enabled=True, ocr_language="eng", embedding_api_base="http://local.test/v1",
        embedding_api_key="test-key", embedding_model="test", semantic_weight=0.5,
        object_storage_backend="local", s3_endpoint="127.0.0.1:3900",
        s3_bucket="uploads", s3_access_key="key", s3_secret_key="secret", s3_region="garage",
        llm_api_base=None, llm_api_key=None, llm_model=None, verification_model=None,
    ))
    monkeypatch.setattr(main, "embed_texts", lambda texts, *args: [[1.0, 0.0] for _ in texts])
    client = TestClient(app)
    session = uuid4()
    uploaded = client.post(
        f"/sessions/{session}/inputs",
        files=[
            ("files", ("paper.txt", b"The planet has a large volcanic mountain.", "text/plain"))
        ],
    )
    assert uploaded.status_code == 200, uploaded.text

    result = client.get(f"/sessions/{session}/retrieve", params={"q": "Olympus Mons"})
    assert result.status_code == 200
    assert result.json()["retrieval_method"] == "hybrid_rrf"
    assert "volcanic mountain" in result.json()["hits"][0]["text"]


def test_answer_checks_generated_citations_and_records_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        data_dir=str(tmp_path), max_upload_bytes=1024 * 1024,
        chunk_size_chars=300, chunk_overlap_chars=30,
        ocr_enabled=True, ocr_language="eng", embedding_api_base=None,
        embedding_api_key=None, embedding_model="test", semantic_weight=0.5,
        object_storage_backend="local", s3_endpoint="127.0.0.1:3900",
        s3_bucket="uploads", s3_access_key="key", s3_secret_key="secret", s3_region="garage",
        llm_api_base="http://local.test/v1", llm_api_key="test-key",
        llm_model="writer", verification_model="checker",
    ))
    client = TestClient(app)
    session = uuid4()
    upload = client.post(
        f"/sessions/{session}/inputs",
        files=[
            (
                "files",
                ("paper.txt", b"Betelgeuse has a mass of about 16 solar masses.", "text/plain"),
            )
        ],
    )
    document_id = upload.json()["documents"][0]["id"]
    citation = f"{document_id}:1:0"
    models_called = []

    def fake_chat_json(*, model, **kwargs):
        models_called.append(model)
        if model == "writer":
            return {"segments": [{
                "text": "Betelgeuse has about 16 solar masses.",
                "citations": [citation, "made-up-source"],
            }]}
        return {"verdict": "supported", "citations": [citation], "notes": "Matches the passage."}

    monkeypatch.setattr(main, "chat_json", fake_chat_json)
    response = client.post(
        f"/sessions/{session}/answer",
        json={"question": "What is Betelgeuse's mass?"},
    )
    assert response.status_code == 200, response.text
    segment = response.json()["segments"][0]
    assert segment["citations"] == [citation]
    assert segment["verdict"] == "supported"
    assert models_called == ["writer", "checker"]
    audit_id = response.json()["audit_log_id"]
    audit = client.get(f"/sessions/{session}/audit/{audit_id}")
    assert audit.status_code == 200
    assert audit.json()["details"]["segments"][0]["verdict"] == "supported"


def test_scholarly_search_marks_results_as_scholarly(monkeypatch) -> None:
    monkeypatch.setattr(main, "search_arxiv", lambda query, limit: [{
        "id": "https://arxiv.org/abs/1234.5678",
        "title": "A scientific paper",
        "abstract": "An abstract.",
        "updated": "2026-01-01T00:00:00Z",
        "authors": ["A. Author"],
        "source_tier": "t1_scholarly",
        "source": "arXiv",
    }])
    response = TestClient(app).get("/scholar/search", params={"q": "stellar evolution"})
    assert response.status_code == 200
    assert response.json()["results"][0]["source_tier"] == "t1_scholarly"


def test_retried_document_insert_is_idempotent(tmp_path) -> None:
    session_id, document_id = uuid4(), uuid4()
    document = {
        "id": str(document_id), "session_id": str(session_id), "filename": "retry.txt",
        "mime_type": "text/plain", "sha256": "a" * 64, "storage_key": "files/retry.txt",
        "parser": "utf8-text", "page_count": 1, "created_at": "2026-01-01T00:00:00+00:00",
    }
    chunks = [{"page": 1, "order": 0, "text": "A retry must not duplicate this passage."}]
    catalog = Catalog(tmp_path / "idempotent.sqlite3")
    catalog.add_document(document, chunks)
    catalog.add_document(document, chunks)

    hits = catalog.search(session_id, "duplicate passage", 5)
    assert len(hits) == 1
