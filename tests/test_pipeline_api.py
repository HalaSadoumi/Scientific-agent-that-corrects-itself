import io
from subprocess import CompletedProcess
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from agent.api import main
from agent.api.main import app
from agent.ingestion import pipeline
from agent.ingestion.pipeline import ExtractedPage, chunk_pages, chunk_pages_with_metadata
from agent.storage.catalog import Catalog


def test_chunking_preserves_page_and_overlap() -> None:
    source = "word " * 120
    chunks = chunk_pages((ExtractedPage(4, source),), size=200, overlap=30)
    assert len(chunks) > 1
    assert all(page == 4 for page, _ in chunks)
    assert chunks[0][1][-20:] in chunks[1][1]


def test_docx_tables_become_searchable_table_chunks(tmp_path) -> None:
    from docx import Document

    path = tmp_path / "astro_table.docx"
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Planet"
    table.cell(0, 1).text = "Orbital period"
    table.cell(1, 0).text = "Kepler-452b"
    table.cell(1, 1).text = "385 days"
    document.save(path)

    parsed = pipeline.parse_file(path, path.name, 1200, 100)
    chunks = chunk_pages_with_metadata(parsed.pages, 1200, 100)
    assert "Kepler-452b | 385 days" in parsed.pages[0].text
    assert any(chunk.kind == "table" and "385 days" in chunk.text for chunk in chunks)


def test_pdf_vector_table_becomes_searchable_table_chunk(tmp_path) -> None:
    import pymupdf as fitz

    path = tmp_path / "astro_table.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=300)
    cells = ((50, 50, 230, 95), (230, 50, 480, 95), (50, 95, 230, 140), (230, 95, 480, 140))
    for cell in cells:
        page.draw_rect(cell, width=1)
    page.insert_text((60, 78), "Object", fontsize=12)
    page.insert_text((240, 78), "Distance", fontsize=12)
    page.insert_text((60, 123), "NGC 1300", fontsize=12)
    page.insert_text((240, 123), "61 Mly", fontsize=12)
    pdf.save(path)
    pdf.close()

    parsed = pipeline.parse_file(path, path.name, 1200, 100)
    chunks = chunk_pages_with_metadata(parsed.pages, 1200, 100)
    assert "NGC 1300 | 61 Mly" in parsed.pages[0].text
    assert any(chunk.kind == "table" and "61 Mly" in chunk.text for chunk in chunks)


def test_image_ocr_uses_explicit_tesseract_path(tmp_path, monkeypatch) -> None:
    path = tmp_path / "chart.png"
    path.write_bytes(b"image-bytes")
    tesseract_path = tmp_path / "tesseract.exe"
    tesseract_path.touch()
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: None)
    called = {}

    def fake_run(command, **kwargs):
        called["command"] = command
        return CompletedProcess(command, 0, stdout=b"NGC 1300 galaxy image labels", stderr=b"")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    parsed = pipeline.parse_file(
        path, path.name, 1200, 100, ocr_enabled=True, ocr_language="eng",
        tesseract_cmd=str(tesseract_path),
    )
    assert called["command"][0].endswith("tesseract.exe")
    assert parsed.ocr_status == "completed"
    assert "NGC 1300" in parsed.pages[0].text


def test_upload_is_indexed_and_retrieval_is_session_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            data_dir=str(tmp_path),
            max_upload_bytes=1024 * 1024,
            chunk_size_chars=300,
            chunk_overlap_chars=30,
            ocr_enabled=True,
            ocr_language="eng",
            embedding_api_base=None,
            embedding_api_key=None,
            embedding_model="test",
            semantic_weight=0.5,
            object_storage_backend="local",
            s3_endpoint="127.0.0.1:3900",
            s3_bucket="uploads",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_region="garage",
            llm_api_base=None,
            llm_api_key=None,
            llm_model=None,
            verification_model=None,
        ),
    )
    client = TestClient(app)
    session = uuid4()
    from docx import Document

    docx = Document()
    table = docx.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Planet"
    table.cell(0, 1).text = "Orbital period"
    table.cell(1, 0).text = "Kepler-452b"
    table.cell(1, 1).text = "385 days"
    table_bytes = io.BytesIO()
    docx.save(table_bytes)
    docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    response = client.post(
        f"/sessions/{session}/inputs",
        data={"question": "What is the stellar mass?"},
        files=[
            ("files", ("paper.txt", b"The stellar mass is about 1.4 solar masses.", "text/plain")),
            ("files", ("table.docx", table_bytes.getvalue(), docx_type)),
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
    assert found.json()["retrieval_method"] == "sqlite_fts5_bm25"
    assert "keyword" in found.json()["route_reason"]

    table_hit = client.get(f"/sessions/{session}/retrieve", params={"q": "385 days"})
    assert table_hit.status_code == 200
    assert "Kepler-452b | 385 days" in table_hit.json()["hits"][0]["text"]
    assert table_hit.json()["hits"][0]["kind"] == "table"

    isolated = client.get(f"/sessions/{uuid4()}/retrieve", params={"q": "stellar mass"})
    assert isolated.status_code == 200
    assert isolated.json()["hits"] == []


def test_image_is_stored_without_claiming_ocr(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            data_dir=str(tmp_path),
            max_upload_bytes=1024 * 1024,
            chunk_size_chars=300,
            chunk_overlap_chars=30,
            ocr_enabled=True,
            ocr_language="eng",
            embedding_api_base=None,
            embedding_api_key=None,
            embedding_model="test",
            semantic_weight=0.5,
            object_storage_backend="local",
            s3_endpoint="127.0.0.1:3900",
            s3_bucket="uploads",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_region="garage",
            llm_api_base=None,
            llm_api_key=None,
            llm_model=None,
            verification_model=None,
        ),
    )
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
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            data_dir=str(tmp_path),
            max_upload_bytes=1024 * 1024,
            chunk_size_chars=300,
            chunk_overlap_chars=30,
            ocr_enabled=True,
            ocr_language="eng",
            embedding_api_base="http://local.test/v1",
            embedding_api_key="test-key",
            embedding_model="test",
            semantic_weight=0.5,
            object_storage_backend="local",
            s3_endpoint="127.0.0.1:3900",
            s3_bucket="uploads",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_region="garage",
            llm_api_base=None,
            llm_api_key=None,
            llm_model=None,
            verification_model=None,
        ),
    )
    monkeypatch.setattr(
        main, "create_embeddings", lambda texts, settings: [[1.0, 0.0] for _ in texts]
    )
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
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            data_dir=str(tmp_path),
            max_upload_bytes=1024 * 1024,
            chunk_size_chars=300,
            chunk_overlap_chars=30,
            ocr_enabled=True,
            ocr_language="eng",
            embedding_api_base=None,
            embedding_api_key=None,
            embedding_model="test",
            semantic_weight=0.5,
            object_storage_backend="local",
            s3_endpoint="127.0.0.1:3900",
            s3_bucket="uploads",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_region="garage",
            llm_api_base="http://local.test/v1",
            llm_api_key="test-key",
            llm_model="writer",
            verification_model="checker",
        ),
    )
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
            return {
                "segments": [
                    {
                        "text": "Betelgeuse has about 16 solar masses.",
                        "citations": [citation, "made-up-source"],
                    }
                ]
            }
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
    history = client.get(f"/sessions/{session}/history")
    assert history.status_code == 200
    assert history.json()["items"][0]["audit_log_id"] == audit_id
    assert history.json()["items"][0]["question"] == "What is Betelgeuse's mass?"
    assert history.json()["items"][0]["segments"][0]["verdict"] == "supported"
    isolated_history = client.get(f"/sessions/{uuid4()}/history")
    assert isolated_history.status_code == 200
    assert isolated_history.json()["items"] == []

    def fail_verifier(*, model, **kwargs):
        if model == "checker":
            raise main.ModelServiceError("Model provider returned HTTP 429: quota exceeded")
        return fake_chat_json(model=model, **kwargs)

    monkeypatch.setattr(main, "chat_json", fail_verifier)
    failed_answer = client.post(
        f"/sessions/{session}/answer",
        json={"question": "What is Betelgeuse's mass?"},
    )
    assert failed_answer.status_code == 502
    assert "HTTP 429: quota exceeded" in failed_answer.json()["detail"]


def test_scholarly_search_marks_results_as_scholarly(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "search_arxiv",
        lambda query, limit: [
            {
                "id": "https://arxiv.org/abs/1234.5678",
                "title": "A scientific paper",
                "abstract": "An abstract.",
                "updated": "2026-01-01T00:00:00Z",
                "authors": ["A. Author"],
                "source_tier": "t1_scholarly",
                "source": "arXiv",
            }
        ],
    )
    response = TestClient(app).get("/scholar/search", params={"q": "stellar evolution"})
    assert response.status_code == 200
    assert response.json()["results"][0]["source_tier"] == "t1_scholarly"


def test_retried_document_insert_is_idempotent(tmp_path) -> None:
    session_id, document_id = uuid4(), uuid4()
    document = {
        "id": str(document_id),
        "session_id": str(session_id),
        "filename": "retry.txt",
        "mime_type": "text/plain",
        "sha256": "a" * 64,
        "storage_key": "files/retry.txt",
        "parser": "utf8-text",
        "page_count": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    chunks = [{"page": 1, "order": 0, "text": "A retry must not duplicate this passage."}]
    catalog = Catalog(tmp_path / "idempotent.sqlite3")
    catalog.add_document(document, chunks)
    catalog.add_document(document, chunks)

    hits = catalog.search(session_id, "duplicate passage", 5)
    assert len(hits) == 1
