from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from agent.config import get_settings
from agent.ingestion.pipeline import SUPPORTED_TYPES, chunk_pages, parse_file
from agent.storage.catalog import Catalog

app = FastAPI(title="Scientific Verification Agent", version="0.1.0")


class IngestedDocument(BaseModel):
    id: UUID
    filename: str
    mime_type: str
    sha256: str
    parser: str
    page_count: int
    chunk_count: int
    text_extracted: bool


class IngestResponse(BaseModel):
    session_id: UUID
    question: str | None
    documents: list[IngestedDocument]


class RetrievalHit(BaseModel):
    document_id: UUID
    filename: str
    page: int
    chunk_order: int
    text: str
    score: float
    source_tier: str = "t0_user"
    sha256: str
    parser: str


class RetrievalResponse(BaseModel):
    session_id: UUID
    query: str
    hits: list[RetrievalHit]
    retrieval_method: str = "sqlite_fts5_bm25"


def _catalog() -> Catalog:
    settings = get_settings()
    return Catalog(Path(settings.data_dir) / "catalog.sqlite3")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": get_settings().app_env}


@app.post("/sessions/{session_id}/inputs", response_model=IngestResponse)
async def ingest_inputs(
    session_id: UUID,
    question: str | None = Form(default=None, max_length=20_000),
    files: list[UploadFile] | None = File(default=None),  # noqa: B008
) -> IngestResponse:
    """Accept a question and uploaded evidence, persist originals, parse, and index text."""
    settings = get_settings()
    files = files or []
    if not (question and question.strip()) and not files:
        raise HTTPException(
            status_code=422, detail="Provide a question, at least one file, or both"
        )
    if question and question.strip():
        try:
            _catalog().add_prompt(session_id, question.strip(), datetime.now(UTC).isoformat())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not save the question") from exc
    results: list[IngestedDocument] = []
    for upload in files:
        filename = Path(upload.filename or "upload").name[:255]
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type: {suffix or '(no extension)'}"
            )
        contents = await upload.read(settings.max_upload_bytes + 1)
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_upload_bytes} byte limit",
            )
        if not contents:
            raise HTTPException(status_code=422, detail=f"File is empty: {filename}")

        document_id = uuid4()
        digest = hashlib.sha256(contents).hexdigest()
        file_path = Path(settings.data_dir) / "files" / f"{document_id}{suffix}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(contents)
        try:
            parsed = parse_file(
                file_path, filename, settings.chunk_size_chars, settings.chunk_overlap_chars
            )
        except Exception as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422, detail=f"Could not parse {filename}: {exc}"
            ) from exc
        pieces = chunk_pages(parsed.pages, settings.chunk_size_chars, settings.chunk_overlap_chars)
        mime_type = SUPPORTED_TYPES[suffix]
        record = {
            "id": str(document_id), "session_id": str(session_id), "filename": filename,
            "mime_type": mime_type, "sha256": digest,
            "storage_key": str(file_path.relative_to(Path(settings.data_dir))),
            "parser": parsed.parser, "page_count": len(parsed.pages),
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            _catalog().add_document(record, [{"page": page, "order": i, "text": text}
                                              for i, (page, text) in enumerate(pieces)])
        except Exception as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503, detail="Could not save the input to the catalog"
            ) from exc
        results.append(IngestedDocument(
            id=document_id, filename=filename, mime_type=mime_type, sha256=digest,
            parser=parsed.parser, page_count=len(parsed.pages), chunk_count=len(pieces),
            text_extracted=any(page.text.strip() for page in parsed.pages),
        ))
    return IngestResponse(session_id=session_id, question=question.strip() if question else None,
                          documents=results)


@app.get("/sessions/{session_id}/retrieve", response_model=RetrievalResponse)
def retrieve(
    session_id: UUID,
    q: str = Query(min_length=1, max_length=2_000),
    limit: int = Query(default=5, ge=1, le=20),
) -> RetrievalResponse:
    hits = _catalog().search(session_id, q, limit)
    return RetrievalResponse(session_id=session_id, query=q, hits=hits)
