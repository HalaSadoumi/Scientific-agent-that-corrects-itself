"""Shared synchronous job that turns one stored upload into searchable chunks."""

from __future__ import annotations

import hashlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from agent.config import get_settings
from agent.ingestion.pipeline import chunk_pages_with_metadata, parse_file
from agent.retrieval.embeddings import embed_texts
from agent.storage.catalog import get_catalog
from agent.storage.objects import ObjectStore


def process_input_job(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    contents = ObjectStore(
        backend=settings.object_storage_backend,
        data_dir=Path(settings.data_dir),
        endpoint=(
            settings.s3_endpoint
            if "://" in settings.s3_endpoint
            else f"http://{settings.s3_endpoint}"
        ),
        bucket=settings.s3_bucket,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    ).load(payload["storage_key"])
    if hashlib.sha256(contents).hexdigest() != payload["sha256"]:
        raise ValueError("Stored upload checksum did not match the submitted file")

    with tempfile.TemporaryDirectory(prefix="science-ingest-") as temporary_directory:
        path = Path(temporary_directory) / Path(payload["filename"]).name
        path.write_bytes(contents)
        parsed = parse_file(
            path,
            payload["filename"],
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
            settings.ocr_enabled,
            settings.ocr_language,
        )
    pieces = chunk_pages_with_metadata(
        parsed.pages, settings.chunk_size_chars, settings.chunk_overlap_chars
    )
    vectors = None
    if pieces and settings.embedding_api_base:
        vectors = embed_texts(
            [text for _, text in pieces],
            settings.embedding_api_base,
            settings.embedding_api_key or "",
            settings.embedding_model,
        )
    document_id = UUID(payload["document_id"])
    session_id = UUID(payload["session_id"])
    get_catalog(settings).add_document(
        {
            "id": str(document_id),
            "session_id": str(session_id),
            "filename": payload["filename"],
            "mime_type": payload["mime_type"],
            "sha256": payload["sha256"],
            "storage_key": payload["storage_key"],
            "parser": parsed.parser,
            "page_count": len(parsed.pages),
            "created_at": datetime.now(UTC).isoformat(),
            "source_tier": payload.get("source_tier", "t0_user"),
        },
        [
            {
                "page": piece.page,
                "order": i,
                "text": piece.text,
                "start_offset": piece.start_offset,
                "end_offset": piece.end_offset,
                "kind": piece.kind,
                "embedding": vectors[i] if vectors else None,
            }
            for i, piece in enumerate(pieces)
        ],
    )
    return {
        "document_id": str(document_id),
        "parser": parsed.parser,
        "page_count": len(parsed.pages),
        "chunk_count": len(pieces),
        "text_extracted": any(page.text.strip() for page in parsed.pages),
        "ocr_status": parsed.ocr_status,
        "source_tier": payload.get("source_tier", "t0_user"),
    }
