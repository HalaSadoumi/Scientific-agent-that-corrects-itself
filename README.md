# Scientific Verification Agent

This repository currently implements the **input → ingestion → retrieval** slice of the architecture. It accepts a question and evidence files, stores each original under a generated name, extracts searchable text, and returns source-linked passages for that session.

## Why these components

- **Input API:** `POST /sessions/{session_id}/inputs` accepts `question` and multiple `files` as multipart form fields. A caller creates and reuses a UUID session ID to keep evidence isolated between conversations. The upload name is metadata only; files are stored by generated UUID, with a SHA-256 digest for identity and audit checks.
- **Parsing and chunking:** `src/agent/ingestion/pipeline.py` extracts page text from PDF, paragraphs and tables from DOCX, visible text from HTML, and UTF-8 text/Markdown. Chunks retain page number, overlap adjacent passages to preserve boundary context, and are indexed only after successful parsing. Images are stored and identified but not OCR processed; an OCR/vision service is not configured yet.
- **Persistence and retrieval:** `src/agent/storage/catalog.py` stores document metadata and chunks in a local SQLite catalog. SQLite FTS5 BM25 gives a useful lexical baseline without requiring services to be running. Search is session-scoped and returns the original filename, hash, parser, page, and passage so downstream generation can cite its evidence.

Postgres/pgvector and Redis are already described in Compose, but no embedding model, database client, migrations, or job worker existed in the scaffold. This slice uses SQLite and synchronous parsing to make the requested path runnable now. Moving to vector/hybrid retrieval, asynchronous jobs, web/scholarly source tiers, OCR, and the later generation/verification stages remains follow-on work.

## Run locally

Install dependencies with `uv sync`, then start the API with `uv run uvicorn agent.api.main:app --app-dir src --reload`. Uploaded originals and `data/catalog.sqlite3` are written under `data/` by default. Set `DATA_DIR` to change that location and `MAX_UPLOAD_BYTES` to change the per-file limit (default 20 MiB).

Example upload (PowerShell):

```powershell
$session = [guid]::NewGuid().ToString()
curl.exe -X POST "http://127.0.0.1:8000/sessions/$session/inputs" `
  -F "question=What does the paper say about stellar mass?" `
  -F "files=@paper.pdf"
```

Then retrieve evidence with `GET /sessions/{session_id}/retrieve?q=stellar%20mass&limit=5`. `/docs` exposes the interactive API schema. Retrieval results are evidence passages, not a generated or verified scientific answer.

## Limits to account for

- Image OCR and figure/table understanding are not implemented.
- Scanned PDFs may produce no text unless they already contain an OCR text layer.
- Lexical BM25 will miss paraphrases; it is not semantic or hybrid retrieval.
- Parsing is synchronous and intended for modest uploads; production deployment needs background jobs, authentication/authorization, quotas, durable object storage, migrations, and monitoring.
