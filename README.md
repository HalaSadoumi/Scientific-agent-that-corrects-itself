# Scientific Verification Agent

This repository implements a configurable evidence pipeline: **input → ingestion → retrieval → answer drafting → separate claim checks → audit record**. See [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) for a detailed explanation of every project area and setting.

## What works now

- Upload a question and PDF, DOCX, HTML, TXT, Markdown, or image files to a session.
- Extract document text, preserve PDF page numbers, and use Tesseract OCR for images or textless PDF pages when Tesseract is installed.
- Search evidence with SQLite FTS5 BM25; optionally add semantic embeddings and hybrid rank fusion through an explicitly configured OpenAI-compatible endpoint.
- Plan focused query clauses, rerank by query coverage and caller-supplied source tier, and briefly cache session searches.
- Select SQLite/local files for development or Postgres/pgvector plus Garage S3-compatible object storage for a service-backed setup.
- Queue larger uploads through Redis Streams, run a worker with up to three attempts, and poll job status.
- Search arXiv abstracts and request an answer composed from evidence, checked by a separately named verification model with a second retrieval pass. Generated citations are restricted to sources actually returned by retrieval, and the result is saved to the audit log.

API details are available at `/docs` when the server is running.

## Local setup

Use Python 3.12 and `uv`. Copy `.env.example` to `.env`, adjust local ports or credentials as needed, then:

```powershell
uv sync
uv run uvicorn agent.api.main:app --app-dir src --reload
```

The simplest setup needs no containers: SQLite and local file storage are the defaults. Uploaded originals and the catalog are written beneath `data/`.

To use the service-backed setup, set `STORAGE_BACKEND=postgres` and `OBJECT_STORAGE_BACKEND=s3` in `.env`, then start services:

```powershell
docker compose up -d postgres redis garage
```

If a port is already in use, change the matching host port in `.env`. Start the API in one terminal and the worker in another:

```powershell
uv run uvicorn agent.api.main:app --app-dir src --reload
uv run python -m agent.ingestion.worker
```

Postgres initializes its tables and pgvector extension on first use. Garage creates the configured bucket on startup. The single-node Garage setup is for local development and has no data redundancy.

## Common API calls

Upload synchronously:

```powershell
$session = [guid]::NewGuid().ToString()
curl.exe -X POST "http://127.0.0.1:8000/sessions/$session/inputs" `
  -F "question=What does the paper say about stellar mass?" `
  -F "files=@paper.pdf"
```

Queue files for background processing instead:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/sessions/$session/input-jobs" `
  -F "files=@paper.pdf"
```

The queued response returns a job ID. Poll `GET /sessions/{session_id}/input-jobs/{job_id}` until the status is `completed` or `failed`.

Search ingested passages with `GET /sessions/{session_id}/retrieve?q=stellar%20mass&limit=5`. Search public scholarly metadata with `GET /scholar/search?q=stellar%20mass`. Generate and verify a response with `POST /sessions/{session_id}/answer` and JSON such as `{"question":"What is the measured mass?","include_scholarly":false}`. Answer generation requires an LLM endpoint, an answer model, and a different `VERIFICATION_MODEL`.

Uploads accept an optional multipart `source_tier` (`t0_user`, `t1_scholarly`, `t2_reference`, or `t3_web`; default `t0_user`). This label is caller-provided and is not independently verified. Chunks retain page-relative character offsets and a coarse content kind; DOCX and detectable PDF tables are indexed with row/cell separators. OCR can extract words from embedded PDF figure images when Tesseract is available, but does not interpret chart geometry or visual meaning.

## Optional integrations and data sharing

- OCR needs the Tesseract program and language data installed on the computer. It is separate from Python dependencies.
- Semantic retrieval requires `EMBEDDING_API_BASE` and, for authenticated providers, `EMBEDDING_API_KEY`. When enabled, uploaded text chunks and search queries are sent to that endpoint.
- Answering requires `LLM_API_BASE`, `LLM_API_KEY` (if required by the provider), `LLM_MODEL`, and `VERIFICATION_MODEL`. Evidence is sent to the configured model service. Keep those settings unset if evidence must remain local.
- arXiv lookup contacts the public arXiv API only when the scholarly search route is called or `include_scholarly` is requested.

## Current limits

- OCR reads visible words, not chart meaning or plotted values; scanned-document OCR quality depends on the image and installed Tesseract language data.
- The Postgres vector query currently uses exact search without an approximate-nearest-neighbor index. It is suitable for a first deployment, not a large corpus.
- The schema is initialized in code; versioned migrations and a full data-deletion workflow are not yet implemented.
- The job queue has retry and pending-message recovery behavior, but the worker is a separate process that must be run. Parsing is still CPU-bound and should be load-tested for the target document sizes.
- The verifier is a second configured model call over a separate retrieval pass, not a scientific proof system. It can still make mistakes. Automatic surgical correction and cascade/dependency checks are not implemented.
- API authentication, per-user authorization, request quotas, and production monitoring still need to be added before exposing the service to untrusted users.

Run checks with `uv run pytest` and `uv run ruff check src tests`.
