# Beginner's Guide to This Project

This guide explains the project as it exists today, assuming you are new to both Python and AI application development. It covers what each important file is for, how the working parts connect, how to run and try them, and what the architecture diagram describes that has not been built yet.

## 1. What is this project?

The project is the beginning of a **scientific verification agent**. Its intended job is to answer scientific questions using evidence, keep track of where that evidence came from, check claims, and record changes. The architecture image shows the larger intended system.

The implemented software currently covers the first three steps:

1. **Input:** accept a question and files supplied by a user.
2. **Ingestion:** save files, extract text, split that text into pieces, and index those pieces.
3. **Retrieval:** find relevant pieces later and return them with source information.

It does **not** yet write a scientific answer, independently verify claims, retrieve papers from the web, fix claims, or run dependency checks. Those are later architecture stages.

## 2. A few words used in this guide

- **API:** a set of URLs that another program (or a person using the API page) can call.
- **Endpoint:** one such URL and its behavior. For example, the upload endpoint accepts files.
- **Session:** a conversation or work area identified by a random ID. This project uses it to keep one user's evidence search separate from another's.
- **Ingestion:** preparing supplied material so a program can search it. Here it means parsing, chunking, and indexing.
- **Parser:** code that reads a file format and extracts useful text.
- **Chunk:** a shorter piece of a document. Searching chunks is more useful than returning an entire long document.
- **Index:** a searchable structure built from the chunks. It helps the computer find relevant passages quickly.
- **Provenance:** information that tells you where a passage came from, such as its filename and page number.
- **SQLite:** a small database kept in a file. This project uses it so the current path can run without a database server.
- **FTS5 / BM25:** SQLite's full-text search and its ranking method. The ranking is based on matching words; it does not understand meaning the way an embedding model might.
- **UUID:** a long, randomly generated identifier. A session UUID groups requests; a document UUID identifies one uploaded file.

## 3. How one upload becomes a search result

Here is the path through the project:

```text
Question + file
      |
      v
FastAPI endpoint in src/agent/api/main.py
      |
      +--> save question and file metadata
      +--> save original file under a generated name
      |
      v
Parser in src/agent/ingestion/pipeline.py
      |
      +--> extract text and page numbers where available
      +--> split text into overlapping chunks
      |
      v
SQLite catalog and full-text index in src/agent/storage/catalog.py
      |
      v
Retrieval endpoint searches chunks for the same session
      |
      v
Ranked passages with filename, page, hash, and parser name
```

The original file is kept so there is a source artifact behind the extracted text. A SHA-256 hash is also saved: it is a compact fingerprint of the file contents and can help detect whether two files are identical. The hash does not prove that a document is scientifically correct.

## 4. Main folders and files

### Project root

- **`README.md`** — Short setup notes and example API requests. This guide explains the project in more detail.
- **`BEGINNER_GUIDE.md`** — This file.
- **`pyproject.toml`** — Project name, supported Python version, libraries, and developer tool settings.
- **`uv.lock`** — Exact dependency versions chosen by `uv`. It helps different computers install a consistent environment. Normally edit dependencies in `pyproject.toml`, then let `uv lock` update this file; do not hand-edit lock entries.
- **`docker-compose.yml`** — Describes Postgres with the pgvector extension and Redis. These services are part of the planned architecture, but the current upload/search implementation uses SQLite and does not connect to these containers.
- **`.env`** — Local configuration values. It may contain passwords, so it is excluded from Git. Do not paste real secrets into source code or commit them.
- **`.gitignore`** — Tells Git to skip local data, virtual environments, caches, secrets, and uploaded test documents.

### Application source: `src/agent/`

- **`__init__.py`** — Marks `agent` as a Python package. It is intentionally empty.
- **`config.py`** — Reads settings from environment variables and `.env`. It includes the local data folder, upload limit, and chunk length settings. It also contains connection settings for planned Postgres, Redis, and MinIO services. The current SQLite path uses `data_dir`; the other service settings are not currently used by ingestion/retrieval.
- **`api/main.py`** — Creates the FastAPI application and defines its HTTP endpoints and response shapes. This is the entry point used when launching the server.
- **`schemas/`** — Pydantic data models. These describe and validate structured concepts such as documents, chunks, pages, trust tiers, quantities, and visual objects. Some of these models describe the planned scientific agent and are not all connected to the current SQLite/API path yet.
- **`ingestion/pipeline.py`** — Knows how to read supported formats and split extracted text into chunks.
- **`storage/catalog.py`** — Creates and accesses the SQLite tables and full-text search index. It saves questions, document metadata, and text chunks, and performs session-scoped ranked search.
- **`api/`, `ingestion/`, `schemas/`, and `storage/` `__init__.py` files** — Mark folders as Python packages. They are currently empty; they do not run the application by themselves.

### Tests: `tests/`

- **`test_health.py`** — Checks that the health endpoint responds successfully.
- **`test_schemas.py`** — Checks that model validation accepts good data and rejects invalid data, such as an empty prompt or a backwards page bounding box.
- **`test_pipeline_api.py`** — Checks chunk overlap/page preservation, upload-to-retrieval behavior, session isolation, and that image uploads do not claim OCR text was extracted.
- **`fixtures/`** — Place for small sample files used in tests. Real/private documents should not be committed.
- **`__init__.py`** — Marks the tests folder as a Python package.

## 5. The API in plain language

### Health check

`GET /health` returns a small status object. It is a quick way to see that the API process is alive. It does not check that every future service (Postgres, Redis, or MinIO) is healthy.

### Upload a question and files

`POST /sessions/{session_id}/inputs` accepts a multipart form. “Multipart” is the standard web format for sending text fields and file bytes together.

- `session_id` is a UUID in the URL. Reuse the same value when you want to search the evidence from this upload. Use a different value for a separate session.
- `question` is optional text. If supplied, it is saved in the catalog.
- `files` can be repeated to attach more than one file.
- A request must include a question, a file, or both.

Example in PowerShell, from the project folder:

```powershell
$session = [guid]::NewGuid().ToString()
curl.exe -X POST "http://127.0.0.1:8000/sessions/$session/inputs" `
  -F "question=What does this paper say about stellar mass?" `
  -F "files=@paper.pdf"
```

Keep the value printed in `$session`; it is needed for the search request. The server response includes a document ID, the file's hash, which parser was used, page and chunk counts, and whether text was actually extracted.

### Search the session's evidence

`GET /sessions/{session_id}/retrieve?q=...&limit=5` searches only documents belonging to the given session. `q` is the search text. `limit` controls the maximum number of passages and must be from 1 to 20.

Example:

```powershell
curl.exe "http://127.0.0.1:8000/sessions/$session/retrieve?q=stellar%20mass&limit=5"
```

Each hit gives a text passage plus its filename, page, chunk order, parser name, and file hash. `score` is a search ranking value; do not read it as a probability or confidence in the passage. A returned passage is evidence to inspect, not a verified answer.

### Interactive API page

While the server is running, open `http://127.0.0.1:8000/docs` in a browser. FastAPI builds this page from the endpoint definitions. You can inspect request fields and try calls there.

## 6. What the file reader supports

The extension on the uploaded filename determines which parser is selected:

| File type | What happens | Current caveat |
| --- | --- | --- |
| PDF | PyMuPDF extracts text one page at a time. | A scanned page with no embedded text needs OCR, which is not configured. |
| DOCX | Paragraph text and table rows are extracted. | Original Word layout and exact table geometry are not preserved. |
| HTML / HTM | Visible page text is extracted; script and style content is skipped. | This is not a full browser renderer. |
| TXT / Markdown | Text is read as UTF-8, replacing invalid byte sequences if necessary. | Complex markup is treated as text. |
| PNG / JPG / JPEG / WEBP | Original image is stored and cataloged. | No OCR, chart-reading, or image-understanding model is connected, so there are no searchable text chunks. |

Each file is limited to 20 MiB by default. The code reads at most one byte beyond the configured limit so it can reject an oversized upload. The upload's original filename is kept as descriptive metadata, while the physical file path is generated by the server.

## 7. How chunks and search work

Text is divided into chunks of about 1,200 characters by default. The next chunk repeats about 180 characters from the end of the previous chunk. This overlap helps keep a sentence or a fact together if it happens to cross a chunk boundary. PDF page numbers remain attached to their chunks; DOCX, HTML, and plain-text files are currently treated as one page.

When an upload is stored, the catalog records the document row and inserts its chunks into a full-text index. At search time, the query is split into words and SQLite FTS5 ranks matching chunks with BM25. The session ID is part of the search filter, which prevents a search from returning another session's chunks.

This search is a practical first version, but it mostly finds shared words. For example, a passage about “stellar mass” may not rank well for “how heavy is the star?” Semantic embeddings and hybrid ranking are future improvements, not current behavior.

## 8. Where data goes

By default, local runtime data is written beneath `data/`:

```text
data/
  catalog.sqlite3       # questions, file metadata, and searchable chunks
  files/                # original uploads, named with generated UUIDs
```

`data/` is ignored by Git because uploaded evidence can be private and can take up space. Back it up if you need to preserve local data; deleting it removes the local catalog and uploaded originals. In production, the architecture expects a managed database and object store, with access control and retention policies.

Settings can be supplied as environment variables or in `.env`. The most relevant current settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `APP_ENV` | `dev` | Label returned by the health endpoint. |
| `DATA_DIR` | `data` | Folder for the SQLite catalog and original uploads. |
| `MAX_UPLOAD_BYTES` | `20971520` | Maximum bytes allowed for one upload (20 MiB). |
| `CHUNK_SIZE_CHARS` | `1200` | Approximate maximum characters per text chunk. |
| `CHUNK_OVERLAP_CHARS` | `180` | Characters repeated between neighboring chunks. |

Pydantic Settings maps uppercase environment variable names such as `DATA_DIR` onto Python setting fields such as `data_dir`.

## 9. Install and run it

You need Python 3.12 and `uv` installed. `uv` manages the Python project environment and dependency versions.

From the project folder in PowerShell:

```powershell
uv sync
uv run uvicorn agent.api.main:app --app-dir src --reload
```

The first command installs the libraries listed in `pyproject.toml`. The second starts the local development web server. `--reload` restarts the server when source code changes; use it for development, not as a production server configuration.

Stop the server with `Ctrl+C`. You do not need to start Docker Compose for the current SQLite-based upload/search flow.

## 10. Run the checks

From the project folder:

```powershell
uv run pytest
uv run ruff check src tests
```

Pytest runs the automated tests. Ruff checks code style and common mistakes. The tests use a temporary folder/database where needed so test uploads do not go into your regular `data/` directory.

## 11. Dependencies, in everyday terms

The main dependencies are in `pyproject.toml`:

- **FastAPI** creates the web API.
- **Uvicorn** runs the web server.
- **Pydantic** validates settings and request/response data.
- **python-multipart** lets FastAPI receive uploaded files with form fields.
- **PyMuPDF** reads PDF page text.
- **python-docx** reads DOCX paragraphs and tables.
- Python's built-in `sqlite3`, HTML parser, hashing, and file modules handle the local catalog, HTML/text reading, checksums, and file paths without extra packages.

Development-only tools include **pytest** for tests, **httpx** for API test requests, **Ruff** for linting, **mypy** for static type checks, and **pre-commit** for optional checks before Git commits.

## 12. The schemas and the intended architecture

The `schemas` folder includes richer scientific-agent data concepts:

- `InputItem` represents text or a stored file and checks that the required content exists.
- `Document` represents a source document and its parsing state and trust tier.
- `Chunk` represents a source passage, with optional page, bounding box, section, quantity, and astronomy-object information.
- `VisualObject` represents an image, figure, table, or equation region that could be cited later.
- `ContextBundle` groups the prompt, inputs, documents, chunks, and visuals for a session, and checks that chunks refer to known documents.
- `BoundingBox`, `Quantity`, and `ObjectMention` hold structured evidence details.
- Enums such as `SourceTier`, `ParseStatus`, and `ChunkKind` restrict fields to known categories.

These schemas are the vocabulary for future stages. The present SQLite implementation stores a simpler subset (prompt text, file metadata, page number, chunk order, and chunk text). It does not yet fill in bounding boxes, units, resolved astronomy object IDs, or all schema parse states.

Similarly, `docker-compose.yml` is a description of planned service infrastructure:

- **Postgres + pgvector** could store structured records and vector embeddings for semantic search.
- **Redis** could later support queues and caching.
- The architecture image also names **MinIO** for file/object storage, but MinIO is not currently defined in this Compose file or used by the implementation.

The app does not automatically use every service just because it appears in Compose or has a setting in `config.py`. A future integration must add the client code, schema/migrations, lifecycle management, and tests before those services are part of the working path.

## 13. What is still needed for the full architecture

To grow from this working first slice toward the diagram, likely next steps are:

1. Add a proper document/job state model so large files can be ingested in the background and safely retried.
2. Add OCR for scanned PDFs and images, then preserve figure/table regions and coordinates.
3. Move persistence to Postgres and file storage to an object store, with migrations and clear deletion/retention behavior.
4. Add embeddings and combine semantic search with full-text search; evaluate retrieval using representative scientific questions.
5. Add scholarly source connectors and source-tier rules, then generation with citations, independent verification, issue flags, and bounded fixes.
6. Add authentication, authorization, quotas, monitoring, and deployment safeguards before exposing the service beyond a trusted local environment.

These are not active features yet. Keeping this distinction clear helps avoid assuming that a box in the architecture diagram is already implemented in code.
