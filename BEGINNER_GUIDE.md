# Beginner's Guide to This Project

This guide explains the project as it exists today, assuming you are new to Python and AI application development. It covers what each important file is for, how the working parts connect, how to run and try them, and which architecture features remain unfinished.

## 1. What is this project?

The project is a configurable **scientific verification agent**. Its intended job is to answer scientific questions using evidence, keep track of where that evidence came from, check claims, and record changes. Some features need a local service or an external model provider configured before they can run.

The implemented software covers these steps:

1. **Input:** accept a question and files supplied by a user.
2. **Ingestion:** save files, extract text (including optional OCR), split that text into pieces, and index those pieces.
3. **Retrieval:** find relevant pieces with keyword search and optional semantic search, and return them with source information.
4. **Answer and verification:** optionally search arXiv, draft cited segments with a configured model, retrieve evidence again for each claim, ask a separately named model to check it, and save an audit record.

It does **not** automatically patch claims marked unsupported or run cascade/dependency checks. A separate model call is not a proof of scientific correctness. API authentication and per-user authorization are also not implemented.

The astronomy-focused version can optionally use a local AstroBERT embedding model and a Neo4j graph database. These are optional because they need extra packages and setup. The app still runs with its dependable BM25 keyword-search baseline when they are not configured.

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

## 3. How uploads become searchable

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
Synchronous ingestion or Redis Streams worker
      |
      v
Catalog stores chunks with filename, page, hash, and parser name
      |
      v
Retrieval endpoint searches only this session's chunks
```

The original file is kept so there is a source artifact behind the extracted text. A SHA-256 hash is also saved: it is a compact fingerprint of the file contents and can help detect whether two files are identical. The hash does not prove that a document is scientifically correct.

## 4. Main folders and files

### Project root

- **`README.md`** — Short setup notes and example API requests. This guide explains the project in more detail.
- **`BEGINNER_GUIDE.md`** — This file.
- **`pyproject.toml`** — Project name, supported Python version, libraries, and developer tool settings.
- **`uv.lock`** — Exact dependency versions chosen by `uv`. It helps different computers install a consistent environment. Normally edit dependencies in `pyproject.toml`, then let `uv lock` update this file; do not hand-edit lock entries.
- **`docker-compose.yml`** — Describes Postgres with pgvector, Redis, and Garage (S3-compatible object storage). Postgres and Garage can be selected as storage backends; Redis powers optional background ingestion.
- **`.env.example`** — Safe starter settings. Copy it to `.env`, set local ports and credentials, and keep `.env` out of Git.
- **`.env`** — Local configuration values. It may contain passwords, so it is excluded from Git. Do not paste real secrets into source code or commit them.
- **`.gitignore`** — Tells Git to skip local data, virtual environments, caches, secrets, and uploaded test documents.

### Application source: `src/agent/`

- **`__init__.py`** — Marks `agent` as a Python package. It is intentionally empty.
- **`config.py`** — Reads settings from environment variables and `.env`. It includes local data paths, upload and chunk limits, optional OCR/model settings, and Postgres, Redis, and S3-compatible storage connections.
- **`api/main.py`** — Creates the FastAPI application and defines its HTTP endpoints and response shapes. This is the entry point used when launching the server.
- **`schemas/`** — Pydantic data models. These describe and validate structured concepts such as documents, chunks, pages, trust tiers, quantities, and visual objects. Some of these models describe the planned scientific agent and are not all connected to the current SQLite/API path yet.
- **`ingestion/pipeline.py`** — Knows how to read supported formats, optionally ask Tesseract to read images/scanned pages, and split extracted text into chunks.
- **`ingestion/jobs.py`** — Performs one queued ingestion job: load the original file, parse it, create chunks, optionally create embeddings, and save it to the catalog.
- **`ingestion/worker.py`** — Runs as a separate process, reads Redis Stream jobs, updates their status, retries failures up to three attempts, and recovers abandoned jobs.
- **`retrieval/embeddings.py`** — Optional client for an OpenAI-compatible embeddings endpoint. It converts text into numeric vectors so related phrases can be found even when they do not share words. It is inactive unless an endpoint is configured.
- **`retrieval/scholar.py`** — Searches arXiv metadata and abstracts when the scholarly search endpoint is called.
- **`generation.py`** — Optional OpenAI-compatible chat client for answer drafting and claim verification.
- **`storage/catalog.py`** — Provides SQLite and Postgres/pgvector catalogs for questions, document metadata, chunks, audit entries, and session-scoped search.
- **`storage/objects.py`** — Saves, reads, and deletes original files in the local data folder or an S3-compatible object store such as Garage.
- **`api/`, `ingestion/`, `schemas/`, and `storage/` `__init__.py` files** — Mark folders as Python packages. They are currently empty; they do not run the application by themselves.

### Tests: `tests/`

- **`test_health.py`** — Checks that the health endpoint responds successfully.
- **`test_schemas.py`** — Checks that model validation accepts good data and rejects invalid data, such as an empty prompt or a backwards page bounding box.
- **`test_pipeline_api.py`** — Checks chunking, upload/retrieval, session isolation, OCR status, mocked semantic search, citation validation, audit records, and scholarly source labels.
- **`fixtures/`** — Place for small sample files used in tests. Real/private documents should not be committed.
- **`__init__.py`** — Marks the tests folder as a Python package.

## 5. The API in plain language

### Health check

`GET /health` returns a small status object. It shows that the API process is alive, but it does not check that Postgres, Redis, object storage, OCR, or model services are all healthy.

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

### Queue a larger upload

`POST /sessions/{session_id}/input-jobs` stores each original and places a job in Redis. It returns HTTP status `202` (accepted) and a job ID before parsing is finished. The separate worker must be running for queued jobs to complete:

```powershell
uv run python -m agent.ingestion.worker
```

Check progress with `GET /sessions/{session_id}/input-jobs/{job_id}`. The status becomes `completed` or `failed`; a completed result includes document ID, parser, page and chunk counts, and OCR status. Failed jobs are attempted up to three times. Both API and worker need access to the same Redis, catalog, and original-file store.

### Search the session's evidence

`GET /sessions/{session_id}/retrieve?q=...&limit=5` searches only documents belonging to the given session. `q` is the search text. `limit` controls the maximum number of passages and must be from 1 to 20.

Example:

```powershell
curl.exe "http://127.0.0.1:8000/sessions/$session/retrieve?q=stellar%20mass&limit=5"
```

Each hit gives a text passage plus its filename, page, chunk order, parser name, and file hash. `score` is a search ranking value; do not read it as a probability or confidence in the passage. A returned passage is evidence to inspect, not a verified answer.

### Search scholarly metadata

`GET /scholar/search?q=...&limit=5` asks the public arXiv metadata API for papers and abstracts. Results are marked `t1_scholarly` and include arXiv links. This endpoint contacts arXiv only when called; it does not download full papers or automatically add results to the session catalog.

### Draft and verify an answer

`POST /sessions/{session_id}/answer` accepts JSON such as:

```json
{"question":"What is the measured mass?","include_scholarly":false}
```

It retrieves session evidence, sends the evidence to the configured generation model, checks that its citations refer to sources the app actually retrieved, retrieves evidence again for each generated segment, and asks the configured verification model for a verdict. The result contains claim text, citations, verification citations, and a verdict of `supported`, `contradicted`, `uncertain`, or `unsupported`. The audit ID can be used with `GET /sessions/{session_id}/audit/{audit_id}` to read the saved record.

Answering requires `LLM_API_BASE`, `LLM_MODEL`, and `VERIFICATION_MODEL`; set `LLM_API_KEY` if your provider needs a key. The two model names must differ. The evidence passages go to that model service. Setting `include_scholarly` also sends the question and claim searches to arXiv. A second model and retrieval pass can still make mistakes, so treat the verdict as a review aid rather than proof.

### Interactive API page

While the server is running, open `http://127.0.0.1:8000/docs` in a browser. FastAPI builds this page from the endpoint definitions. You can inspect request fields and try calls there.

### Dashboard pages

Open `http://127.0.0.1:8000/` for the app-style interface. The left navigation changes between six views while keeping the same session:

- **Chat** asks a question, previews retrieved context, displays answer claims, and shows the verifier's verdicts, citations, and audit record link. **History** lists saved questions and their verified results for the current session.
- **Pipeline** uploads supported sources and reports the parser, pages, chunks, and OCR status returned by ingestion.
- **Metrics** reads real per-session document, page, chunk, word, and vector counts from the catalog. It also lists document parsers and chunk kinds.
- **Graph** visualizes the current session's Neo4j entity and co-occurrence snapshot. When Neo4j is not configured, it explains how to enable it rather than drawing sample data.
- **Evidence** runs normal BM25/vector retrieval and graph-only retrieval side by side, with source filename, page, chunk, and trust label.
- **Policy** describes the real deterministic routing rules and the most recent route. It does not show made-up Q-learning scores; that learning feature is not implemented.

The right panel is a session snapshot. Its metrics are fetched from `/sessions/{session_id}/snapshot`, graph counts from `/sessions/{session_id}/graph`, and safe configuration indicators from `/capabilities`. The credentials themselves are never returned to the browser. `/sessions/{session_id}/graph/search` returns graph-only evidence for the Evidence view, and `/routing/policy` describes the route rules.

## 6. What the file reader supports

The extension on the uploaded filename determines which parser is selected:

| File type | What happens | Current caveat |
| --- | --- | --- |
| PDF | PyMuPDF extracts text one page at a time, detects many vector-grid tables, OCRs scanned pages, and OCRs embedded image regions when available. | Borderless or unusually laid out tables may be missed; OCR of an image-based table returns words but may not preserve its rows and columns. |
| DOCX | Paragraph text and table rows are extracted. | Original Word layout and exact table geometry are not preserved. |
| HTML / HTM | Visible page text is extracted; script and style content is skipped. | This is not a full browser renderer. |
| TXT / Markdown | Text is read as UTF-8, replacing invalid byte sequences if necessary. | Complex markup is treated as text. |
| PNG / JPG / JPEG / WEBP | Original image is stored. If Tesseract is installed, it can extract visible words into a searchable chunk. | OCR reads words; it does not understand a chart's plotted data or visual meaning. |

Tesseract is a separate computer program, not a Python package. It is now installed on this Windows computer, and the local `.env` points to its executable using `TESSERACT_CMD`. The `capabilities` endpoint checks that configured path, so the dashboard can show whether the OCR program is actually available. On another computer, install Tesseract and English language data, then set `TESSERACT_CMD` to that machine's `tesseract.exe` path. OCR text becomes ordinary searchable evidence; Tesseract does not understand a chart's plotted curve, axis meaning, or visual trend.

Each file is limited to 20 MiB by default. The code reads at most one byte beyond the configured limit so it can reject an oversized upload. The upload's original filename is kept as descriptive metadata, while the physical file path is generated by the server.

## 7. How chunks and search work

Text is divided into chunks of about 1,200 characters by default. The next chunk repeats about 180 characters from the end of the previous chunk. This overlap helps keep a sentence or a fact together if it happens to cross a chunk boundary. PDF page numbers remain attached to their chunks; DOCX, HTML, and plain-text files are currently treated as one page.

When an upload is stored, the catalog records the document row and inserts its chunks into a full-text index. Chunk records include page-relative character offsets and a coarse kind (`paragraph`, `table`, or `equation`). DOCX tables and detected vector-grid PDF tables are represented with cell separators, classified as table chunks, indexed, and returned with that kind in retrieval results. Tables that exist only as pixels rely on OCR, which can read labels and values but cannot guarantee the original cell structure. At search time, a lightweight planner keeps the original query and may split it into up to three focused clauses. SQLite FTS5 ranks matching chunks with BM25; PostgreSQL uses full text search. If an embeddings endpoint is explicitly configured, planned queries are also embedded, and keyword and semantic rankings are combined using reciprocal-rank fusion. A final reranker considers query-term coverage, exact phrase matches, and a small source-tier preference. Results are cached per session for up to five seconds. The session ID is part of every search and cache key.

Without an embeddings endpoint, search mostly finds shared words. With one configured, it can find related phrases, but results still depend on the model and should be evaluated. Source tiers on uploaded documents are caller-supplied labels, not independent trust assessments. A ranking score is a heuristic, not a probability or scientific confidence value.

To enable embeddings, set `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`, and optionally `EMBEDDING_MODEL`. The base URL should include the API prefix (commonly `/v1`); the client calls `/embeddings` under that base. **When enabled, document chunks and search queries leave the computer and go to that provider.** Do not enable it for private material unless the provider is appropriate. With no endpoint set, text and search stay local.

## 8. Where data goes

With the local defaults, runtime data is written beneath `data/`:

```text
data/
  catalog.sqlite3       # questions, file metadata, and searchable chunks
  files/                # original uploads, named with generated UUIDs
```

`data/` is ignored by Git because uploaded evidence can be private and can take up space. Back it up if you need to preserve local data; deleting it removes the local catalog and uploaded originals. With Postgres and Garage selected, database data and originals instead live in Docker volumes. Garage is a single-node local development setup and has no redundancy.

Settings can be supplied as environment variables or in `.env`. The most relevant current settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `APP_ENV` | `dev` | Label returned by the health endpoint. |
| `DATA_DIR` | `data` | Folder for the SQLite catalog and original uploads. |
| `MAX_UPLOAD_BYTES` | `20971520` | Maximum bytes allowed for one upload (20 MiB). |
| `CHUNK_SIZE_CHARS` | `1200` | Approximate maximum characters per text chunk. |
| `CHUNK_OVERLAP_CHARS` | `180` | Characters repeated between neighboring chunks. |
| `OCR_ENABLED` | `true` | Whether to try Tesseract on images and PDF pages with no extracted text. |
| `OCR_LANGUAGE` | `eng` | Tesseract language code; matching language data must be installed. |
| `EMBEDDING_API_BASE` | unset | Enables embeddings when set, usually to an OpenAI-compatible `/v1` API base. |
| `EMBEDDING_API_KEY` | unset | Credential sent to the configured embedding endpoint. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name sent to the embeddings service. |
| `EMBEDDING_BACKEND` | `auto` | `auto` preserves API URL behavior; choose `local` for AstroBERT, `api` for hosted vectors, or `none` for BM25 only. |
| `LOCAL_EMBEDDING_MODEL` | `NeuML/astrobert-small-embeddings` | Astronomy model downloaded and run on the API computer when the local backend is selected. |
| `SEMANTIC_WEIGHT` | `0.5` | Semantic share in hybrid ranking, from `0` (keyword) to `1` (semantic). |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | unset | Optional Neo4j graph connection; unset means graph routing and graph indexing stay off. |
| `NEO4J_DATABASE` | `neo4j` | Database name on the configured graph server. |
| `LLM_API_BASE` | unset | Enables answer drafting and verification through an OpenAI-compatible endpoint. |
| `LLM_API_KEY` | unset | Provider credential; needed only if the provider requires it. |
| `LLM_MODEL` | unset | Model used to draft answers. |
| `VERIFICATION_MODEL` | unset | A different model name used for claim checks. |
| `STORAGE_BACKEND` | `sqlite` | `sqlite` for the local file catalog or `postgres` for Postgres/pgvector. |
| `OBJECT_STORAGE_BACKEND` | `local` | `local` for files beneath `data/` or `s3` for Garage/S3-compatible storage. |
| `POSTGRES_HOST` | `127.0.0.1` | Postgres server name/address from the API host. |
| `POSTGRES_PORT` | `5432` | Postgres server port. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | development defaults | Postgres login and database. Set these to match Compose or your database. |
| `REDIS_HOST` | `127.0.0.1` | Redis server address. `localhost` is normalized to IPv4 on Windows. |
| `REDIS_PORT` | `6380` in `.env.example` | Host port mapped to Redis container port 6379. |
| `S3_ENDPOINT` | `127.0.0.1:3900` | Garage/S3-compatible server address. |
| `S3_BUCKET` | `uploads` | Bucket where original files are kept. |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | development placeholders | Garage credentials. Replace them before sharing or deploying. |
| `S3_REGION` | `garage` | Region value expected by the local Garage service. |

Pydantic Settings maps uppercase environment variable names such as `DATA_DIR` onto Python setting fields such as `data_dir`.

### Optional Postgres, Redis, and Garage

The local defaults need no containers. To use server storage and queued ingestion, install Docker Desktop, copy `.env.example` to `.env`, set private credentials, and run:

```powershell
docker compose up -d postgres redis garage
```

Set `STORAGE_BACKEND=postgres` and `OBJECT_STORAGE_BACKEND=s3` in `.env` before starting the API. The Postgres backend creates the `vector` extension and tables on first use. Garage creates the configured default bucket on startup. Start the API and the worker in separate terminals; the worker reads Redis Stream jobs. If host port 6379 is already in use, set `REDIS_PORT=6380`; this repository's `.env.example` uses 6380 for that reason.

The architecture diagram specifies MinIO, but the configured Docker Hub image could not be pulled in this environment. The local Compose stack now uses Garage, which provides an S3-compatible interface. The running Postgres/pgvector, Garage, and Redis setup was smoke-tested; the singleton Garage service is for development and has no redundancy.

## 9. Install and run it

You need Python 3.12 and `uv` installed. `uv` manages the Python project environment and dependency versions.

From the project folder in PowerShell:

```powershell
uv sync
uv run uvicorn agent.api.main:app --app-dir src --reload
```

The first command installs the libraries listed in `pyproject.toml`. The second starts the local development web server. `--reload` restarts the server when source code changes; use it for development, not as a production server configuration.

Stop the server and worker with `Ctrl+C`. The synchronous SQLite/local upload/search path does not need Docker. The background job endpoint does need Redis and a worker; Postgres and Garage are optional alternatives to local persistence.

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

These schemas are the vocabulary for richer future evidence. The current database stores the fields used by this path: session, prompt, file metadata, parser, page, chunk order, chunk text, optional vectors, and answer-audit details. It does not yet fill bounding boxes, normalized quantities, resolved astronomy IDs, or every parse-state field from the schemas.

The services are connected when selected by configuration:

- **Postgres + pgvector** stores documents, prompts, chunks, vectors, and audit records. PostgreSQL full-text search uses a GIN index; vector search currently scans the matching session's vectors exactly, without an approximate-nearest-neighbor index.
- **Redis Streams** carries background ingestion jobs and stores their progress/status values.
- **Garage** stores original uploads through its S3-compatible API. It replaces the MinIO image in this local Compose setup because the configured MinIO Docker Hub image could no longer be pulled. The code uses the general S3 API and can be pointed at another compatible provider.

The local development setup uses one Garage node with no redundancy. The Postgres catalog creates its schema in code, not through versioned migrations. Those are reasons to treat this as a development implementation rather than a production deployment.

## 13. The astronomy agent: route, graph, answer, and verification

### Choosing a retrieval method for each question

`src/agent/retrieval/routing.py` is the small, transparent “agent” that selects a retrieval route. It examines the words and astronomy identifiers in the question and returns both a method name and a human-readable reason. It does not ask an LLM to choose, so a result is repeatable and does not spend extra API calls.

- A relationship question (for example, how an object relates to an observation) uses **graph plus regular retrieval** when Neo4j is configured.
- When semantic vectors are available, an ordinary question uses **hybrid BM25 + vector search**.
- When they are not, it uses **SQLite BM25 keyword search**. That route works locally without an API key.

This is an initial policy, not a learned decision system. The interface and retrieval response show the selected method/reason. Once a labeled astronomy evaluation set exists, the policy can be compared against alternatives. Q-learning should wait until feedback is tied to evidence correctness rather than user satisfaction alone.

### Free astronomy embeddings

`src/agent/retrieval/embeddings.py` implements two explicitly selected choices. `EMBEDDING_BACKEND=local` loads `NeuML/astrobert-small-embeddings` through Sentence Transformers and embeds both document chunks and search queries on the API computer. This is free of embedding API charges and keeps text local, but it needs a one-time model download plus local disk, memory, and CPU. Install the optional package group with `uv sync --extra astro-embeddings`. `EMBEDDING_BACKEND=api` uses the configured OpenAI-compatible service and sends chunk/query text to that service. `none` keeps search keyword-only.

The same model must be used at ingestion and search. If you change embedding model after documents already have vectors, re-ingest those documents; different models may use different vector sizes or vector meaning. The app falls back to BM25 only when embeddings are explicitly off. A configured embedding failure is surfaced as an error instead of quietly pretending semantic search succeeded.

The local astronomy model was compared in `benchmarks/ASTRONOMY_EMBEDDINGS_REPORT.md` using a public astronomy abstract benchmark. That test measures retrieval of relevant paper abstracts; it does not prove exact passage accuracy, answer quality, OCR quality, or scientific correctness. Run a separate evaluation on the project's target documents before presenting it as an accuracy guarantee.

### Neo4j graph retrieval

`src/agent/retrieval/graph.py` is an optional Neo4j connection. The catalog remains the source of full chunk text and ordinary BM25/vector search; Neo4j stores searchable evidence chunks plus recognizable astronomy entities and their links to the passages that mention them. It can also link co-occurring entities with the ID of the passage that supports that link. A graph hit therefore includes the source filename, page, chunk number, and text rather than being an unsupported fact on its own.

Entity extraction currently recognizes a conservative list of astronomical catalog identifiers, named celestial objects, and instruments. It is rule-based, not a general scientific knowledge extractor. Unrecognized names will not become graph entities, and co-occurrence alone does not prove causation. The graph route is only active when `NEO4J_URI` is set. Install the connector with `uv sync --extra graph`; then set the URI, username, password, and database for your Neo4j instance. The connector uses `truststore` so TLS certificate verification follows the computer's trusted certificate store. A free Neo4j Aura instance keeps the graph outside this computer, but graph data sent there is still external data sharing. Without Neo4j settings, the rest of the application remains available.

### Answer generation and independent claim checking

The web page now has a **Draft and verify** action. The backend uses retrieved passages to ask the configured writing model to produce short, separately checkable claims. It rejects citations that do not match IDs in its retrieved evidence. For every claim it runs retrieval again, then asks the separately named verifier model whether that claim is supported, contradicted, or uncertain. It saves the answer and evidence record in the audit log. The interface shows the verdict, the draft's citations, the verifier's evidence, and verifier notes.

The `.env.example` shows Gemini's native API base with separate answer and verifier model IDs. Gemini API free-tier quotas and model availability change; configure a key and check the current provider limits. Evidence submitted to Gemini leaves the computer, and free-tier terms may permit use of requests to improve the provider's products. The key stays on the API server in `.env`, never in the web page. If there is no model configuration, upload and retrieval still work, but answer generation returns a setup error. Independent model calls can still be wrong, so `supported` is a model verdict based on supplied passages—not proof.

The browser's answer view escapes text before displaying it, labels empty/uncertain verdicts, and includes an audit ID so a user can open the saved API audit record. It does not automatically rewrite unsupported claims: correction and dependency checks need a stronger evaluated design before they should alter scientific statements.

### Tesseract OCR's role

`src/agent/ingestion/pipeline.py` uses Tesseract for standalone image uploads, scanned PDF pages, and embedded PDF image blocks when PyMuPDF exposes them. The API and background ingestion worker both pass the optional `TESSERACT_CMD` path into the parser. OCR contributes recognized words to ordinary chunking and retrieval. Tesseract does not infer plotted values, axis relationships, figure meaning, or the content of a mathematical diagram. OCR settings are `OCR_ENABLED`, `OCR_LANGUAGE`, and optional `TESSERACT_CMD`. A response status of `unavailable` means the executable could not be resolved; `completed` means it returned searchable text.

### Full flow, from upload to an answer

```text
PDF/text/image
  -> parser (PyMuPDF, DOCX/HTML/text reader, optional Tesseract)
  -> chunks with page and character offsets
  -> SQLite/Postgres catalog + BM25 index + optional AstroBERT vectors
  -> optional Neo4j entity/evidence graph
  -> router chooses graph+hybrid, vector+BM25 hybrid, or BM25
  -> evidence passages with file/page/chunk provenance
  -> answer model creates cited atomic claims
  -> second retrieval pass gathers evidence for each claim
  -> separate verifier assigns supported/contradicted/uncertain/unsupported
  -> claim verdicts, evidence links, model names, and audit ID in the UI
```

## 14. What is still needed for the full architecture

To move closer to the whole architecture, the remaining work includes:

1. Add versioned database migrations, session deletion/retention endpoints, and transaction cleanup for failed multi-file uploads.
2. Add a vector index once the embedding model's vector size is fixed, and measure retrieval accuracy against a representative scientific-question set.
3. Add stronger OCR quality checks and figure/table/chart extraction with page coordinates.
4. Expand scholarly connectors beyond arXiv abstracts, improve astronomy identifier resolution, and independently assess source quality.
5. Add bounded surgical patching and cascade/dependency checks for claims that fail verification.
6. Add API authentication, per-user authorization, quotas, observability, and production deployment safeguards.
7. Add migrations and a model-version field to embeddings before switching embedding models in an already populated deployment; until then, re-ingest documents when the chosen embedding model changes.

External embeddings, chat-model calls, and live arXiv requests were not exercised by tests because they require provider credentials or external network access. Those features use mocked tests for their application behavior. Tesseract OCR logic was implemented, but this computer does not currently have the Tesseract executable, so real OCR quality still needs a live sample-file check.

The end-to-end path that was exercised locally uses Postgres/pgvector, Garage S3-compatible storage, Redis Streams, a queued text upload, and subsequent retrieval. The complete generation path has unit coverage for citation filtering and audit logging, but still needs live evaluation with the chosen models. The graph connector, local AstroBERT model download/encoding, Gemini calls, and real Tesseract OCR require their optional services/programs and credentials, so offline automated tests mock or skip those external capabilities.
