# Jury guide: document ingestion, retrieval, and astronomy embedding benchmark

This guide is written so a presenter can explain the project without assuming the jury already knows search engines, embeddings, or machine learning. It separates the **application pipeline** from the **offline benchmark**. AstroBERT was evaluated in the benchmark; it has not yet been connected to application ingestion or retrieval.

## 1. Short presentation summary

> The system takes scientific documents, extracts their text, divides that text into traceable passages, and indexes those passages so they can be searched later. The current application uses keyword search with BM25, and can optionally add vectors from a configured embedding service. To decide whether an astronomy-focused free model would improve search, I separately benchmarked AstroBERT Small, a general embedding model called MiniLM, and BM25 on a public astronomy paper retrieval dataset. AstroBERT gave the best dense retrieval result, while combining AstroBERT with BM25 gave the best top-one and nDCG scores. The benchmark supports integrating local AstroBERT as a next step, but it is not yet part of the running application, and the public benchmark is not a substitute for evaluation on our own documents.

## 2. The architecture in one picture

```text
APPLICATION: user-provided document
        |
        v
  Upload API and session
        |
        v
  Save original file and metadata
        |
        v
  Parse text (OCR for scans/images when configured)
        |
        v
  Split text into chunks and keep page/offset metadata
        |
        +------------------------------+
        |                              |
        v                              v
  Keyword full-text index       Optional embedding vector
  (SQLite FTS5 / BM25)           (currently external endpoint)
        |                              |
        +---------------+--------------+
                        |
                        v
             User asks a search question
                        |
                        v
       Plan query -> retrieve candidates -> fuse rankings
                        |
                        v
       Rerank -> return passages with provenance

OFFLINE BENCHMARK (separate data and process)
  Public astronomy abstracts + queries + relevance labels
         -> BM25 / embedding / hybrid ranking
         -> compare rankings to labels using retrieval metrics
```

## 3. Explain the application flow step by step

### Step 1 — Input and session

The user uploads a document to a session. A session is an identifier that groups one user's documents and searches. Retrieval uses the session identifier so one session does not accidentally search another session's material. Supported inputs include common text-bearing documents such as PDF, DOCX, HTML, text, Markdown, and images.

**Why keep a session?** It scopes the evidence that can be retrieved and makes it possible to trace a result back to a particular upload.

### Step 2 — Preserve the original and its metadata

The application saves the source file and records a document entry in its catalog. In the local configuration, files and the catalog are kept locally. Other configured deployments can use object storage and PostgreSQL. For larger jobs, the app can queue ingestion for a worker instead of doing all parsing inside the upload request.

**Why keep the original?** Extracted text is useful for search, but the original file is the authoritative source. Keeping it allows a person to check the returned passage in its original context.

### Step 3 — Parse or extract text

The parser reads the document format and extracts text. PDF extraction can retain page numbers. Table content may be represented with separators between cells. For scans or images, OCR can recognize visible characters if the Tesseract program and the required language data are installed and OCR is enabled.

OCR means optical character recognition: software that converts visible letters in an image into machine-readable text. It reads words; it does not automatically understand a chart's geometry, recover a plotted trend, or establish the scientific meaning of a figure.

**Why parse into text?** Search systems need searchable content. Retaining page and source metadata lets the system show where a result came from instead of returning anonymous text.

### Step 4 — Chunk the extracted text

Long extracted text is split into smaller passages called **chunks**. The project uses a configurable approximate chunk size and overlap. Overlap repeats a small amount of text at the boundary between adjacent chunks.

For example, if a sentence is split at a chunk boundary, overlap helps avoid losing its meaning because the first half ended one chunk and the second half started the next. Chunking also makes it possible to retrieve a focused passage rather than an entire book or paper.

Each chunk carries provenance metadata such as its document, session, page, order, character offsets, and coarse kind (for example, paragraph, table, or equation). Provenance means information about where the evidence came from.

**Trade-off:** chunks that are too small can lose context; chunks that are too large can make search less precise and return more irrelevant text. The right size should be evaluated on the target documents and questions.

### Step 5 — Index each chunk for keyword retrieval

The current app creates a full-text search index. In SQLite mode, the project uses FTS5, SQLite's full-text search feature. FTS5 uses BM25 to rank matching passages.

BM25 is a lexical method. It mainly matches words and phrases rather than inferring that different words have related meanings. It adjusts for how often a term occurs in a passage and how common that term is across the collection. A rare query term can be more informative than a very common word.

**Why keep BM25?** Astronomy contains exact object names, instrument names, catalog identifiers, units, symbols, and specialized terms. Exact lexical matching can retrieve these even when semantic similarity is uncertain. BM25 is also local and does not require an embedding model or API.

### Step 6 — Optional vector creation in the current app

The application currently supports embeddings only when `EMBEDDING_API_BASE` is configured. With that setting, it sends chunk text to an OpenAI-compatible embedding endpoint during ingestion and sends query text to the same endpoint during retrieval. The endpoint converts text into a list of numbers called a **vector**.

The application stores those vectors with chunks. The database compares a query vector with passage vectors to find text that is semantically similar, even if it uses different words. The app currently does not select or run AstroBERT locally by itself.

**Privacy and cost:** whether this external mode is free or paid depends on the endpoint provider. The text leaves the computer and is sent to the configured service. Local AstroBERT could avoid per-request API charges and keep embedding work local, but integrating it is a separate code change and requires installing/downloading the model.

### Step 7 — Plan the user's search

At search time, the app's deterministic planner keeps the original query and may generate up to a few focused query clauses. These are simple query transformations intended to improve coverage, not a second language model and not a source of new scientific facts.

**Why plan a query?** A question can contain several concepts. Searching focused clauses can find passages that cover different parts of it, while preserving the original wording for direct matching.

### Step 8 — Retrieve candidate passages

For each planned query, the app asks the catalog for candidate passages. With no vector available, it uses keyword retrieval. If embedding vectors are available, it also runs semantic retrieval and combines lexical and semantic rankings using **reciprocal-rank fusion (RRF)**.

RRF combines the *positions* of results rather than directly adding raw BM25 scores to cosine similarity scores. This is useful because the two methods produce different score scales. A passage ranked highly by both methods gets a strong combined position; a passage found by just one method can still be included.

The app combines duplicate passage candidates from different planned queries and keeps the better-ranked occurrence.

### Step 9 — Rerank and return traceable evidence

The current app applies a lightweight final reranker. It considers query-term coverage, an exact phrase match, and a small preference based on the supplied source tier. This is a heuristic: its score is not a probability that the passage is true. The source tier is supplied by the caller and is not independently verified by the application.

The API returns ranked passages and their metadata so a person or later answer stage can inspect the evidence and where it came from. Search results are briefly cached per session to avoid repeating the same search immediately.

**Important:** retrieval finds candidate evidence; it does not prove a scientific claim. The separate answer and verification stages are downstream components and were not evaluated by the embedding benchmark.

## 4. Explain the benchmark as a separate experiment

### What data was used?

The benchmark uses a public astronomy similarity/retrieval dataset. It includes a corpus of astronomy paper abstracts, short queries, and relevance labels identifying which paper is relevant to each query. The files and model revisions are pinned in the benchmark script so that a future run uses the same versions unless the script is deliberately changed.

The benchmark **does not** read the user's uploaded documents, the app's database, or private project files. The benchmark is repeatable and separate from the user-facing app pipeline.

### What models and methods were compared?

1. **BM25 alone:** local lexical search over the benchmark abstracts. This establishes a baseline.
2. **AstroBERT Small alone:** an astronomy-focused embedding model. The script encodes abstracts and queries, normalizes vectors, and ranks abstracts by vector dot product. Since the vectors are normalized, this ordering is equivalent to cosine similarity.
3. **MiniLM alone:** a general-purpose embedding model, used as a comparison to test whether a domain-specific model helps.
4. **BM25 + AstroBERT RRF:** combines lexical and semantic ranking lists.
5. **BM25 + model plus project-style reranker:** tests whether the simple project reranking rules improve the ranking on this dataset.

AstroBERT produced 384-dimensional vectors in this benchmark. That means each abstract/query is represented as 384 numbers. These numbers do not correspond one-to-one with human-readable concepts; the model learns a representation where related text tends to be nearby in vector space.

SPECTER2 was considered, and its query/document adapters were smoke-tested for valid output, but it was not included in the scored table. It is designed with distinct query and scientific-document adapters and commonly uses title-plus-abstract input. The selected dataset provides abstracts without a separate title field. A full score would therefore require a clearly defined and fair input setup. The SPECTER2 repository documents this adapter distinction: https://github.com/allenai/SPECTER2.

### How are rankings measured?

All metrics compare the ranked result list against the benchmark's relevance labels. Higher is better.

| Metric | Beginner explanation | Example interpretation |
| --- | --- | --- |
| Recall@1 | Fraction of queries with a relevant paper at rank 1. | 0.60 means the correct paper is first for about 60% of queries. |
| Recall@5 | Fraction of queries with a relevant paper somewhere in the first five. | Measures whether a user can find it after scanning a short list. |
| Recall@10 | Fraction of queries with a relevant paper somewhere in the first ten. | Useful for judging retrieval coverage at a modest candidate list size. |
| MRR@10 | Average reciprocal rank of the first relevant result within ten; a rank 1 result contributes 1, rank 2 contributes 1/2, rank 5 contributes 1/5. | Rewards relevant results that appear earlier. |
| nDCG@10 | A rank-sensitive score that rewards relevant documents higher in the list and discounts lower positions. | Useful for comparing the overall ordering near the top. |

The dataset has one labeled relevant paper for each benchmark query in this run. Therefore, recall is easy to interpret as the fraction of queries for which that known paper appears in the first *k* results.

### Measured results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5290 | 0.7390 | 0.8165 | 0.6205 | 0.6676 |
| AstroBERT dense | 0.6060 | 0.8355 | 0.8925 | 0.7016 | 0.7478 |
| MiniLM dense | 0.3570 | 0.5790 | 0.6740 | 0.4545 | 0.5069 |
| BM25 + AstroBERT RRF | 0.6135 | 0.8215 | 0.8905 | 0.7035 | 0.7485 |
| BM25 + AstroBERT + project-style reranker | 0.6115 | 0.8075 | 0.8715 | 0.6958 | 0.7381 |
| BM25 + MiniLM RRF | 0.4835 | 0.7315 | 0.8210 | 0.5916 | 0.6467 |
| BM25 + MiniLM + project-style reranker | 0.5535 | 0.7825 | 0.8520 | 0.6523 | 0.7005 |

### What do the results mean?

- AstroBERT's Recall@10 is 0.8925, versus 0.8165 for BM25. The absolute increase is 0.076, or **7.6 percentage points**. Across 2,000 queries, that is 152 additional queries where the labeled paper was retrieved in the top ten.
- AstroBERT alone has the highest Recall@5 and Recall@10 in this table. It is strongest when the priority is to include the relevant paper in a short candidate list.
- BM25 + AstroBERT RRF has the highest Recall@1 (0.6135) and nDCG@10 (0.7485). It combines exact-word evidence with semantic similarity and is a reasonable hybrid design to test in the app.
- Its Recall@10 (0.8905) is 0.002 lower than AstroBERT alone (0.8925): a difference of just 0.2 percentage points in this experiment.
- The project-style reranker reduced the AstroBERT hybrid Recall@10 from 0.8905 to 0.8715. This indicates that the current heuristic may not be well calibrated for this benchmark. It does not prove that all reranking is harmful.
- MiniLM alone underperformed BM25 and AstroBERT on this astronomy benchmark. This supports the value of selecting a model for the target domain rather than assuming any embedding model will improve retrieval.

### Time and cost

In the uncached CPU run on the benchmark machine, AstroBERT encoded the corpus and queries in about 875.1 seconds; MiniLM took about 740.9 seconds. These are bulk encoding times for the benchmark, not search latency per query. The script caches vectors; the later rerun reused those vectors to recompute ranking metrics.

The benchmark uses publicly downloadable model and data files, and local inference, so the measured embedding calls did not incur hosted API charges. A production local deployment still needs disk space for downloaded model files, RAM/CPU for inference, and time to embed uploaded chunks. The model card and dataset card list Apache 2.0 licensing: https://huggingface.co/NeuML/astrobert-small-embeddings and https://huggingface.co/datasets/NeuML/arxiv-astronomy-similarity. The organization should still review whether the license and usage conditions fit its deployment.

## 5. Limitations to tell the jury honestly

1. **Paper-level, not passage-level:** the benchmark ranks abstracts. The app needs to retrieve supporting passages inside user documents. That must be tested separately.
2. **One known relevant paper per query:** real research questions may have many relevant sources, controversial evidence, or no answer in the corpus.
3. **Different input length and style:** benchmark queries and abstracts may not match long, multi-part questions or chunks from scanned PDFs.
4. **No answer verification evaluation:** retrieval metrics do not establish whether a generated answer is correct, adequately cited, or scientifically safe.
5. **No actual AstroBERT app integration:** the benchmark is a model-selection experiment. It did not modify ingestion, settings, database schema, or query embedding in the application.
6. **Reranker proxy:** the benchmark approximates the app's lightweight reranker on abstract text and equal source tiers. Application behavior can differ with passages, metadata, and trust tier values.
7. **Hardware-dependent speed:** the encoding times describe one local CPU run and should not be treated as a universal speed guarantee.

## 6. What the benchmark files do

- `benchmarks/compare_astronomy_embeddings.py` downloads the pinned public dataset if needed; computes BM25 and selected embedding rankings; optionally caches vectors; evaluates the rankings; and writes CSV metrics. The benchmark package dependencies were installed in a temporary environment for the run and were not added to the app's project dependencies.
- `benchmarks/results.csv` stores the measured results in a format that spreadsheet tools can open.
- `benchmarks/ASTRONOMY_EMBEDDINGS_REPORT.md` gives a concise comparison and recommendations.
- `benchmarks/JURY_EXPLANATION.md` is this longer oral-explanation guide.

To reproduce the two directly comparable embedding runs from the repository root, install `sentence-transformers` and its dependencies in an isolated Python environment, then run:

```powershell
python benchmarks/compare_astronomy_embeddings.py `
  --data-dir .cache/astro-embedding-benchmark `
  --output benchmarks/results.csv `
  --models astrobert,minilm
```

On the first run it downloads data/models and performs slow bulk encoding. Later runs can load the cached vectors. SPECTER2 can be selected by passing `--models specter2`, but a fair comparison should first define title-plus-abstract document input and query text formatting.

## 7. Likely jury questions and suggested answers

### “Why not just use keyword search?”

Keyword search is a strong, transparent baseline and it is useful for exact terms, identifiers, units, and names. It can miss relevant text when a question uses different wording from the paper. Astronomy-tuned embeddings add a way to match related wording. The benchmark shows a gain for AstroBERT on its astronomy paper-retrieval task, while BM25 remains valuable in a hybrid system.

### “What does an embedding actually represent?”

An embedding is a fixed-length list of numbers produced from text. It is not a summary or an explanation. The model is trained so text with related meaning tends to have vectors that are close under a similarity measure. Retrieval ranks the stored passage vectors by how close they are to the query vector.

### “Why compare a domain-specific model with MiniLM?”

MiniLM provides a general-purpose reference point. Comparing it with an astronomy-focused model tests whether domain adaptation matters for the project's scientific field. In this test it did: AstroBERT substantially outperformed MiniLM on the astronomy dataset.

### “Why combine BM25 and AstroBERT?”

They have complementary strengths. BM25 is good at literal word overlap; AstroBERT can connect related wording. RRF merges the two ranked lists by rank rather than mixing incompatible raw score scales. The measured hybrid ranked the correct item first more often than either BM25 alone or AstroBERT alone in this dataset.

### “Is 89.25% the accuracy of the scientific agent?”

No. It is Recall@10 on a paper retrieval benchmark: the fraction of benchmark queries for which the one labeled relevant abstract appeared within the first ten results. It is not answer accuracy, fact-checking accuracy, or a general guarantee about user documents.

### “Why was the reranker worse?”

The current reranker is a simple heuristic based on query term coverage and phrase matching. Abstracts can repeat query words without being the best semantic match, so the heuristic can move a strong semantic result downward. The result indicates that the reranker should be tuned and evaluated on representative data, not that ranking rules should never be used.

### “Is AstroBERT free?”

The selected model and benchmark dataset are publicly available under Apache 2.0 according to their cards, and this experiment ran the model locally without a hosted embedding API. There is no per-request embedding API charge in that setup. There are still download, storage, compute, and maintenance costs, and the app must be changed before it can use AstroBERT locally.

### “What would you implement next?”

Add a selectable local AstroBERT backend used consistently for both document chunks during ingestion and search queries during retrieval. Continue combining semantic rankings with BM25. Record which model generated stored vectors so vectors from different models are not compared accidentally, and require re-embedding when the model changes. Then evaluate with labeled astronomy passages from the actual documents and questions the project targets, including equations, tables, identifiers, and units.

## 8. Suggested closing statement

> The benchmark gives evidence that an astronomy-specific embedding model can improve literature retrieval over keyword search on a public astronomy dataset. I recommend a local AstroBERT plus BM25 hybrid because the methods complement each other and the hybrid performed well near the top of the ranked list. This is a measured direction, not yet a completed integration or proof of answer correctness. The next validation is to run the integrated pipeline against labeled passages from representative project documents.
