# Astronomy embedding comparison: results and explanation

**Purpose:** decide which free embedding approach is the best fit for this astronomy-focused evidence retrieval project. This is a report of an offline benchmark, not a claim that the application has already switched embedding providers.

## In one minute

I compared the project's keyword-search baseline (SQLite FTS5/BM25) with two embedding models on a public astronomy paper-retrieval benchmark. The benchmark contains 10,000 astronomy abstracts, 2,000 short queries, and one labeled relevant paper per query. The script and the exact measured metrics are included alongside this report.

The astronomy-specific **AstroBERT Small** had the strongest results. Dense AstroBERT retrieval found the relevant paper in the top ten for **89.25%** of queries, compared with **81.65%** for BM25 and **67.40%** for general-purpose MiniLM. Combining AstroBERT with BM25 using reciprocal-rank fusion (RRF) gave the best top-one recall and the best nDCG in this run, while top-ten recall was almost the same as AstroBERT alone.

**Recommendation:** use AstroBERT Small locally, with BM25 + AstroBERT RRF as the starting retrieval design. It requires no per-request API payment and can run on CPU. Keep the model choice configurable and evaluate with actual questions and passages from this project before treating it as production-proven. The project application code has not been modified as part of this comparison.

## What I compared

| Method | What it does | Role in the comparison |
| --- | --- | --- |
| SQLite FTS5 with BM25 | Looks for matching words and ranks documents using word frequency and rarity. | Baseline and the same family of local keyword search already used by this project. |
| AstroBERT Small dense retrieval | Turns each abstract and query into a 384-number vector; ranks by cosine similarity. It is trained for astronomy text. | Domain-specific semantic search candidate. |
| MiniLM dense retrieval | Uses the same vector approach with a general-purpose sentence model. | Free, lightweight general model control; tests whether astronomy specialization helps. |
| BM25 + AstroBERT RRF | Combines the two ranked lists using reciprocal-rank fusion, so a document can score well from either exact words or semantic similarity. | Proposed hybrid design. |
| BM25 + AstroBERT + project reranker | Applies a simplified version of the current project's final query-term coverage and exact-phrase reranking to the hybrid results. | Checks whether the current final reranking heuristic helps on this benchmark. |
| BM25 + MiniLM RRF and its reranked variant | Same hybrid and reranking comparisons using MiniLM instead. | Shows whether combining a weaker generic embedding model with BM25 helps. |

SPECTER2 is also a relevant scientific-paper retrieval model, but it uses separate adapters for query and paper text and is normally designed around title-plus-abstract inputs. The selected benchmark exposes abstracts but not a separate paper-title field. I confirmed the adapters load and produce finite vectors, but did not include SPECTER2 in the comparable scored table; this avoids presenting a mismatched input setup as a fair head-to-head result. Its project/model documentation describes the query and paper adapter distinction. [SPECTER2 repository](https://github.com/allenai/SPECTER2)

## Results

All values below come from the same 2,000 queries and the same labeled relevant-paper set. Higher is better. Recall values are the fraction of queries whose relevant paper appears within the first *k* results. MRR rewards placing the relevant paper near the top. nDCG@10 also rewards high placement, with a logarithmic discount for lower ranks.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 (project-style lexical baseline) | 0.5290 | 0.7390 | 0.8165 | 0.6205 | 0.6676 |
| **AstroBERT dense** | **0.6060** | **0.8355** | **0.8925** | 0.7016 | 0.7478 |
| MiniLM dense | 0.3570 | 0.5790 | 0.6740 | 0.4545 | 0.5069 |
| BM25 + AstroBERT (RRF) | **0.6135** | 0.8215 | 0.8905 | **0.7035** | **0.7485** |
| BM25 + AstroBERT + project reranker | 0.6115 | 0.8075 | 0.8715 | 0.6958 | 0.7381 |
| BM25 + MiniLM (RRF) | 0.4835 | 0.7315 | 0.8210 | 0.5916 | 0.6467 |
| BM25 + MiniLM + project reranker | 0.5535 | 0.7825 | 0.8520 | 0.6523 | 0.7005 |

### How to read the differences

- AstroBERT improves Recall@10 by **7.60 percentage points** over BM25 alone (89.25% versus 81.65%). In this benchmark, that means it found the labeled paper in the first ten results for 152 more of the 2,000 queries.
- AstroBERT's MRR@10 improves from 0.6205 to 0.7016. In practical terms, the correct paper tends to appear earlier in the list.
- Hybrid RRF is the strongest balanced starting point here: it has the best Recall@1 (61.35%) and nDCG@10 (0.7485). Pure AstroBERT has slightly higher Recall@5 and Recall@10. The difference in Recall@10 is just 0.20 percentage points (89.05% hybrid vs. 89.25% dense).
- The project's simple term-coverage reranker **reduced** AstroBERT hybrid Recall@10 from 89.05% to 87.15% (a 1.90 percentage-point drop), and also reduced MRR and nDCG. That is a reason to tune or gate that heuristic when AstroBERT is used, not evidence that reranking is always harmful.
- MiniLM performed worse than both BM25 and AstroBERT as a dense retriever on this astronomy benchmark. Hybridizing it with BM25 recovered some quality, but did not catch up to AstroBERT.

## Free means free of API charges, with local resource costs

AstroBERT Small and the benchmark dataset are published under Apache 2.0 on their Hugging Face cards ([AstroBERT Small model card](https://huggingface.co/NeuML/astrobert-small-embeddings), [astronomy dataset card](https://huggingface.co/datasets/NeuML/arxiv-astronomy-similarity)). They can be downloaded and run locally, so embedding a query or document does not require a paid hosted embedding API. The benchmark script explicitly runs both embedding models on CPU and contains no hosted embedding API client.

There are still practical costs: the model needs to be downloaded once, uses local disk and memory, and CPU embedding takes time. In the uncached run on this machine, encoding 10,000 abstracts and 2,000 queries took **875.1 seconds for AstroBERT** and **740.9 seconds for MiniLM**. That one-time corpus encoding is not the latency of one query; the application only needs to encode each new chunk at ingestion and each search query at retrieval. Re-embedding already stored passages is required if the model changes.

## Limits: what this result does and does not establish

This measures **finding a paper by its abstract from a short query**. It does not yet measure retrieval of the exact supporting passage in an uploaded paper, scanned document, table, equation, or figure. It also does not measure answer correctness or scientific verification. There is one labeled relevant abstract per benchmark query, which is a narrower judgment than real research search. So these are useful comparative measurements for model choice, not a guarantee of production quality.

Before making a final production decision, build a small held-out evaluation set from real astronomy questions and the actual documents the agent should search. Label which passages support each question, then rerun these same ranking metrics. In particular, evaluate technical identifiers, units, equations, object names, and questions that require evidence from a specific uploaded paper.

## What changed in the repository

| File | Change | Why |
| --- | --- | --- |
| `benchmarks/compare_astronomy_embeddings.py` | Added a repeatable benchmark runner. It pins dataset/model revisions, retrieves with FTS5 BM25 and cosine similarity, fuses rankings with RRF, mirrors the current project's lightweight reranker, computes standard retrieval metrics, and caches generated vectors locally. | Makes the model comparison reproducible and prevents silently comparing different versions of the data/models. |
| `benchmarks/results.csv` | Added the measured numeric results. | Lets the comparison be loaded into a spreadsheet or processed by another script. Model encoding times are the measured first-run times (AstroBERT 875.1 s; MiniLM 740.9 s); derived ranking rows share the respective model time. BM25's recorded run was 10.04 s. |
| `benchmarks/ASTRONOMY_EMBEDDINGS_REPORT.md` | Added this explanation of methods, metrics, findings, cost, and limits. | Gives a beginner-friendly interpretation instead of leaving the reader with raw scores. |

No ingestion, retrieval, model-selection, database, dependency, or `.env` setting in the running application was changed by this benchmark work. The existing application still enables semantic embeddings only when `EMBEDDING_API_BASE` is configured; that sends document chunks and queries to the selected external endpoint. The comparison's recommendation is therefore **not yet wired into the app**. This is important because an offline win does not on its own make a model available to the application's ingestion and retrieval code.

## Reproducing the comparison

The benchmark requires Python plus `sentence-transformers` and its ML dependencies. It downloads the fixed public benchmark and model files on first use. It does not read the application's upload folder or database. From the repository root, after installing those packages into an isolated environment:

```powershell
python benchmarks/compare_astronomy_embeddings.py `
  --data-dir .cache/astro-embedding-benchmark `
  --output benchmarks/results.csv `
  --models astrobert,minilm
```

The first run is slow because it encodes the corpus. Later runs can reuse the `.npz` vector caches and quickly recompute rankings. The dataset and model revisions are pinned in the script so a later upstream update does not silently change the experiment.

## Recommendation for the next implementation step

If you want the application itself to use this free result, the next code change should add an optional local embedding backend (AstroBERT Small), call the same model during document ingestion and search-query embedding, preserve BM25 as the lexical half of hybrid RRF, and keep the external API backend opt-in. It should also identify which model produced stored vectors and require re-ingestion when that model changes. The reranker should be configurable because this benchmark shows the current heuristic can lower AstroBERT hybrid ranking quality.
