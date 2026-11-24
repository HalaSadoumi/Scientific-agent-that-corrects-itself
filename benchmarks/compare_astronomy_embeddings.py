"""Free, local-only comparison of astronomy-paper retrievers on a public benchmark.

This script never reads project uploads and never calls a hosted embedding API.
Run from the project root after installing the optional benchmark packages.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from pathlib import Path
from urllib.request import urlopen

DATASET = "NeuML/arxiv-astronomy-similarity"
DATA_REV = "6ebcb7647071b5d22bc4df5bbdb1a284c9ce5718"
FILES = {
    "corpus.jsonl": "beir/corpus.jsonl",
    "queries.jsonl": "beir/queries.jsonl",
    "qrels.tsv": "beir/qrels/test.tsv",
}
MODELS = {
    "astrobert": ("NeuML/astrobert-small-embeddings", "8da45d590d7a23cd1aa6bd0a92c45766c1743c00"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
}
SPECTER = {
    "base": "allenai/specter2_base",
    "base_rev": "3447645e1def9117997203454fa4495937bfbd83",
    "doc": "allenai/specter2",
    "doc_rev": "2081559630a80fc5851d8f798a05ba81e9468089",
    "query": "allenai/specter2_adhoc_query",
    "query_rev": "3f4448817028388648a74349ece07af4518ec5bd",
}
CANDIDATE_K = 400
RRF_K = 60


def fetch(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}"
    for name, remote in FILES.items():
        target = folder / name
        if not target.exists():
            print(f"Downloading public benchmark file: {remote}")
            with urlopen(f"{base}/{remote}", timeout=60) as response:
                target.write_bytes(response.read())


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return result


def keyword(corpus: list[dict], queries: list[dict]) -> dict[str, list[str]]:
    """SQLite FTS5 BM25, the same family as the project's local lexical search."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, text, tokenize='unicode61')")
    db.executemany("INSERT INTO docs VALUES (?,?)", [(str(x["_id"]), x.get("text", "")) for x in corpus])
    ranked = {}
    for query in queries:
        terms = list(dict.fromkeys(re.findall(r"[\w]+", query["text"].lower())))
        if not terms:
            ranked[str(query["_id"])] = []
            continue
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        rows = db.execute(
            "SELECT docid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (match, CANDIDATE_K),
        ).fetchall()
        ranked[str(query["_id"])] = [str(row[0]) for row in rows]
    db.close()
    return ranked


def dense(corpus: list[dict], queries: list[dict], qvec, dvec) -> dict[str, list[str]]:
    import numpy as np

    scores = np.asarray(qvec, dtype=np.float32) @ np.asarray(dvec, dtype=np.float32).T
    k = min(CANDIDATE_K, scores.shape[1])
    result = {}
    for i, query in enumerate(queries):
        row = scores[i]
        best = np.argpartition(row, -k)[-k:]
        best = best[np.argsort(row[best])[::-1]]
        result[str(query["_id"])] = [str(corpus[int(j)]["_id"]) for j in best]
    del scores
    return result


def sentence_transformer(model_id: str, revision: str, corpus: list[dict], queries: list[dict], batch: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    args = {"batch_size": batch, "show_progress_bar": True,
            "normalize_embeddings": True, "convert_to_numpy": True}
    dvec = model.encode([x.get("text", "") for x in corpus], **args)
    qvec = model.encode([x.get("text", "") for x in queries], **args)
    dims = int(dvec.shape[1])
    del model
    return qvec, dvec, dims


def specter2(corpus: list[dict], queries: list[dict], batch: int):
    """SPECTER2 has separate short-query and scientific-paper adapters."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SPECTER["base"], revision=SPECTER["base_rev"])
    model = AutoAdapterModel.from_pretrained(SPECTER["base"], revision=SPECTER["base_rev"])
    doc_adapter = snapshot_download(SPECTER["doc"], revision=SPECTER["doc_rev"])
    query_adapter = snapshot_download(SPECTER["query"], revision=SPECTER["query_rev"])
    model.load_adapter(doc_adapter, load_as="document", set_active=True)
    model.load_adapter(query_adapter, load_as="query", set_active=True)
    model.eval()

    def encode(texts: list[str], adapter: str):
        model.set_active_adapters(adapter)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                tokens = tokenizer(texts[start:start + batch], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt",
                                   return_token_type_ids=False)
                cls = model(**tokens).last_hidden_state[:, 0, :]
                batches.append(F.normalize(cls, p=2, dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    # Dataset has abstracts but no title field; standard SPECTER2 paper input is title + abstract.
    dvec = encode([x.get("text", "") for x in corpus], "document")
    qvec = encode([x.get("text", "") for x in queries], "query")
    dims = int(dvec.shape[1])
    del model
    return qvec, dvec, dims


def rrf(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {}
    for query_id in a.keys() | b.keys():
        score: dict[str, float] = {}
        for rows in (a.get(query_id, []), b.get(query_id, [])):
            for rank, doc_id in enumerate(rows, 1):
                score[doc_id] = score.get(doc_id, 0) + 0.5 / (RRF_K + rank)
        result[query_id] = [doc for doc, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]
    return result


def project_rerank(
    lexical: dict[str, list[str]],
    semantic: dict[str, list[str]],
    corpus: list[dict],
    queries: list[dict],
) -> dict[str, list[str]]:
    """Approximate the app's current RRF candidate pool and lightweight reranker."""
    corpus_text = {str(row["_id"]): str(row.get("text", "")).casefold() for row in corpus}
    reranked = {}
    for query in queries:
        query_id = str(query["_id"])
        text = str(query.get("text", ""))
        terms = {term.casefold() for term in re.findall(r"[\w]+", text) if len(term) > 1}
        scores: dict[str, float] = {}
        for rows in (lexical.get(query_id, []), semantic.get(query_id, [])):
            for rank, doc_id in enumerate(rows, 1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 0.5 / (RRF_K + rank)
        candidates = sorted(scores, key=scores.get, reverse=True)[:40]
        adjusted = []
        for doc_id in candidates:
            passage = corpus_text.get(doc_id, "")
            present = {term for term in terms if re.search(rf"\b{re.escape(term)}\b", passage)}
            coverage = len(present) / max(1, len(terms))
            phrase = 0.15 if text.casefold().strip(" ?.") in passage else 0.0
            adjusted.append((scores[doc_id] + 0.25 * coverage + phrase, doc_id))
        reranked[query_id] = [doc_id for _, doc_id in sorted(adjusted, reverse=True)[:10]]
    return reranked


def evaluate(rankings: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    import math

    recall = {k: 0.0 for k in (1, 5, 10)}
    mrr = ndcg = 0.0
    for query_id, relevant in qrels.items():
        results = rankings.get(query_id, [])
        for k in recall:
            recall[k] += len(set(results[:k]) & relevant) / max(1, len(relevant))
        for rank, doc_id in enumerate(results[:10], 1):
            if doc_id in relevant:
                mrr += 1 / rank
                ndcg += 1 / math.log2(rank + 1)
                break
    n = max(1, len(qrels))
    return {"queries": n, "recall_at_1": recall[1] / n, "recall_at_5": recall[5] / n,
            "recall_at_10": recall[10] / n, "mrr_at_10": mrr / n, "ndcg_at_10": ndcg / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/astro-embedding-benchmark"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", default="astrobert,minilm,specter2",
                        help="Any subset of astrobert,minilm,specter2")
    args = parser.parse_args()
    fetch(args.data_dir)
    corpus, queries = jsonl(args.data_dir / "corpus.jsonl"), jsonl(args.data_dir / "queries.jsonl")
    qrels = load_qrels(args.data_dir / "qrels.tsv")
    methods: dict[str, dict[str, list[str]]] = {}
    timings: dict[str, float] = {}
    start = time.perf_counter()
    methods["BM25 (SQLite FTS5)"] = keyword(corpus, queries)
    timings["BM25"] = time.perf_counter() - start
    selected = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    if selected - {"astrobert", "minilm", "specter2"}:
        parser.error("Unknown model: " + ", ".join(selected - {"astrobert", "minilm", "specter2"}))
    for name in ("astrobert", "minilm"):
        if name in selected:
            model_id, revision = MODELS[name]
            cache = args.data_dir / f"{name}-vectors.npz"
            if cache.exists():
                import numpy as np

                saved = np.load(cache)
                qvec, dvec = saved["queries"], saved["documents"]
                dims = int(dvec.shape[1])
                timings[name] = 0.0
                print(f"\nUsing saved {name} vectors ({dims} dimensions)")
            else:
                print(f"\nEncoding {model_id} locally on CPU")
                start = time.perf_counter()
                qvec, dvec, dims = sentence_transformer(
                    model_id, revision, corpus, queries, args.batch_size
                )
                timings[name] = time.perf_counter() - start
                import numpy as np

                np.savez_compressed(cache, queries=qvec, documents=dvec)
            methods[f"{name} dense"] = dense(corpus, queries, qvec, dvec)
            print(f"{dims} dimensions; encoding time {timings[name]:.1f} seconds")
            del qvec, dvec
    if "specter2" in selected:
        print("\nEncoding SPECTER2 query/document adapters locally on CPU")
        start = time.perf_counter()
        qvec, dvec, dims = specter2(corpus, queries, args.batch_size)
        methods["specter2 dense"] = dense(corpus, queries, qvec, dvec)
        timings["specter2"] = time.perf_counter() - start
        print(f"{dims} dimensions; {timings['specter2']:.1f} seconds")
        del qvec, dvec
    for name in ("astrobert", "minilm", "specter2"):
        dense_name = f"{name} dense"
        if dense_name in methods:
            lexical = methods["BM25 (SQLite FTS5)"]
            semantic = methods[dense_name]
            methods[f"BM25 + {name} (RRF)"] = rrf(lexical, semantic)
            methods[f"BM25 + {name} + project reranker"] = project_rerank(
                lexical, semantic, corpus, queries
            )
    rows = []
    for name, ranking in methods.items():
        row = {"method": name, **evaluate(ranking, qrels)}
        if name == "BM25 (SQLite FTS5)":
            timer = "BM25"
        elif name.startswith("BM25 + "):
            timer = name.split()[2].lower()
        else:
            timer = name.split()[0].lower()
        row["runtime_seconds"] = round(timings.get(timer, 0), 2)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "queries", "recall_at_1", "recall_at_5", "recall_at_10",
               "mrr_at_10", "ndcg_at_10", "runtime_seconds"]
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} results to {args.output}")
    for row in rows:
        print(f"{row['method']}: R@10={row['recall_at_10']:.4f}, "
              f"MRR@10={row['mrr_at_10']:.4f}, nDCG@10={row['ndcg_at_10']:.4f}")


if __name__ == "__main__":
    main()
