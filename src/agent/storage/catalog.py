"""Small persistent catalog for ingested inputs and searchable text chunks."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class Catalog:
    def __init__(self, database: Path):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL, sha256 TEXT NOT NULL, storage_key TEXT NOT NULL,
                    parser TEXT NOT NULL, page_count INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event TEXT NOT NULL,
                    details TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    session_id TEXT NOT NULL, page INTEGER NOT NULL, chunk_order INTEGER NOT NULL,
                    text TEXT NOT NULL, embedding TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
                    text, content='chunks', content_rowid='id', tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunk_search(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunk_search(chunk_search, rowid, text)
                    VALUES ('delete', old.id, old.text);
                END;
                """
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_order_idx "
                "ON chunks(document_id, chunk_order)"
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")

    def add_prompt(self, session_id: UUID, text: str, created_at: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO prompts VALUES (?, ?, ?, ?)",
                       (str(uuid4()), str(session_id), text, created_at))

    def add_audit(
        self, session_id: UUID, event: str, details: dict[str, Any], created_at: str
    ) -> UUID:
        audit_id = uuid4()
        with self.connect() as db:
            db.execute(
                "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)",
                (str(audit_id), str(session_id), event, json.dumps(details), created_at),
            )
        return audit_id

    def get_audit(self, session_id: UUID, audit_id: UUID) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT event, details, created_at FROM audit_log WHERE id = ? AND session_id = ?",
                (str(audit_id), str(session_id)),
            ).fetchone()
        if row is None:
            return None
        return {
            "event": row["event"],
            "details": json.loads(row["details"]),
            "created_at": row["created_at"],
        }

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def add_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    document[key]
                    for key in (
                        "id", "session_id", "filename", "mime_type", "sha256",
                        "storage_key", "parser", "page_count", "created_at",
                    )
                ),
            )
            db.executemany(
                """INSERT OR IGNORE INTO chunks
                   (document_id, session_id, page, chunk_order, text, embedding)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        document["id"], document["session_id"], c["page"], c["order"],
                        c["text"], json.dumps(c["embedding"]) if c.get("embedding") else None,
                    )
                    for c in chunks
                ],
            )

    def search(
        self,
        session_id: UUID,
        query: str,
        limit: int,
        query_embedding: list[float] | None = None,
        semantic_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        # FTS5 query syntax is intentionally kept out of the user-controlled MATCH expression.
        terms = list(dict.fromkeys(w.lower() for w in re.findall(r"[\w]+", query)))
        if not terms and query_embedding is None:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self.connect() as db:
            lexical_rows = db.execute(
                """SELECT c.id, c.document_id, d.filename, d.sha256, d.storage_key, d.parser,
                          c.page, c.chunk_order, c.text, bm25(chunk_search) AS score
                   FROM chunk_search JOIN chunks c ON c.id = chunk_search.rowid
                   JOIN documents d ON d.id = c.document_id
                   WHERE chunk_search MATCH ? AND c.session_id = ?
                   ORDER BY score LIMIT ?""",
                (match, str(session_id), max(limit * 10, 50)),
            ).fetchall() if terms else []
            semantic_rows = db.execute(
                """SELECT c.id, c.document_id, d.filename, d.sha256, d.storage_key, d.parser,
                          c.page, c.chunk_order, c.text, c.embedding
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE c.session_id = ? AND c.embedding IS NOT NULL""",
                (str(session_id),),
            ).fetchall() if query_embedding is not None else []

        candidates: dict[int, dict[str, Any]] = {}
        lexical_rank: dict[int, int] = {}
        for rank, row in enumerate(lexical_rows, start=1):
            candidate = dict(row)
            candidate.pop("score", None)
            candidates[row["id"]] = candidate
            lexical_rank[row["id"]] = rank

        semantic_rank: dict[int, int] = {}
        if query_embedding is not None:
            similarities: list[tuple[float, sqlite3.Row]] = []
            for row in semantic_rows:
                vector = json.loads(row["embedding"])
                if len(vector) != len(query_embedding):
                    continue
                dot = sum(a * b for a, b in zip(vector, query_embedding, strict=True))
                norm_a = math.sqrt(sum(a * a for a in vector))
                norm_b = math.sqrt(sum(b * b for b in query_embedding))
                similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
                similarities.append((similarity, row))
            similarities.sort(key=lambda pair: pair[0], reverse=True)
            for rank, (_, row) in enumerate(similarities[:max(limit * 10, 50)], start=1):
                candidate = dict(row)
                candidate.pop("embedding", None)
                candidates[row["id"]] = candidate
                semantic_rank[row["id"]] = rank

        reciprocal_rank_constant = 60
        scored = []
        for candidate_id, candidate in candidates.items():
            score = 0.0
            if candidate_id in lexical_rank:
                score += (1 - semantic_weight) / (
                    reciprocal_rank_constant + lexical_rank[candidate_id]
                )
            if candidate_id in semantic_rank:
                score += semantic_weight / (reciprocal_rank_constant + semantic_rank[candidate_id])
            candidate["score"] = score
            scored.append(candidate)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


class PostgresCatalog:
    """Postgres/pgvector implementation of the same catalog interface."""

    def __init__(self, database_url: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use Postgres storage") from exc
        self._psycopg = psycopg
        self._dict_row = psycopg.rows.dict_row
        self.database_url = database_url
        with self.connect() as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            db.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY, session_id UUID NOT NULL, filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL, sha256 TEXT NOT NULL, storage_key TEXT NOT NULL,
                    parser TEXT NOT NULL, page_count INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS prompts (
                    id UUID PRIMARY KEY, session_id UUID NOT NULL,
                    text TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    id UUID PRIMARY KEY, session_id UUID NOT NULL, event TEXT NOT NULL,
                    details JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY, document_id UUID NOT NULL REFERENCES documents(id),
                    session_id UUID NOT NULL, page INTEGER NOT NULL, chunk_order INTEGER NOT NULL,
                    text TEXT NOT NULL, embedding vector,
                    search_vector TSVECTOR GENERATED ALWAYS AS
                        (to_tsvector('english', text)) STORED
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS chunks_session_idx ON chunks(session_id)"
            )
            db.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_order_idx
                   ON chunks(document_id, chunk_order)"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING GIN(search_vector)"
            )

    def connect(self) -> Any:
        return self._psycopg.connect(
            self.database_url,
            row_factory=self._dict_row,
            connect_timeout=5,
        )

    def add_prompt(self, session_id: UUID, text: str, created_at: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO prompts(id, session_id, text, created_at) VALUES (%s, %s, %s, %s)",
                (uuid4(), session_id, text, created_at),
            )

    def add_audit(
        self, session_id: UUID, event: str, details: dict[str, Any], created_at: str
    ) -> UUID:
        audit_id = uuid4()
        with self.connect() as db:
            db.execute(
                "INSERT INTO audit_log VALUES (%s, %s, %s, %s::jsonb, %s)",
                (audit_id, session_id, event, json.dumps(details), created_at),
            )
        return audit_id

    def get_audit(self, session_id: UUID, audit_id: UUID) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT event, details, created_at FROM audit_log
                   WHERE id = %s AND session_id = %s""",
                (audit_id, session_id),
            ).fetchone()
        return dict(row) if row else None

    def add_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO documents
                   (id, session_id, filename, mime_type, sha256, storage_key,
                    parser, page_count, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""",
                (
                    document["id"], document["session_id"], document["filename"],
                    document["mime_type"], document["sha256"], document["storage_key"],
                    document["parser"], document["page_count"], document["created_at"],
                ),
            )
            for chunk in chunks:
                vector = chunk.get("embedding")
                vector_text = f"[{','.join(map(str, vector))}]" if vector else None
                db.execute(
                    """INSERT INTO chunks
                       (document_id, session_id, page, chunk_order, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s::vector)
                       ON CONFLICT (document_id, chunk_order) DO NOTHING""",
                    (
                        document["id"], document["session_id"], chunk["page"],
                        chunk["order"], chunk["text"], vector_text,
                    ),
                )

    def search(
        self,
        session_id: UUID,
        query: str,
        limit: int,
        query_embedding: list[float] | None = None,
        semantic_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        lexical: list[dict[str, Any]] = []
        semantic: list[dict[str, Any]] = []
        with self.connect() as db:
            if re.search(r"[\w]+", query):
                rows = db.execute(
                    """SELECT c.id, c.document_id, d.filename, d.sha256, d.storage_key, d.parser,
                              c.page, c.chunk_order, c.text
                       FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE c.session_id = %s AND c.search_vector @@ plainto_tsquery('english', %s)
                       ORDER BY ts_rank_cd(c.search_vector, plainto_tsquery('english', %s)) DESC
                       LIMIT %s""",
                    (session_id, query, query, max(limit * 10, 50)),
                ).fetchall()
                lexical = [dict(row) for row in rows]
            if query_embedding is not None:
                vector_text = f"[{','.join(map(str, query_embedding))}]"
                rows = db.execute(
                    """SELECT c.id, c.document_id, d.filename, d.sha256, d.storage_key, d.parser,
                              c.page, c.chunk_order, c.text
                       FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE c.session_id = %s AND c.embedding IS NOT NULL
                       ORDER BY c.embedding <=> %s::vector
                       LIMIT %s""",
                    (session_id, vector_text, max(limit * 10, 50)),
                ).fetchall()
                semantic = [dict(row) for row in rows]
        return _merge_ranked(lexical, semantic, limit, semantic_weight)


def _merge_ranked(
    lexical: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    limit: int,
    semantic_weight: float,
) -> list[dict[str, Any]]:
    """Combine pre-ranked search lists without treating either score as confidence."""
    reciprocal_rank_constant = 60
    candidates: dict[Any, dict[str, Any]] = {}
    scores: dict[Any, float] = {}
    for weight, rows in ((1 - semantic_weight, lexical), (semantic_weight, semantic)):
        for rank, row in enumerate(rows, start=1):
            candidate_id = row["id"]
            candidate = candidates.setdefault(candidate_id, row)
            candidate.pop("embedding", None)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (
                reciprocal_rank_constant + rank
            )
    ranked = sorted(candidates.values(), key=lambda row: scores[row["id"]], reverse=True)
    for row in ranked:
        row["score"] = scores[row["id"]]
        row.pop("id", None)
    return ranked[:limit]


@lru_cache(maxsize=4)
def _sqlite_catalog(database: str) -> Catalog:
    return Catalog(Path(database))


@lru_cache(maxsize=4)
def _postgres_catalog(database_url: str) -> PostgresCatalog:
    return PostgresCatalog(database_url)


def get_catalog(settings: Any) -> Catalog | PostgresCatalog:
    if getattr(settings, "storage_backend", "sqlite") == "postgres":
        return _postgres_catalog(settings.database_url)
    return _sqlite_catalog(str(Path(settings.data_dir) / "catalog.sqlite3"))
