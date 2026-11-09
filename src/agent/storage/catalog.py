"""Small persistent catalog for ingested inputs and searchable text chunks."""

from __future__ import annotations

import re
import sqlite3
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
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    session_id TEXT NOT NULL, page INTEGER NOT NULL, chunk_order INTEGER NOT NULL,
                    text TEXT NOT NULL
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

    def add_prompt(self, session_id: UUID, text: str, created_at: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO prompts VALUES (?, ?, ?, ?)",
                       (str(uuid4()), str(session_id), text, created_at))

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def add_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    document[key]
                    for key in (
                        "id", "session_id", "filename", "mime_type", "sha256",
                        "storage_key", "parser", "page_count", "created_at",
                    )
                ),
            )
            db.executemany(
                """INSERT INTO chunks(document_id, session_id, page, chunk_order, text)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (document["id"], document["session_id"], c["page"], c["order"], c["text"])
                    for c in chunks
                ],
            )

    def search(self, session_id: UUID, query: str, limit: int) -> list[dict[str, Any]]:
        # FTS5 query syntax is intentionally kept out of the user-controlled MATCH expression.
        terms = list(dict.fromkeys(w.lower() for w in re.findall(r"[\w]+", query)))
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.id, c.document_id, d.filename, d.sha256, d.storage_key, d.parser,
                          c.page, c.chunk_order, c.text, bm25(chunk_search) AS score
                   FROM chunk_search JOIN chunks c ON c.id = chunk_search.rowid
                   JOIN documents d ON d.id = c.document_id
                   WHERE chunk_search MATCH ? AND c.session_id = ?
                   ORDER BY score LIMIT ?""",
                (match, str(session_id), limit),
            ).fetchall()
        return [dict(row) for row in rows]
