"""Optional Neo4j evidence graph with deterministic astronomy entity linking."""

from __future__ import annotations

import re
import ssl
from typing import Any


class GraphServiceError(RuntimeError):
    """Neo4j is configured but could not be used."""


# Keep recognition conservative: a false graph link is worse than a missed one.
_ENTITY_PATTERNS = [
    re.compile(r"\b(?:NGC|IC|HD|HIP|HR|PSR|TOI|Kepler|2MASS)\s*[A-Z]?-?\s*\d+(?:\.\d+)?\b", re.I),
    re.compile(r"\bGaia\s+(?:DR|EDR)\s*\d+\b", re.I),
    re.compile(r"\b(?:M|Messier)\s*\d{1,3}\b", re.I),
    re.compile(
        r"\b(?:Andromeda Galaxy|Milky Way|Orion Nebula|Crab Nebula|Betelgeuse|Sirius|Vega|"
        r"Polaris|Proxima Centauri|Sagittarius A\*)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:James Webb Space Telescope|Hubble Space Telescope|Chandra X-ray Observatory|"
        r"XMM-Newton|ALMA|JWST|Chandra|TESS|Kepler Space Telescope|Gaia|Fermi)\b",
        re.I,
    ),
]


def extract_entities(text: str) -> list[str]:
    """Find high-precision, recognizable astronomy names; this is not open IE."""
    found: dict[str, str] = {}
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            display = re.sub(r"\s+", " ", match.group(0)).strip()
            found.setdefault(display.casefold(), display)
    return list(found.values())


def _driver(settings: Any):
    uri = getattr(settings, "neo4j_uri", None)
    if not uri:
        return None
    if "YOUR_INSTANCE" in str(uri).upper() or "EXAMPLE.COM" in str(uri).upper():
        raise GraphServiceError(
            "NEO4J_URI still contains a template hostname. Copy the real URI from your "
            "Neo4j Aura instance connection details into .env."
        )
    if not getattr(settings, "neo4j_user", None) or not getattr(settings, "neo4j_password", None):
        raise GraphServiceError(
            "Set NEO4J_USER and NEO4J_PASSWORD in .env for the graph connection."
        )
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise GraphServiceError(
            "Neo4j support is optional; install it with `uv sync --extra graph`."
        ) from exc
    uri = str(settings.neo4j_uri)
    driver_options: dict[str, Any] = {}
    if uri.startswith(("neo4j+s://", "bolt+s://")):
        try:
            import truststore
        except ImportError as exc:
            raise GraphServiceError(
                "Install the graph extra with `uv sync --extra graph` to enable secure "
                "system certificate validation."
            ) from exc
        # Aura certificates can chain to a CA trusted by Windows/macOS but absent from
        # a bundled Python CA file. Truststore uses the OS store while still verifying
        # the server certificate; TLS checks remain enabled.
        uri = uri.replace("+s://", "://", 1)
        driver_options = {
            "encrypted": True,
            "ssl_context": truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        }
    try:
        return GraphDatabase.driver(
            uri,
            auth=(getattr(settings, "neo4j_user", None), getattr(settings, "neo4j_password", None)),
            **driver_options,
        )
    except Exception as exc:
        raise GraphServiceError("Could not connect to the configured Neo4j service") from exc


def graph_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "neo4j_uri", None))


def index_evidence(settings: Any, evidence: dict[str, Any]) -> None:
    """Store evidence nodes and entity co-occurrence edges, each tied to a source chunk."""
    index_evidence_batch(settings, [evidence])


def index_evidence_batch(settings: Any, evidence: list[dict[str, Any]]) -> None:
    """Index one document's evidence in one connection and transaction."""
    if not evidence:
        return
    driver = _driver(settings)
    if driver is None:
        return
    try:
        with driver.session(database=getattr(settings, "neo4j_database", "neo4j")) as session:
            session.run(
                "CREATE CONSTRAINT astro_entity_key IF NOT EXISTS "
                "FOR (n:AstroEntity) REQUIRE n.key IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT evidence_chunk_id IF NOT EXISTS "
                "FOR (n:EvidenceChunk) REQUIRE n.id IS UNIQUE"
            ).consume()
            session.execute_write(_write_evidence_batch, evidence)
    except Exception as exc:
        raise GraphServiceError("Could not index evidence in Neo4j") from exc
    finally:
        driver.close()


def _write_evidence_batch(tx: Any, evidence: list[dict[str, Any]]) -> None:
    for item in evidence:
        _write_evidence(tx, item, extract_entities(item["text"]))


def _write_evidence(tx: Any, evidence: dict[str, Any], entities: list[str]) -> None:
    tx.run(
        """MERGE (c:EvidenceChunk {id: $id})
           SET c.session_id=$session_id, c.document_id=$document_id, c.filename=$filename,
               c.page=$page, c.chunk_order=$chunk_order, c.text=$text,
               c.sha256=$sha256, c.parser=$parser, c.source_tier=$source_tier""",
        id=evidence["id"],
        session_id=evidence["session_id"],
        document_id=evidence["document_id"],
        filename=evidence["filename"],
        page=evidence["page"],
        chunk_order=evidence["chunk_order"],
        text=evidence["text"],
        sha256=evidence["sha256"],
        parser=evidence["parser"],
        source_tier=evidence["source_tier"],
    ).consume()
    if entities:
        tx.run(
            """MATCH (c:EvidenceChunk {id: $id})
               UNWIND $entities AS entity_name
           MERGE (e:AstroEntity {key: toLower(entity_name)})
           SET e.name=entity_name
           MERGE (e)-[:MENTIONED_IN]->(c)""",
            id=evidence["id"],
            entities=entities,
        ).consume()
    if len(entities) > 1:
        tx.run(
            """UNWIND $entities AS left_name
               UNWIND $entities AS right_name
               WITH left_name, right_name WHERE toLower(left_name) < toLower(right_name)
               MATCH (a:AstroEntity {key: toLower(left_name)})
               MATCH (b:AstroEntity {key: toLower(right_name)})
               MERGE (a)-[r:CO_OCCURS_WITH {evidence_id: $id}]->(b)
               SET r.session_id=$session_id""",
            entities=entities,
            id=evidence["id"],
            session_id=evidence["session_id"],
        ).consume()


def _schema(session: Any) -> tuple[set[str], set[str]]:
    """List existing graph types without querying nonexistent labels or relations."""
    labels = {row["label"] for row in session.run("CALL db.labels() YIELD label RETURN label")}
    relationships = {
        row["relationshipType"]
        for row in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
    }
    return labels, relationships


def search(settings: Any, session_id: str, query: str, limit: int) -> list[dict[str, Any]]:
    """Search exact entity names and evidence-linked co-occurrences in the current session."""
    driver = _driver(settings)
    if driver is None:
        return []
    terms = extract_entities(query)
    # For relational questions, also allow recognized key terms as exact entity names.
    if not terms:
        terms = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9*+-]{2,}", query)[:8]]
    try:
        with driver.session(database=getattr(settings, "neo4j_database", "neo4j")) as session:
            labels, relationship_types = _schema(session)
            if "AstroEntity" not in labels or "EvidenceChunk" not in labels:
                return []
            if "CO_OCCURS_WITH" not in relationship_types:
                return session.execute_read(_read_evidence_by_mentions, session_id, terms, limit)
            result = session.execute_read(_read_evidence, session_id, terms, limit)
            return [dict(row) for row in result]
    except Exception as exc:
        raise GraphServiceError("Could not search the configured Neo4j service") from exc
    finally:
        driver.close()


def snapshot(settings: Any, session_id: str, limit: int = 160) -> dict[str, Any]:
    """Return a small visualization snapshot for exactly one session."""
    driver = _driver(settings)
    if driver is None:
        return {
            "enabled": False,
            "entity_count": 0,
            "relationship_count": 0,
            "entities": [],
            "relationships": [],
        }
    try:
        with driver.session(database=getattr(settings, "neo4j_database", "neo4j")) as session:
            return session.execute_read(_read_snapshot, session_id, limit)
    except Exception as exc:
        raise GraphServiceError("Could not read the configured Neo4j graph") from exc
    finally:
        driver.close()


def _read_snapshot(tx: Any, session_id: str, limit: int) -> dict[str, Any]:
    labels, relationship_types = _schema(tx)
    if "AstroEntity" not in labels or "EvidenceChunk" not in labels:
        return {
            "enabled": True,
            "entity_count": 0,
            "relationship_count": 0,
            "entities": [],
            "relationships": [],
        }
    entities = [
        dict(row)
        for row in tx.run(
            """MATCH (e:AstroEntity)-[:MENTIONED_IN]->(c:EvidenceChunk)
               WHERE c.session_id=$session_id
               RETURN e.name AS id, e.name AS name, count(DISTINCT c) AS mentions
               ORDER BY mentions DESC, name LIMIT $limit""",
            session_id=session_id,
            limit=limit,
        )
    ]
    relationships = [
        dict(row)
        for row in tx.run(
            """MATCH (a:AstroEntity)-[r:CO_OCCURS_WITH]->(b:AstroEntity)
               MATCH (c:EvidenceChunk {id:r.evidence_id, session_id:$session_id})
               RETURN a.name AS source, b.name AS target, 'CO_OCCURS_WITH' AS type,
                      count(DISTINCT c) AS evidence_count
               ORDER BY evidence_count DESC LIMIT $limit""",
            session_id=session_id,
            limit=limit,
        )
    ] if "CO_OCCURS_WITH" in relationship_types else []
    return {
        "enabled": True,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entities": entities,
        "relationships": relationships,
    }


def _read_evidence(tx: Any, session_id: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
    result = tx.run(
        """MATCH (seed:AstroEntity)
           WHERE any(term IN $terms
             WHERE toLower(seed.name) CONTAINS toLower(term)
                OR toLower(term) CONTAINS toLower(seed.name))
           OPTIONAL MATCH (seed)-[r:CO_OCCURS_WITH]-(neighbor:AstroEntity)
           WHERE r.session_id=$session_id
           WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS graph_entities
           UNWIND graph_entities AS e
           MATCH (e)-[:MENTIONED_IN]->(c:EvidenceChunk)
           WHERE c.session_id=$session_id
           WITH c, count(DISTINCT e) AS matched_entities
           RETURN c.document_id AS document_id, c.filename AS filename, c.page AS page,
                  c.chunk_order AS chunk_order, c.text AS text, c.sha256 AS sha256,
                  c.parser AS parser, c.source_tier AS source_tier,
                  toFloat(matched_entities) AS score
           ORDER BY matched_entities DESC LIMIT $limit""",
        session_id=session_id,
        terms=terms,
        limit=limit,
    )
    return [dict(row) for row in result]


def _read_evidence_by_mentions(
    tx: Any, session_id: str, terms: list[str], limit: int
) -> list[dict[str, Any]]:
    """Search entity mentions when the graph has not formed any relation edges yet."""
    result = tx.run(
        """MATCH (seed:AstroEntity)
           WHERE any(term IN $terms
             WHERE toLower(seed.name) CONTAINS toLower(term)
                OR toLower(term) CONTAINS toLower(seed.name))
           MATCH (seed)-[:MENTIONED_IN]->(c:EvidenceChunk)
           WHERE c.session_id=$session_id
           WITH c, count(DISTINCT seed) AS matched_entities
           RETURN c.document_id AS document_id, c.filename AS filename, c.page AS page,
                  c.chunk_order AS chunk_order, c.text AS text, c.sha256 AS sha256,
                  c.parser AS parser, c.source_tier AS source_tier,
                  toFloat(matched_entities) AS score
           ORDER BY matched_entities DESC LIMIT $limit""",
        session_id=session_id,
        terms=terms,
        limit=limit,
    )
    return [dict(row) for row in result]
