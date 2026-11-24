"""Small, explainable query router for astronomy-focused retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    method: str
    reason: str


_GRAPH_INTENT = re.compile(
    r"\b(relationship|related|connect(?:ed|ion)?|compare|between|associated|"
    r"observed by|measured by|which instrument|which survey|how does .* affect)\b",
    re.IGNORECASE,
)
_EXACT_INTENT = re.compile(
    r"\b(doi|arxiv|ngc\s*\d+|ic\s*\d+|hd\s*\d+|hip\s*\d+|"
    r"gaia\s+(?:dr\d+|edr\d+)|kepler-?\d+|toi-?\d+|2mass\s*j)\b",
    re.IGNORECASE,
)


def choose_route(query: str, *, embeddings: bool, graph: bool) -> Route:
    """Choose a route using visible rules; no hidden LLM call or random method selection."""
    if graph and _GRAPH_INTENT.search(query):
        return Route("graph_hybrid", "The question asks about links between astronomy entities.")
    if graph and _EXACT_INTENT.search(query):
        return Route(
            "graph_hybrid", "The question contains a recognizable astronomy/catalog identifier."
        )
    if embeddings:
        return Route("hybrid_rrf", "Keyword and astronomy-semantic search are both available.")
    return Route(
        "sqlite_fts5_bm25", "Only local keyword search is configured; using the safe baseline."
    )
