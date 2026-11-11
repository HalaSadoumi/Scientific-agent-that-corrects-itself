"""Deterministic, dependency-free query planning and candidate reranking."""

from __future__ import annotations

import re
from typing import Any

_SPLIT = re.compile(r"\s+(?:and|or|but|while|whereas)\s+|[?;.!]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "but",
    "the",
    "is",
    "what",
    "which",
    "with",
    "for",
    "to",
    "of",
    "in",
    "on",
    "does",
    "do",
}


def plan_queries(query: str, maximum: int = 4) -> list[str]:
    """Return the original query plus useful clauses, preserving the user's wording."""
    original = re.sub(r"\s+", " ", query).strip()
    clauses = [re.sub(r"\s+", " ", part).strip(" ,:-") for part in _SPLIT.split(original)]
    planned = [original]
    for clause in clauses:
        words = re.findall(r"\w+", clause.casefold())
        if (
            words
            and any(word not in _STOPWORDS for word in words)
            and clause.casefold() not in {value.casefold() for value in planned}
        ):
            planned.append(clause)
        if len(planned) >= maximum:
            break
    return planned


def rerank(query: str, hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Rerank candidates by query coverage, exact phrase, source tier, and base rank."""
    terms = {word.casefold() for word in re.findall(r"[\w]+", query) if len(word) > 1}
    tier_weight = {"t0_user": 1.0, "t1_scholarly": 1.06, "t2_reference": 1.03, "t3_web": 0.96}
    for hit in hits:
        text = hit["text"].casefold()
        present = {word for word in terms if re.search(rf"\b{re.escape(word)}\b", text)}
        coverage = len(present) / max(1, len(terms))
        phrase_bonus = 0.15 if query.casefold().strip(" ?.") in text else 0.0
        base = float(hit.get("score", 0.0))
        hit["score"] = (base + 0.25 * coverage + phrase_bonus) * tier_weight.get(
            hit.get("source_tier", "t0_user"), 1.0
        )
    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits[:limit]
