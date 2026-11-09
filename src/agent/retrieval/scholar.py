"""Small arXiv Atom API client for public scholarly metadata and abstracts."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ScholarServiceError(RuntimeError):
    """Raised when the scholarly metadata service cannot be queried."""


def search_arxiv(query: str, limit: int = 5) -> list[dict[str, Any]]:
    search_terms = re.sub(r"\s+", " AND ", query.strip())
    url = (
        "https://export.arxiv.org/api/query?search_query=all:"
        f"{quote(search_terms, safe='')}&start=0&max_results={limit}&sortBy=relevance"
    )
    request = Request(url, headers={"User-Agent": "scientific-verification-agent/0.1"})
    try:
        with urlopen(request, timeout=15) as response:
            xml = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ScholarServiceError("The arXiv search request failed") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ScholarServiceError("arXiv returned invalid Atom XML") from exc

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", namespace):
        link = entry.find("atom:id", namespace)
        title = entry.find("atom:title", namespace)
        summary = entry.find("atom:summary", namespace)
        updated = entry.find("atom:updated", namespace)
        link_text = link.text if link is not None else None
        title_text = title.text if title is not None else None
        summary_text = summary.text if summary is not None else None
        authors = [
            (author.findtext("atom:name", default="", namespaces=namespace))
            for author in entry.findall("atom:author", namespace)
        ]
        results.append(
            {
                "id": link_text.strip() if link_text else "",
                "title": re.sub(r"\s+", " ", title_text or "").strip(),
                "abstract": re.sub(r"\s+", " ", summary_text or "").strip(),
                "updated": updated.text if updated is not None else None,
                "authors": authors,
                "source_tier": "t1_scholarly",
                "source": "arXiv",
            }
        )
    return results
