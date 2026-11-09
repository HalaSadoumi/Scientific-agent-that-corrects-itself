"""File parsing and chunking for user supplied evidence."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str


@dataclass(frozen=True)
class ParsedFile:
    parser: str
    title: str
    pages: tuple[ExtractedPage, ...]


SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def parse_file(path: Path, filename: str, chunk_size: int, overlap: int) -> ParsedFile:
    """Extract text while retaining page boundaries when the format has them."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return ParsedFile("image-storage-only", filename, ())
    if suffix == ".pdf":
        import fitz

        with fitz.open(path) as pdf:
            pages = tuple(ExtractedPage(i + 1, page.get_text("text")) for i, page in enumerate(pdf))
        return ParsedFile("pymupdf", filename, pages)
    if suffix == ".docx":
        from docx import Document as DocxDocument

        document = DocxDocument(path)
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        for table in document.tables:
            table_text = "\n".join(
                " | ".join(cell.text for cell in row.cells) for row in table.rows
            )
            text += "\n" + table_text
        return ParsedFile("python-docx", filename, (ExtractedPage(1, text),))
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: list[str] = []
                self.hidden_depth = 0

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag in {"script", "style", "noscript"}:
                    self.hidden_depth += 1
                elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
                    self.parts.append("\n")

            def handle_endtag(self, tag: str) -> None:
                if tag in {"script", "style", "noscript"} and self.hidden_depth:
                    self.hidden_depth -= 1
                elif tag in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
                    self.parts.append("\n")

            def handle_data(self, data: str) -> None:
                if not self.hidden_depth:
                    self.parts.append(data)

        extractor = TextExtractor()
        extractor.feed(raw)
        raw = html.unescape("".join(extractor.parts))
    clean = re.sub(r"[ \t]+", " ", raw)
    return ParsedFile("html-parser" if suffix in {".html", ".htm"} else "utf8-text", filename,
                      (ExtractedPage(1, clean),))


def chunk_pages(pages: tuple[ExtractedPage, ...], size: int, overlap: int) -> list[tuple[int, str]]:
    if size < 100 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be at least 100 and overlap must be smaller than size")
    chunks: list[tuple[int, str]] = []
    for page in pages:
        text = re.sub(r"\s+", " ", page.text).strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append((page.number, piece))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
