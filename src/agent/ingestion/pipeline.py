"""File parsing and chunking for user supplied evidence."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
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
    ocr_status: str = "not_needed"


@dataclass(frozen=True)
class TextChunk:
    page: int
    text: str
    start_offset: int
    end_offset: int
    kind: str = "paragraph"


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


def _ocr(image: bytes, language: str, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return "", "disabled"
    if shutil.which("tesseract") is None:
        return "", "unavailable"
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", language],
            input=image,
            capture_output=True,
            check=True,
            timeout=90,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "", "failed"
    text = result.stdout.decode("utf-8", errors="replace").strip()
    return text, "completed" if text else "no_text"


def parse_file(
    path: Path,
    filename: str,
    chunk_size: int,
    overlap: int,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
) -> ParsedFile:
    """Extract text while retaining page boundaries when the format has them."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        text, status = _ocr(path.read_bytes(), ocr_language, ocr_enabled)
        return ParsedFile(
            "tesseract" if status in {"completed", "no_text"} else "image-storage-only",
            filename,
            (ExtractedPage(1, text),) if text else (),
            status,
        )
    if suffix == ".pdf":
        import fitz  # type: ignore[import-untyped]

        with fitz.open(str(path)) as pdf:
            pages_list: list[ExtractedPage] = []
            ocr_states: list[str] = []
            used_ocr = False
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if not text:
                    text, status = _ocr(
                        page.get_pixmap(dpi=200).tobytes("png"), ocr_language, ocr_enabled
                    )
                    ocr_states.append(status)
                    used_ocr = used_ocr or status in {"completed", "no_text"}
                # OCR embedded figure regions when PyMuPDF exposes their image bytes.
                # This extracts labels only; it does not interpret chart geometry or data.
                if ocr_enabled:
                    try:
                        blocks = page.get_text("dict").get("blocks", [])
                        for block in blocks:
                            image_bytes = block.get("image") if block.get("type") == 1 else None
                            if image_bytes:
                                image_text, status = _ocr(image_bytes, ocr_language, True)
                                ocr_states.append(status)
                                if image_text:
                                    text += f"\n[FIGURE OCR TEXT]\n{image_text}"
                                    used_ocr = True
                    except (AttributeError, RuntimeError, ValueError):
                        pass
                # Keep detected tables searchable with their row/column structure. PyMuPDF
                # versions without find_tables still retain ordinary page text.
                try:
                    finder = getattr(page, "find_tables", None)
                    if finder is not None:
                        tables = finder().tables
                        rendered = []
                        for table in tables:
                            rows = table.extract()
                            if rows:
                                rendered.append(
                                    "\n[TABLE]\n"
                                    + "\n".join(
                                        " | ".join((cell or "").strip() for cell in row)
                                        for row in rows
                                    )
                                    + "\n[/TABLE]"
                                )
                        text = text + "\n" + "\n".join(rendered) if rendered else text
                except Exception:
                    pass
                pages_list.append(ExtractedPage(i + 1, text))
        state = (
            "completed"
            if "completed" in ocr_states
            else (ocr_states[0] if ocr_states else "not_needed")
        )
        return ParsedFile(
            "pymupdf+tesseract" if used_ocr else "pymupdf", filename, tuple(pages_list), state
        )
    if suffix == ".docx":
        from docx import Document as DocxDocument

        document = DocxDocument(str(path))
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        for table in document.tables:
            table_text = "\n".join(
                " | ".join(cell.text for cell in row.cells) for row in table.rows
            )
            text += "\n[TABLE]\n" + table_text + "\n[/TABLE]"
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
    return ParsedFile(
        "html-parser" if suffix in {".html", ".htm"} else "utf8-text",
        filename,
        (ExtractedPage(1, clean),),
    )


def chunk_pages_with_metadata(
    pages: tuple[ExtractedPage, ...], size: int, overlap: int
) -> list[TextChunk]:
    if size < 100 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be at least 100 and overlap must be smaller than size")
    chunks: list[TextChunk] = []
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
                is_table = "[TABLE]" in piece or " | " in piece
                is_equation = bool(re.search(r"(?:[=<>≤≥±×÷∑∫√]|\\(?:frac|sum|int|sqrt))", piece))
                kind = "table" if is_table else "equation" if is_equation else "paragraph"
                chunks.append(TextChunk(page.number, piece, start, end, kind))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def chunk_pages(pages: tuple[ExtractedPage, ...], size: int, overlap: int) -> list[tuple[int, str]]:
    """Compatibility helper returning page/text pairs."""
    return [(chunk.page, chunk.text) for chunk in chunk_pages_with_metadata(pages, size, overlap)]
