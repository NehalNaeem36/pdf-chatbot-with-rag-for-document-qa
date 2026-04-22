from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_INLINE_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+", re.MULTILINE)


@dataclass(slots=True)
class ExtractedPageData:
    page_number: int
    source_file: str
    raw_text: str


@dataclass(slots=True)
class NormalizedPageData:
    page_number: int
    source_file: str
    text: str
    char_count: int
    paragraph_count: int


def normalize_page_text(text: str) -> str:
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    collapsed_blank_lines = _BLANK_LINE_RE.sub("\n\n", normalized_newlines.strip())

    paragraphs: list[str] = []
    for raw_paragraph in collapsed_blank_lines.split("\n\n"):
        lines = [line.strip() for line in raw_paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        paragraph_text = _INLINE_WHITESPACE_RE.sub(" ", " ".join(lines)).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def extract_raw_pdf_pages(pdf_path: str | Path) -> list[ExtractedPageData]:
    import fitz

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[ExtractedPageData] = []

    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(
                ExtractedPageData(
                    page_number=index,
                    source_file=path.name,
                    raw_text=page.get_text("text"),
                )
            )

    return pages


def normalize_pages(pages: list[ExtractedPageData]) -> list[NormalizedPageData]:
    normalized_pages: list[NormalizedPageData] = []

    for page in pages:
        cleaned_text = normalize_page_text(page.raw_text)
        if not cleaned_text:
            continue

        paragraph_count = len(cleaned_text.split("\n\n"))
        normalized_pages.append(
            NormalizedPageData(
                page_number=page.page_number,
                source_file=page.source_file,
                text=cleaned_text,
                char_count=len(cleaned_text),
                paragraph_count=paragraph_count,
            )
        )

    return normalized_pages
