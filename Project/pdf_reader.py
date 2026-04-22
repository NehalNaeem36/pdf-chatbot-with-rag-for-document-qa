from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PageData:
    page_number: int
    text: str
    source_file: str


def clean_page_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[PageData]:
    import fitz

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[PageData] = []

    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            cleaned_text = clean_page_text(page.get_text("text"))
            if not cleaned_text:
                continue

            pages.append(
                PageData(
                    page_number=index,
                    text=cleaned_text,
                    source_file=path.name,
                )
            )

    return pages
