from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
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


def _save_json_artifact(data: list[object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) if is_dataclass(item) else item for item in data]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return path


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


def resolve_pdf_input(pdf_arg: Path | None, source_dir: str | Path) -> Path:
    source_path = Path(source_dir)

    if pdf_arg is None:
        return find_default_pdf(source_path)

    if pdf_arg.exists():
        return pdf_arg

    if pdf_arg.parent == Path(".") or len(pdf_arg.parts) == 1:
        candidate = source_path / pdf_arg.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"PDF not found: {pdf_arg}")


def find_default_pdf(source_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(
            f"No PDF found in {source_path}. Add one or pass --pdf."
        )

    pdf_candidates = sorted(
        path
        for path in source_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_candidates:
        raise FileNotFoundError(
            f"No PDF found in {source_path}. Add one or pass --pdf."
        )

    if len(pdf_candidates) > 1:
        names = ", ".join(path.name for path in pdf_candidates)
        raise RuntimeError(
            f"Multiple PDFs found in {source_path}: {names}. Pass --pdf to choose one."
        )

    return pdf_candidates[0]


def save_page_artifacts(
    raw_pages: list[ExtractedPageData],
    normalized_pages: list[NormalizedPageData],
    *,
    extracted_dir: str | Path,
    normalized_dir: str | Path,
    pdf_stem: str,
) -> tuple[Path, Path]:
    raw_pages_path = Path(extracted_dir) / f"{pdf_stem}_pages_raw.json"
    normalized_pages_path = Path(normalized_dir) / f"{pdf_stem}_pages_normalized.json"

    _save_json_artifact(raw_pages, raw_pages_path)
    _save_json_artifact(normalized_pages, normalized_pages_path)

    return raw_pages_path, normalized_pages_path
