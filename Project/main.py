from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

from chunker import chunk_pages
from pdf_reader import extract_raw_pdf_pages, normalize_pages


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_SOURCE_DIR = DATA_DIR / "source"
DEFAULT_EXTRACTED_DIR = DATA_DIR / "extracted"
DEFAULT_NORMALIZED_DIR = DATA_DIR / "normalized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local PDF ingestion and chunking entrypoint."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to a text-based PDF file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Target character length for each chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Character overlap between chunks on the same page.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Default lookup directory for bare PDF filenames.",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=DEFAULT_EXTRACTED_DIR,
        help="Directory for raw extracted page artifacts.",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=DEFAULT_NORMALIZED_DIR,
        help="Directory for normalized page and chunk artifacts.",
    )
    return parser


def resolve_pdf_path(pdf_arg: Path, source_dir: Path) -> Path:
    if pdf_arg.exists():
        return pdf_arg
    if pdf_arg.parent == Path(".") or len(pdf_arg.parts) == 1:
        candidate = source_dir / pdf_arg.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"PDF not found: {pdf_arg}")


def find_default_pdf(source_dir: Path) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"No PDF found in {source_dir}. Add one or pass --pdf."
        )

    pdf_candidates = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_candidates:
        raise FileNotFoundError(
            f"No PDF found in {source_dir}. Add one or pass --pdf."
        )

    if len(pdf_candidates) > 1:
        names = ", ".join(path.name for path in pdf_candidates)
        raise RuntimeError(
            f"Multiple PDFs found in {source_dir}: {names}. Pass --pdf to choose one."
        )

    return pdf_candidates[0]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json_artifact(data: list[object], output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)

    serialized = [asdict(item) if is_dataclass(item) else item for item in data]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return path


def preview_text(text: str, limit: int = 100) -> str:
    single_line = text.replace("\n", " ")
    return single_line[:limit]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.pdf is None:
            pdf_path = find_default_pdf(args.source_dir)
        else:
            pdf_path = resolve_pdf_path(args.pdf, args.source_dir)
    except (FileNotFoundError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    raw_pages = extract_raw_pdf_pages(pdf_path)
    normalized_pages = normalize_pages(raw_pages)
    chunks = chunk_pages(
        normalized_pages,
        target_chars=args.chunk_size,
        overlap_chars=args.overlap,
    )

    pdf_stem = pdf_path.stem
    raw_pages_path = args.extracted_dir / f"{pdf_stem}_pages_raw.json"
    normalized_pages_path = args.normalized_dir / f"{pdf_stem}_pages_normalized.json"
    chunks_path = args.normalized_dir / f"{pdf_stem}_chunks.json"

    save_json_artifact(raw_pages, raw_pages_path)
    save_json_artifact(normalized_pages, normalized_pages_path)
    save_json_artifact(chunks, chunks_path)

    print(f"Source PDF: {pdf_path}")
    print(f"Raw extracted pages: {len(raw_pages)}")
    print(f"Normalized pages: {len(normalized_pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Saved raw pages to: {raw_pages_path}")
    print(f"Saved normalized pages to: {normalized_pages_path}")
    print(f"Saved chunks to: {chunks_path}")

    for page in normalized_pages[:3]:
        print(
            f"Page {page.page_number} ({page.char_count} chars, {page.paragraph_count} paragraphs): "
            f"{preview_text(page.text, limit=120)}"
        )

    for chunk in chunks[:3]:
        print(
            f"Chunk {chunk.chunk_id} [page {chunk.page_number}, chars {chunk.start_char}:{chunk.end_char}]: "
            f"{preview_text(chunk.text, limit=120)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
