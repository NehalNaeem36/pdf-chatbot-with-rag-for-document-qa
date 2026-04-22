from __future__ import annotations

import argparse
from pathlib import Path

from pdf_reader import extract_pdf_pages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local PDF QA starter entrypoint."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to a text-based PDF file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.pdf is None:
        print("PDF QA skeleton is ready. Provide --pdf <path> to test PDF extraction.")
        return 0

    pages = extract_pdf_pages(args.pdf)
    print(f"Loaded {len(pages)} non-empty pages from {args.pdf.name}")

    for page in pages:
        preview = page.text[:120].replace("\n", " ")
        print(f"Page {page.page_number}: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
