from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz


PROJECT_DIR = Path(__file__).resolve().parents[1]


def create_sample_pdf(pdf_path: Path, pages: list[str | None]) -> None:
    document = fitz.open()
    for page_text in pages:
        page = document.new_page()
        if page_text:
            page.insert_text((72, 72), page_text)
    document.save(pdf_path)
    document.close()


def test_modules_import() -> None:
    sys.path.insert(0, str(PROJECT_DIR))

    import app  # noqa: F401
    import chunker  # noqa: F401
    import embedder  # noqa: F401
    import pdf_reader  # noqa: F401
    import qa_engine  # noqa: F401
    import reranker  # noqa: F401
    import retriever  # noqa: F401
    import scope_checker  # noqa: F401


def test_main_fails_without_pdf_when_source_is_empty() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source"
        source_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                "--source-dir",
                str(source_dir),
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 1
    assert "No PDF found in" in result.stderr


def test_normalize_pages_preserves_paragraphs_and_skips_empty_pages() -> None:
    sys.path.insert(0, str(PROJECT_DIR))

    from pdf_reader import ExtractedPageData, normalize_pages

    raw_pages = [
        ExtractedPageData(
            page_number=1,
            source_file="sample.pdf",
            raw_text="First line   of para one.\nSecond line.\n\nThird paragraph.",
        ),
        ExtractedPageData(
            page_number=2,
            source_file="sample.pdf",
            raw_text=" \n\t \n",
        ),
    ]

    normalized = normalize_pages(raw_pages)

    assert len(normalized) == 1
    assert normalized[0].page_number == 1
    assert normalized[0].text == "First line of para one. Second line.\n\nThird paragraph."
    assert normalized[0].paragraph_count == 2


def test_chunk_pages_stay_within_page_and_include_metadata() -> None:
    sys.path.insert(0, str(PROJECT_DIR))

    from chunker import chunk_pages
    from pdf_reader import NormalizedPageData

    pages = [
        NormalizedPageData(
            page_number=1,
            source_file="sample.pdf",
            text="Paragraph one has enough text to build a chunk. " * 10
            + "\n\n"
            + "Paragraph two continues the page with more text. " * 10,
            char_count=0,
            paragraph_count=2,
        ),
        NormalizedPageData(
            page_number=3,
            source_file="sample.pdf",
            text="A later page stays separate. " * 20,
            char_count=0,
            paragraph_count=1,
        ),
    ]
    pages = [
        NormalizedPageData(
            page_number=page.page_number,
            source_file=page.source_file,
            text=page.text,
            char_count=len(page.text),
            paragraph_count=page.paragraph_count,
        )
        for page in pages
    ]

    chunks = chunk_pages(pages, target_chars=220, overlap_chars=40, min_chunk_chars=80)

    assert chunks
    assert all(chunk.page_number in {1, 3} for chunk in chunks)
    assert all(chunk.start_char < chunk.end_char for chunk in chunks)
    assert all(chunk.char_count == len(chunk.text) for chunk in chunks)
    assert all(chunk.chunk_id.startswith(f"p{chunk.page_number}-c") for chunk in chunks)


def test_main_writes_artifacts_for_pdf() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        extracted_dir = temp_path / "extracted"
        normalized_dir = temp_path / "normalized"

        source_dir.mkdir()
        pdf_path = source_dir / "fixture_sample.pdf"
        create_sample_pdf(
            pdf_path,
            [
                "First page first paragraph.\nStill first paragraph.\n\nSecond paragraph on page one.",
                None,
                "Third page text exists after an empty page.\n\nAnother paragraph for chunking.",
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                "--pdf",
                pdf_path.name,
                "--source-dir",
                str(source_dir),
                "--extracted-dir",
                str(extracted_dir),
                "--normalized-dir",
                str(normalized_dir),
                "--chunk-size",
                "120",
                "--overlap",
                "20",
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Raw extracted pages: 3" in result.stdout
        assert "Normalized pages: 2" in result.stdout
        assert "Chunks:" in result.stdout

        raw_pages_path = extracted_dir / "fixture_sample_pages_raw.json"
        normalized_pages_path = normalized_dir / "fixture_sample_pages_normalized.json"
        chunks_path = normalized_dir / "fixture_sample_chunks.json"

        assert raw_pages_path.exists()
        assert normalized_pages_path.exists()
        assert chunks_path.exists()

        raw_pages = json.loads(raw_pages_path.read_text(encoding="utf-8"))
        normalized_pages = json.loads(normalized_pages_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        assert len(raw_pages) == 3
        assert [page["page_number"] for page in normalized_pages] == [1, 3]
        assert all(chunk["page_number"] in {1, 3} for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)


def test_main_auto_detects_the_only_pdf_in_source() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        extracted_dir = temp_path / "extracted"
        normalized_dir = temp_path / "normalized"

        source_dir.mkdir()
        create_sample_pdf(
            source_dir / "only.pdf",
            [
                "Only PDF first page.\n\nSecond paragraph.",
                None,
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                "--source-dir",
                str(source_dir),
                "--extracted-dir",
                str(extracted_dir),
                "--normalized-dir",
                str(normalized_dir),
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0
    assert "Source PDF:" in result.stdout
    assert "only.pdf" in result.stdout


def test_main_fails_when_multiple_pdfs_exist_without_explicit_pdf() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        source_dir.mkdir()
        create_sample_pdf(source_dir / "alpha.pdf", ["Alpha page."])
        create_sample_pdf(source_dir / "beta.pdf", ["Beta page."])

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                "--source-dir",
                str(source_dir),
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 1
    assert "Multiple PDFs found in" in result.stderr
    assert "alpha.pdf" in result.stderr
    assert "beta.pdf" in result.stderr
