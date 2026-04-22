from __future__ import annotations

from dataclasses import dataclass

from pdf_reader import PageData


@dataclass(slots=True)
class ChunkData:
    chunk_id: str
    page_number: int
    source_file: str
    text: str


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[ChunkData]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[ChunkData] = []

    for page in pages:
        start = 0
        chunk_index = 0

        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            chunk_text = page.text[start:end].strip()

            if chunk_text:
                chunks.append(
                    ChunkData(
                        chunk_id=f"p{page.page_number}-c{chunk_index}",
                        page_number=page.page_number,
                        source_file=page.source_file,
                        text=chunk_text,
                    )
                )

            if end >= len(page.text):
                break

            start = end - overlap
            chunk_index += 1

    return chunks
