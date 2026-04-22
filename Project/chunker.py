from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import re

from pdf_reader import NormalizedPageData


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ChunkData:
    chunk_id: str
    page_number: int
    source_file: str
    text: str
    char_count: int
    start_char: int
    end_char: int


def _split_into_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_BOUNDARY_RE.split(text) if segment.strip()]
    return sentences or [text.strip()]


def _hard_split_segment(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    return pieces


def _split_paragraph_to_segments(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    segments: list[str] = []
    current = ""

    for sentence in _split_into_sentences(paragraph):
        if len(sentence) > max_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_hard_split_segment(sentence, max_chars))
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            segments.append(current)
            current = sentence

    if current:
        segments.append(current)

    return segments


def _page_segments(page: NormalizedPageData, max_chars: int) -> list[tuple[int, int, str]]:
    segments: list[tuple[int, int, str]] = []
    cursor = 0

    for paragraph in page.text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            cursor += 2
            continue

        paragraph_start = page.text.find(paragraph, cursor)
        if paragraph_start == -1:
            paragraph_start = cursor

        local_cursor = paragraph_start
        for segment in _split_paragraph_to_segments(paragraph, max_chars):
            segment_start = page.text.find(segment, local_cursor)
            if segment_start == -1:
                segment_start = local_cursor
            segment_end = segment_start + len(segment)
            segments.append((segment_start, segment_end, segment))
            local_cursor = segment_end

        cursor = paragraph_start + len(paragraph) + 2

    return segments


def _slice_overlap_source(page_text: str, chunk_start: int, overlap_chars: int) -> tuple[int, str]:
    if overlap_chars <= 0 or chunk_start <= 0:
        return chunk_start, ""
    overlap_start = max(0, chunk_start - overlap_chars)
    return overlap_start, page_text[overlap_start:chunk_start].strip()


def _append_chunk(
    chunks: list[ChunkData],
    page: NormalizedPageData,
    chunk_index: int,
    start_char: int,
    end_char: int,
    text: str,
) -> None:
    clean_text = text.strip()
    if not clean_text:
        return

    chunks.append(
        ChunkData(
            chunk_id=f"p{page.page_number}-c{chunk_index}",
            page_number=page.page_number,
            source_file=page.source_file,
            text=clean_text,
            char_count=len(clean_text),
            start_char=start_char,
            end_char=end_char,
        )
    )


def chunk_pages(
    pages: list[NormalizedPageData],
    target_chars: int = 900,
    overlap_chars: int = 150,
    min_chunk_chars: int = 200,
) -> list[ChunkData]:
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than target_chars")
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be positive")

    chunks: list[ChunkData] = []

    for page in pages:
        segments = _page_segments(page, target_chars)
        if not segments:
            continue

        chunk_index = 0
        current_text = ""
        current_start = segments[0][0]
        current_end = segments[0][0]

        for index, (segment_start, segment_end, segment_text) in enumerate(segments):
            if not current_text:
                overlap_start, overlap_text = _slice_overlap_source(page.text, segment_start, overlap_chars)
                current_text = segment_text if not overlap_text else f"{overlap_text} {segment_text}"
                current_text = current_text.strip()
                current_start = overlap_start if overlap_text else segment_start
                current_end = segment_end
                continue

            candidate_text = f"{current_text}\n\n{segment_text}".strip()
            if len(candidate_text) <= target_chars:
                current_text = candidate_text
                current_end = segment_end
                continue

            _append_chunk(
                chunks=chunks,
                page=page,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_end,
                text=current_text,
            )
            chunk_index += 1

            overlap_start, overlap_text = _slice_overlap_source(page.text, segment_start, overlap_chars)
            next_text = segment_text if not overlap_text else f"{overlap_text} {segment_text}"
            current_text = next_text.strip()
            current_start = overlap_start if overlap_text else segment_start
            current_end = segment_end

            if (
                len(current_text) < min_chunk_chars
                and index + 1 < len(segments)
            ):
                continue

        _append_chunk(
            chunks=chunks,
            page=page,
            chunk_index=chunk_index,
            start_char=current_start,
            end_char=current_end,
            text=current_text,
        )

    return chunks


def save_chunks_artifact(
    chunks: list[ChunkData],
    *,
    normalized_dir: str | Path,
    pdf_stem: str,
) -> Path:
    chunks_path = Path(normalized_dir) / f"{pdf_stem}_chunks.json"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) if is_dataclass(item) else item for item in chunks]
    chunks_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return chunks_path
