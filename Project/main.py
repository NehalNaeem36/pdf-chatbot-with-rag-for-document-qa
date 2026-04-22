from __future__ import annotations

import argparse
from pathlib import Path
import sys

from chunker import chunk_pages, save_chunks_artifact
from embedder import Embedder, save_embedding_artifacts, validate_embeddings
from pdf_reader import (
    extract_raw_pdf_pages,
    normalize_pages,
    resolve_pdf_input,
    save_page_artifacts,
)
from reranker import Reranker
from retriever import Retriever


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_SOURCE_DIR = DATA_DIR / "source"
DEFAULT_EXTRACTED_DIR = DATA_DIR / "extracted"
DEFAULT_NORMALIZED_DIR = DATA_DIR / "normalized"
DEFAULT_EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DEFAULT_INDEXES_DIR = DATA_DIR / "indexes"
DEFAULT_RERANK_INPUT_K = 10
DEFAULT_FINAL_RESULTS = 3


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
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory for embedding artifacts.",
    )
    parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
        help="Directory for FAISS index artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RERANK_INPUT_K,
        help="Number of FAISS candidates to retrieve before reranking.",
    )
    return parser


def preview_text(text: str, limit: int = 100) -> str:
    single_line = text.replace("\n", " ")
    return single_line[:limit]


def print_retrieval_results(results: list, *, limit: int = 220) -> None:
    if not results:
        print("No chunks retrieved.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result.score:.4f} | page={result.chunk.page_number} | "
            f"chunk={result.chunk.chunk_id}"
        )
        print(preview_text(result.chunk.text, limit=limit))


def interactive_retrieval_loop(
    *,
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    top_k: int,
) -> None:
    while True:
        try:
            question = input("Ask a question (or type 'exit'): ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        query_embedding = embedder.encode_query(question)
        results = retriever.search(query_embedding, top_k=top_k)

        try:
            reranked_results = reranker.rerank(question, results)
            final_results = reranked_results[:DEFAULT_FINAL_RESULTS]
        except Exception:
            print("Reranker unavailable, showing raw retrieval results.")
            final_results = results[:DEFAULT_FINAL_RESULTS]

        print_retrieval_results(final_results)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        pdf_path = resolve_pdf_input(args.pdf, args.source_dir)
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
    raw_pages_path, normalized_pages_path = save_page_artifacts(
        raw_pages,
        normalized_pages,
        extracted_dir=args.extracted_dir,
        normalized_dir=args.normalized_dir,
        pdf_stem=pdf_stem,
    )
    chunks_path = save_chunks_artifact(
        chunks,
        normalized_dir=args.normalized_dir,
        pdf_stem=pdf_stem,
    )

    embedder = Embedder()
    embeddings = embedder.encode_texts([chunk.text for chunk in chunks])
    embedding_rows, embedding_dimension = validate_embeddings(embeddings, len(chunks))
    embeddings_path, embeddings_meta_path = save_embedding_artifacts(
        embeddings,
        output_dir=args.embeddings_dir,
        pdf_stem=pdf_stem,
        source_file=pdf_path.name,
        chunk_artifact_path=chunks_path,
        model_name=embedder.model_name,
    )
    index_path = args.indexes_dir / f"{pdf_stem}_faiss.index"
    index_meta_path = args.indexes_dir / f"{pdf_stem}_index_meta.json"

    try:
        retriever = Retriever()
        saved_index_path, saved_index_meta_path, _ = retriever.build_or_load(
            embeddings=embeddings,
            chunks=chunks,
            index_path=index_path,
            metadata_path=index_meta_path,
            source_file=pdf_path.name,
            chunk_artifact=chunks_path,
            embedding_artifact=embeddings_path,
            embedding_model=embedder.model_name,
        )
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Source PDF: {pdf_path}")
    print(f"Raw extracted pages: {len(raw_pages)}")
    print(f"Normalized pages: {len(normalized_pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {embedding_rows}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Saved raw pages to: {raw_pages_path}")
    print(f"Saved normalized pages to: {normalized_pages_path}")
    print(f"Saved chunks to: {chunks_path}")
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved embedding metadata to: {embeddings_meta_path}")
    print(f"Saved FAISS index to: {saved_index_path}")
    print(f"Saved index metadata to: {saved_index_meta_path}")

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

    reranker = Reranker()

    interactive_retrieval_loop(
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
        top_k=args.top_k,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
