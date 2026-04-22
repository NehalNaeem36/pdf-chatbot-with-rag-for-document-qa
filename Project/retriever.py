from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from chunker import ChunkData


@dataclass(slots=True)
class RetrievalResult:
    chunk: ChunkData
    score: float
    retrieval_score: float | None = None
    reranker_score: float | None = None


@dataclass(slots=True)
class IndexArtifactMetadata:
    source_file: str
    chunk_artifact: str
    embedding_artifact: str
    embedding_model: str
    num_chunks: int
    embedding_dimension: int
    similarity_metric: str
    faiss_index_file: str


class Retriever:
    def __init__(self) -> None:
        self.index = None
        self.chunks: list[ChunkData] = []
        self.embedding_dimension: int | None = None

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return matrix / norms

    @staticmethod
    def _normalize_query_embedding(query_embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return vector / norm

    def build(self, embeddings: np.ndarray, chunks: list[ChunkData]) -> None:
        import faiss

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        if len(chunks) == 0:
            raise ValueError("at least one chunk is required to build the index")

        normalized_embeddings = self._normalize_embeddings(embeddings)
        self.index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        self.index.add(normalized_embeddings.astype(np.float32))
        self.chunks = chunks
        self.embedding_dimension = normalized_embeddings.shape[1]

    def save(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path,
        source_file: str,
        chunk_artifact: str | Path,
        embedding_artifact: str | Path,
        embedding_model: str,
    ) -> tuple[Path, Path]:
        import faiss

        if self.index is None or self.embedding_dimension is None:
            raise RuntimeError("index has not been built")

        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))

        metadata = IndexArtifactMetadata(
            source_file=source_file,
            chunk_artifact=str(chunk_artifact),
            embedding_artifact=str(embedding_artifact),
            embedding_model=embedding_model,
            num_chunks=len(self.chunks),
            embedding_dimension=self.embedding_dimension,
            similarity_metric="cosine_ip_normalized",
            faiss_index_file=str(index_file),
        )
        metadata_file.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

        return index_file, metadata_file

    def load(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path,
        chunks: list[ChunkData],
    ) -> IndexArtifactMetadata:
        import faiss

        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Index metadata not found: {metadata_file}")

        metadata_dict = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata = IndexArtifactMetadata(**metadata_dict)

        if metadata.num_chunks != len(chunks):
            raise ValueError("index metadata chunk count does not match chunk artifact")
        if metadata.similarity_metric != "cosine_ip_normalized":
            raise ValueError("unsupported similarity metric in index metadata")

        self.index = faiss.read_index(str(index_file))
        self.chunks = chunks
        self.embedding_dimension = metadata.embedding_dimension
        if getattr(self.index, "d", None) != metadata.embedding_dimension:
            raise ValueError("loaded FAISS index dimension does not match index metadata")

        return metadata

    def build_or_load(
        self,
        *,
        embeddings: np.ndarray,
        chunks: list[ChunkData],
        index_path: str | Path,
        metadata_path: str | Path,
        source_file: str,
        chunk_artifact: str | Path,
        embedding_artifact: str | Path,
        embedding_model: str,
    ) -> tuple[Path, Path, IndexArtifactMetadata]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if matrix.shape[0] != len(chunks):
            raise ValueError("embedding row count must match chunk count before index build")

        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        if index_file.exists() and metadata_file.exists():
            metadata = self.load(
                index_path=index_file,
                metadata_path=metadata_file,
                chunks=chunks,
            )
            if metadata.chunk_artifact != str(chunk_artifact):
                raise ValueError("index metadata chunk artifact does not match current chunk artifact")
            if metadata.embedding_artifact != str(embedding_artifact):
                raise ValueError("index metadata embedding artifact does not match current embedding artifact")
            if metadata.embedding_dimension != matrix.shape[1]:
                raise ValueError("index metadata embedding dimension does not match current embeddings")
            return index_file, metadata_file, metadata

        self.build(matrix, chunks)
        saved_index_path, saved_metadata_path = self.save(
            index_path=index_file,
            metadata_path=metadata_file,
            source_file=source_file,
            chunk_artifact=chunk_artifact,
            embedding_artifact=embedding_artifact,
            embedding_model=embedding_model,
        )
        metadata = IndexArtifactMetadata(
            source_file=source_file,
            chunk_artifact=str(chunk_artifact),
            embedding_artifact=str(embedding_artifact),
            embedding_model=embedding_model,
            num_chunks=len(chunks),
            embedding_dimension=self.embedding_dimension or 0,
            similarity_metric="cosine_ip_normalized",
            faiss_index_file=str(saved_index_path),
        )
        return saved_index_path, saved_metadata_path, metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("index has not been built")

        query = self._normalize_query_embedding(query_embedding)
        scores, indices = self.index.search(query, top_k)

        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            numeric_score = float(score)
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=numeric_score,
                    retrieval_score=numeric_score,
                )
            )

        return results
