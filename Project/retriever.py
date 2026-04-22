from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chunker import ChunkData


@dataclass(slots=True)
class RetrievalResult:
    chunk: ChunkData
    score: float


class Retriever:
    def __init__(self) -> None:
        self.index = None
        self.chunks: list[ChunkData] = []

    def build(self, embeddings: np.ndarray, chunks: list[ChunkData]) -> None:
        import faiss

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        if len(chunks) == 0:
            raise ValueError("at least one chunk is required to build the index")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("index has not been built")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query, top_k)

        results: list[RetrievalResult] = []
        for score, index in zip(distances[0], indices[0], strict=False):
            if index < 0:
                continue
            results.append(RetrievalResult(chunk=self.chunks[index], score=float(score)))

        return results
