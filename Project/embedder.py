from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAKE_EMBEDDING_DIMENSION = 8


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        vectors = [self._encode_single_text(text) for text in texts]
        matrix = np.vstack(vectors) if vectors else np.empty((0, FAKE_EMBEDDING_DIMENSION), dtype=np.float32)
        return matrix if convert_to_numpy else matrix.tolist()

    @staticmethod
    def _encode_single_text(text: str) -> np.ndarray:
        vector = np.zeros(FAKE_EMBEDDING_DIMENSION, dtype=np.float32)
        encoded = text.encode("utf-8")
        for index, byte in enumerate(encoded):
            vector[index % FAKE_EMBEDDING_DIMENSION] += byte / 255.0
        if encoded:
            vector /= max(1, len(encoded))
        return vector


@dataclass(slots=True)
class EmbeddingArtifactMetadata:
    source_file: str
    chunk_artifact: str
    embedding_model: str
    num_chunks: int
    embedding_dimension: int
    embedding_file: str


def validate_embeddings(embeddings: np.ndarray, expected_rows: int) -> tuple[int, int]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if matrix.shape[0] != expected_rows:
        raise ValueError("embedding row count must match chunk count")
    return matrix.shape


def save_embedding_artifacts(
    embeddings: np.ndarray,
    *,
    output_dir: str | Path,
    pdf_stem: str,
    source_file: str,
    chunk_artifact_path: str | Path,
    model_name: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(embeddings, dtype=np.float32)
    num_chunks, embedding_dimension = validate_embeddings(matrix, matrix.shape[0])

    embeddings_path = output_path / f"{pdf_stem}_embeddings.npy"
    metadata_path = output_path / f"{pdf_stem}_embeddings_meta.json"

    np.save(embeddings_path, matrix)

    metadata = EmbeddingArtifactMetadata(
        source_file=source_file,
        chunk_artifact=str(chunk_artifact_path),
        embedding_model=model_name,
        num_chunks=num_chunks,
        embedding_dimension=embedding_dimension,
        embedding_file=str(embeddings_path),
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    return embeddings_path, metadata_path


@dataclass
class Embedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self):
        if os.getenv("PDF_QA_FAKE_EMBEDDINGS") == "1":
            return _FakeSentenceTransformer(self.model_name)

        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self._get_model().encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._get_model().encode([query], convert_to_numpy=True)[0]
        return np.asarray(vector, dtype=np.float32)
