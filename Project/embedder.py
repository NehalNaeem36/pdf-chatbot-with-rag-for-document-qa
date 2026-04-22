from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Embedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self):
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
