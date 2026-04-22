from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

from retriever import RetrievalResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOKEN_RE = re.compile(r"\w+")


class _FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._score_pair(query, chunk_text) for query, chunk_text in pairs]

    @staticmethod
    def _score_pair(query: str, chunk_text: str) -> float:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        chunk_tokens = set(_TOKEN_RE.findall(chunk_text.lower()))
        if not query_tokens or not chunk_tokens:
            return 0.0

        overlap = query_tokens & chunk_tokens
        coverage = len(overlap) / max(1, len(query_tokens))
        density = len(overlap) / max(1, len(chunk_tokens))
        return float((coverage * 10.0) + density)


@dataclass
class Reranker:
    model_name: str = DEFAULT_RERANKER_MODEL
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self):
        if os.getenv("PDF_QA_FORCE_RERANKER_FAILURE") == "1":
            raise RuntimeError("Forced reranker failure for testing.")

        if os.getenv("PDF_QA_FAKE_RERANKER") == "1":
            return _FakeCrossEncoder(self.model_name)

        from sentence_transformers import CrossEncoder

        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, result.chunk.text) for result in results]
        scores = self._get_model().predict(pairs)

        reranked = [
            RetrievalResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores, strict=False)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked
