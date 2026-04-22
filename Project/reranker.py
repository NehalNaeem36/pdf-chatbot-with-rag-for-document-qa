from __future__ import annotations

from dataclasses import dataclass, field

from retriever import RetrievalResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Reranker:
    model_name: str = DEFAULT_RERANKER_MODEL
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self):
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
