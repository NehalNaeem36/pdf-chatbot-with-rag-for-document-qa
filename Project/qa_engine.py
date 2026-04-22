from __future__ import annotations

from dataclasses import dataclass, field

from retriever import RetrievalResult


DEFAULT_QA_MODEL = "distilbert-base-cased-distilled-squad"


@dataclass(slots=True)
class AnswerResult:
    answer: str
    score: float
    page_number: int
    chunk_id: str


@dataclass
class QAEngine:
    model_name: str = DEFAULT_QA_MODEL
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def _get_pipeline(self):
        from transformers import pipeline

        if self._pipeline is None:
            self._pipeline = pipeline("question-answering", model=self.model_name)
        return self._pipeline

    def answer(self, question: str, evidence: list[RetrievalResult]) -> AnswerResult | None:
        if not evidence:
            return None

        best_chunk = evidence[0].chunk
        result = self._get_pipeline()(question=question, context=best_chunk.text)

        return AnswerResult(
            answer=result["answer"],
            score=float(result["score"]),
            page_number=best_chunk.page_number,
            chunk_id=best_chunk.chunk_id,
        )
