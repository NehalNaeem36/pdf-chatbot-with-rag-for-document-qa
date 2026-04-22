from __future__ import annotations

from dataclasses import dataclass

from qa_engine import AnswerResult
from retriever import RetrievalResult


ABSTENTION_MESSAGE = "This is outside the scope of the PDF."


@dataclass(slots=True)
class ScopeDecision:
    supported: bool
    reason: str


def decide_support(
    retrieval_results: list[RetrievalResult],
    reranked_results: list[RetrievalResult],
    answer: AnswerResult | None,
    min_retrieval_score: float = 0.20,
    min_reranker_score: float = 0.0,
    min_qa_score: float = 0.20,
) -> ScopeDecision:
    if not retrieval_results:
        return ScopeDecision(False, "No relevant chunks were retrieved.")

    top_retrieval_score = (
        retrieval_results[0].retrieval_score
        if retrieval_results[0].retrieval_score is not None
        else retrieval_results[0].score
    )
    if top_retrieval_score < min_retrieval_score:
        return ScopeDecision(False, "Retrieved evidence is too weak.")

    if reranked_results:
        top_reranker_score = reranked_results[0].reranker_score
        if top_reranker_score is not None and top_reranker_score < min_reranker_score:
            return ScopeDecision(False, "Reranked evidence is too weak.")

    if answer is None:
        return ScopeDecision(False, "No valid answer was produced.")

    if answer.score < min_qa_score:
        return ScopeDecision(False, "QA confidence is below threshold.")

    return ScopeDecision(True, "Supported by retrieved evidence.")
