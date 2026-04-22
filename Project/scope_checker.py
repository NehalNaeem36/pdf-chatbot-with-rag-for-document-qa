from __future__ import annotations

from dataclasses import dataclass

from qa_engine import AnswerResult
from retriever import RetrievalResult


@dataclass(slots=True)
class ScopeDecision:
    supported: bool
    reason: str


def is_supported(
    retrieval_results: list[RetrievalResult],
    answer: AnswerResult | None,
    max_distance: float = 250.0,
    min_qa_score: float = 0.20,
) -> ScopeDecision:
    if not retrieval_results:
        return ScopeDecision(False, "No relevant chunks were retrieved.")

    if retrieval_results[0].score > max_distance:
        return ScopeDecision(False, "Retrieved evidence is too weak.")

    if answer is None:
        return ScopeDecision(False, "No answer was produced.")

    if answer.score < min_qa_score:
        return ScopeDecision(False, "QA confidence is below threshold.")

    return ScopeDecision(True, "Supported by retrieved evidence.")
