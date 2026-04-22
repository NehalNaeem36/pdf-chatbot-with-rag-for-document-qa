from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

from retriever import RetrievalResult


DEFAULT_QA_MODEL = "deepset/roberta-base-squad2"
_TOKEN_RE = re.compile(r"\w+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class _FakeQAPipeline:
    def __call__(self, *, question: str, context: str) -> dict[str, object]:
        sentences = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(context) if item.strip()]
        if not sentences:
            sentences = [context.strip()]

        question_tokens = set(_TOKEN_RE.findall(question.lower()))
        best_sentence = ""
        best_score = 0.0

        for sentence in sentences:
            sentence_tokens = set(_TOKEN_RE.findall(sentence.lower()))
            if not question_tokens or not sentence_tokens:
                continue

            overlap = question_tokens & sentence_tokens
            if not overlap:
                continue

            coverage = len(overlap) / max(1, len(question_tokens))
            density = len(overlap) / max(1, len(sentence_tokens))
            score = float((coverage * 0.7) + (density * 0.3))
            if score > best_score:
                best_sentence = sentence
                best_score = score

        return {"answer": best_sentence, "score": best_score}


@dataclass(slots=True)
class AnswerResult:
    answer: str
    score: float
    page_number: int
    chunk_id: str
    source_file: str
    context: str
    retrieval_score: float
    reranker_score: float | None


@dataclass
class QAEngine:
    model_name: str = DEFAULT_QA_MODEL
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def _get_pipeline(self):
        if os.getenv("PDF_QA_FORCE_QA_FAILURE") == "1":
            raise RuntimeError("Forced QA failure for testing.")

        if os.getenv("PDF_QA_FAKE_QA") == "1":
            return _FakeQAPipeline()

        from transformers import pipeline

        if self._pipeline is None:
            self._pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                handle_impossible_answer=True,
            )
        return self._pipeline

    @staticmethod
    def _is_valid_answer(answer: str, context: str) -> bool:
        clean_answer = answer.strip()
        if not clean_answer:
            return False
        if clean_answer not in context:
            return False

        condensed_length = len(clean_answer.replace(" ", ""))
        if condensed_length >= 2:
            return True

        return clean_answer.isnumeric()

    @staticmethod
    def _build_answer_result(
        result: dict[str, object],
        evidence_item: RetrievalResult,
    ) -> AnswerResult | None:
        candidate_answer = str(result.get("answer", "")).strip()
        chunk = evidence_item.chunk
        if not QAEngine._is_valid_answer(candidate_answer, chunk.text):
            return None

        retrieval_score = (
            evidence_item.retrieval_score if evidence_item.retrieval_score is not None else evidence_item.score
        )
        return AnswerResult(
            answer=candidate_answer,
            score=float(result.get("score", 0.0)),
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            context=chunk.text,
            retrieval_score=float(retrieval_score),
            reranker_score=evidence_item.reranker_score,
        )

    def answer(self, question: str, evidence: list[RetrievalResult]) -> AnswerResult | None:
        if not evidence:
            return None

        qa_pipeline = self._get_pipeline()
        best_answer: AnswerResult | None = None

        for item in evidence:
            result = qa_pipeline(question=question, context=item.chunk.text)
            candidate = self._build_answer_result(result, item)
            if candidate is None:
                continue

            if best_answer is None:
                best_answer = candidate
                continue

            if candidate.score > best_answer.score:
                best_answer = candidate
                continue

            if candidate.score == best_answer.score:
                candidate_reranker = (
                    candidate.reranker_score if candidate.reranker_score is not None else float("-inf")
                )
                best_reranker = (
                    best_answer.reranker_score if best_answer.reranker_score is not None else float("-inf")
                )
                if candidate_reranker > best_reranker:
                    best_answer = candidate

        return best_answer
