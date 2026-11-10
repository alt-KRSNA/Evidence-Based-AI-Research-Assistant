
from pathlib import Path
from typing import Any

from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator


class ResearchPipeline:
    """Coordinate retrieval and evidence-grounded answer generation."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.answer_generator = AnswerGenerator()

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Ingest a PDF and index its contents."""

        return self.retriever.ingest_pdf(pdf_path)

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve evidence and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant evidence
        evidence = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # Step 2: Generate an answer from retrieved evidence
        answer = self.answer_generator.generate_answer(
            question=question,
            retrieved_chunks=evidence,
        )

        # Step 3: Return answer and supporting evidence
        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
        }
