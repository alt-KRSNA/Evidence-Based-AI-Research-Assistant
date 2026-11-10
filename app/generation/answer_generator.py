from app.generation.llm import GroqClient
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt


class AnswerGenerator:
    """Generate evidence-grounded answers from retrieved chunks."""

    def __init__(self):
        self.llm = GroqClient()

    @staticmethod
    def format_context(
        retrieved_chunks: list[tuple[dict, float]]
    ) -> str:
        """Format retrieved chunks into readable evidence."""

        context_parts = []

        for index, (chunk, score) in enumerate(
            retrieved_chunks, start=1
        ):
            filename = chunk.get(
                "filename",
                chunk.get("source", "Unknown document")
            )

            page_number = chunk.get(
                "page_number",
                chunk.get("page", "Unknown")
            )

            text = chunk.get("text", "")

            context_parts.append(
                f"""
EVIDENCE {index}
Source: {filename}
Page: {page_number}
Similarity Score: {score:.4f}

Content:
{text}
"""
            )

        return "\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[tuple[dict, float]],
    ) -> str:
        """Generate an answer using retrieved evidence."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return (
                "No relevant evidence was found in the uploaded "
                "documents. Please upload a document containing "
                "information related to your question."
            )

        context = self.format_context(retrieved_chunks)

        user_prompt = build_user_prompt(
            question=question,
            context=context,
        )

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return answer