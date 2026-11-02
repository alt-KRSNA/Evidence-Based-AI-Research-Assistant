
SYSTEM_PROMPT = """
You are an evidence-based research assistant.

Rules:
1. Answer using only the supplied evidence.
2. Do not invent facts or use unsupported outside knowledge.
3. If the evidence does not answer the question, clearly say:
   "I could not find sufficient evidence in the provided document."
4. Keep answers clear, concise, and relevant.
5. Cite supporting sources using the source filename and page number
   provided in the evidence.
6. Distinguish explicit facts from reasonable interpretations.
7. Never fabricate citations.
"""

def build_user_prompt(question: str, context: str) -> str:
    """Build a prompt containing the question and retrieved evidence."""

    return f"""
Evidence from the uploaded document:

<context>
{context}
</context>

Question:
{question}

Answer using the evidence above. Include source citations.
If the evidence is insufficient, say so explicitly.
"""
