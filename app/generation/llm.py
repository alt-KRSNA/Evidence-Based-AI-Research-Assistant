import os

from dotenv import load_dotenv
from groq import Groq


class GroqClient:
    """Handles communication with the Groq API."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Generate a response using Groq."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("Groq returned an empty response.")

        return answer.strip()