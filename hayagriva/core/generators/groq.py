"""Groq generator backend."""

from groq import Groq

from hayagriva.exceptions import GenerationError


class GroqGenerator:
    """Generator backend using Groq chat completions."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        if not api_key:
            raise GenerationError("Groq API key is missing.")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as exc:
            raise GenerationError(f"Groq generation failed: {exc}") from exc
