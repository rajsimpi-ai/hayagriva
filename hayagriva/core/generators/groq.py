"""Groq generator backend."""

from groq import Groq

from hayagriva.exceptions import GenerationError


class GroqGenerator:
    """Generator backend using Groq chat completions.

    Args:
        api_key: Groq API key.
        model: Groq chat model name.

    Attributes:
        client: Groq SDK client.
        model: Model name used for generation.

    Raises:
        GenerationError: If ``api_key`` is missing.
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """Create a Groq chat-completion generator.

        Args:
            api_key: Groq API key.
            model: Groq model name.

        Raises:
            GenerationError: If ``api_key`` is missing.
        """
        if not api_key:
            raise GenerationError("Groq API key is missing.")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate an answer for a completed prompt.

        Args:
            prompt: Full prompt containing instructions, context, and question.

        Returns:
            Generated assistant text.

        Raises:
            GenerationError: If the Groq API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as exc:
            raise GenerationError(f"Groq generation failed: {exc}") from exc
