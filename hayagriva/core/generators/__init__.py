"""LLM generator backends."""

from typing import TYPE_CHECKING

__all__ = ["GroqGenerator"]

if TYPE_CHECKING:
    from hayagriva.core.generators.groq import GroqGenerator


def __getattr__(name: str):
    if name == "GroqGenerator":
        from hayagriva.core.generators.groq import GroqGenerator

        return GroqGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
