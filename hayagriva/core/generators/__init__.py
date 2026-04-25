"""LLM generator backends."""

from typing import TYPE_CHECKING

__all__ = ["GroqGenerator"]

if TYPE_CHECKING:
    from hayagriva.core.generators.groq import GroqGenerator


def __getattr__(name: str):
    """Lazily expose generator classes without importing provider SDKs early.

    Args:
        name: Attribute name requested from this module.

    Returns:
        Requested generator class.

    Raises:
        AttributeError: If ``name`` is not exported by this module.
    """
    if name == "GroqGenerator":
        from hayagriva.core.generators.groq import GroqGenerator

        return GroqGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
