"""Backward-compatible import path for Groq generation.

Prefer importing ``GroqGenerator`` from ``hayagriva.core.generators`` in new
code. This module remains so older imports continue to work.
"""

from hayagriva.core.generators.groq import GroqGenerator

__all__ = ["GroqGenerator"]
