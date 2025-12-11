"""Chat component for the Gradio UI."""
from __future__ import annotations

import importlib.util
from typing import Callable

from hayagriva.exceptions import MissingDependencyError


def chat_box(on_submit: Callable[[str], str]):
    if importlib.util.find_spec("gradio") is None:
        raise MissingDependencyError("gradio is required for the UI. Install with `pip install gradio`.")
    import gradio as gr

    return gr.ChatInterface(on_submit)
