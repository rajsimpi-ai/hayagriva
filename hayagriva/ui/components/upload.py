"""Upload component for the Gradio UI."""
from __future__ import annotations

import importlib.util
from typing import Callable

from hayagriva.exceptions import MissingDependencyError


def uploader(on_upload: Callable[[str], str]):
    if importlib.util.find_spec("gradio") is None:
        raise MissingDependencyError("gradio is required for the UI. Install with `pip install gradio`.")
    import gradio as gr

    with gr.Blocks() as block:
        upload_box = gr.Textbox(label="Paste text to ingest")
        status = gr.Markdown("Ready to ingest")

        def _upload(text: str):
            result = on_upload(text)
            return result

        upload_box.submit(_upload, inputs=upload_box, outputs=status)
    return block
