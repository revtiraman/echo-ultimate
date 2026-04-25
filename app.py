"""HuggingFace Space entry point — delegates to ui/app.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import build_app

import gradio as gr

demo = build_app()
demo.queue()
demo.launch(
    theme=gr.themes.Soft(),
    css=".gradio-container { background: #0d0d18 !important; }",
)
