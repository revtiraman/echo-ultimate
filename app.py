"""HuggingFace Space entry point — delegates to ui/app.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from ui.app import build_app

demo = build_app()
demo.queue()
demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    theme=gr.themes.Soft(),
    css=".gradio-container { background: #0d0d18 !important; }",
)
