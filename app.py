"""HuggingFace Space entry point."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import build_app, _CSS, _JS

demo, theme = build_app()
demo.queue()
demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    css=_CSS,
    js=_JS,
    theme=theme,
)
