"""HuggingFace Space entry point — delegates to ui/app.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import build_app

demo = build_app()
demo.queue()
demo.launch()
