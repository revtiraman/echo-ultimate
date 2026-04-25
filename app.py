"""HuggingFace Space entry point — forwards to FastAPI+Gradio server."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This file is kept for compatibility.
# The actual app is in server/app.py and launched via Dockerfile CMD:
#   python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# All endpoints:
#   /health  /tasks  /reset  /step  /state  /metrics  /fingerprint  /history  /docs  /ui

from server.app import app  # noqa: F401 — imported so this module is a valid ASGI target
