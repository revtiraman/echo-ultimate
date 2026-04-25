"""Stable ASGI entrypoint for Hugging Face Docker Space."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import app

