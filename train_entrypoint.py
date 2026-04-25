"""
ECHO ULTIMATE — HuggingFace Space GPU Training Entrypoint.
Runs full GRPO training then pushes adapter to HF Hub.
Hardware: T4 medium or A10G small (set in Space settings).
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

import threading
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

# ── Tiny status server on :7860 so HF Space health checks pass ────────────────
status_app = FastAPI()
training_log = []

@status_app.get("/health")
def health():
    return {"status": "training", "log_lines": len(training_log)}

@status_app.get("/log", response_class=PlainTextResponse)
def log():
    return "\n".join(training_log[-100:])

def run_status_server():
    uvicorn.run(status_app, host="0.0.0.0", port=7860, log_level="warning")

threading.Thread(target=run_status_server, daemon=True).start()

# ── Training ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("🚀  ECHO ULTIMATE — GRPO Training on HF GPU Space")
print("=" * 60)

from config import cfg
from env.task_bank import TaskBank
from training.train import train

bank = TaskBank()
bank.download_all()

hf_token = os.environ.get("HF_TOKEN", "")
use_wandb = bool(os.environ.get("WANDB_API_KEY", ""))

train(
    model_name=cfg.MODEL_NAME,
    output_dir=cfg.MODEL_SAVE_DIR,
    task_bank=bank,
    use_wandb=use_wandb,
)

print("\n🎉  Training complete! Space will stay running — check /log for details.")
# Keep the status server alive after training
import time
while True:
    time.sleep(60)
