"""
ECHO ULTIMATE — FastAPI OpenEnv-Compliant Server.
Pure FastAPI: no openenv package dependency.
Mounts Gradio UI at /ui.
Runs on port 7860 (HuggingFace Space public port).
"""

import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import cfg
from core.tasks import TASKS
from env.echo_env import EchoEnv
from env.reward import RewardHistory
from env.task_bank import TaskBank

logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────

_task_bank: Optional[TaskBank] = None
_env: Optional[EchoEnv] = None
_history: Optional[RewardHistory] = None


def _get_env() -> EchoEnv:
    if _env is None:
        raise HTTPException(400, "No active episode. POST /reset first.")
    return _env


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Specific task ID to load")
    adversarial: Optional[bool] = Field(False, description="Use adversarial questions")


class StepRequest(BaseModel):
    action: Optional[str] = Field(None, description="Legacy: action string")
    response: Optional[str] = Field(None, description="Agent response with confidence and answer tags")

    def get_response(self) -> str:
        """Accept either 'response' or 'action' field."""
        return self.response or self.action or ""


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    pass_threshold: float
    n_episodes: int


class StepResponse(BaseModel):
    state: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_bank, _env, _history
    logger.info("ECHO ULTIMATE server starting…")
    _task_bank = TaskBank()
    _task_bank.ensure_loaded()
    _history = RewardHistory()
    _env = EchoEnv(task_bank=_task_bank, reward_history=_history, phase=3)
    _env.reset()
    logger.info("ECHO ULTIMATE ready ✅  (7 domains, 3 tasks)")
    print("✅  ECHO ULTIMATE server ready — http://0.0.0.0:7860/docs")
    yield
    logger.info("ECHO ULTIMATE server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ECHO ULTIMATE — Epistemic Calibration RL Environment",
    description=(
        "OpenEnv-compliant training environment for LLM metacognitive calibration. "
        "7 domains · 3 curriculum phases · 5 calibration metrics · Epistemic fingerprint."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "environment": "ECHO-ULTIMATE", "version": "2.0.0",
            "domains": 7, "tasks": 3}


@app.get("/", tags=["Health"])
async def root():
    return {"message": "ECHO ULTIMATE RL Environment",
            "docs": "/docs", "health": "/health",
            "tasks": "/tasks", "metrics": "/metrics", "ui": "/ui"}


@app.get("/tasks", response_model=list[TaskInfo], tags=["Tasks"])
async def list_tasks():
    return [TaskInfo(id=t.id, name=t.name, description=t.description,
                     pass_threshold=t.pass_threshold, n_episodes=t.n_episodes)
            for t in TASKS]


@app.post("/reset", tags=["Environment"])
async def reset(req: ResetRequest = ResetRequest()) -> dict:
    env = _get_env()
    opts = {}
    if req.task_id:
        opts["task_id"] = req.task_id
    if req.adversarial:
        opts["adversarial"] = True
    state, info = env.reset(options=opts if opts else None)
    return state


@app.post("/reset/{task_id}", tags=["Environment"])
async def reset_task(task_id: str) -> dict:
    env = _get_env()
    state, _ = env.reset(options={"task_id": task_id})
    return state


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(req: StepRequest) -> StepResponse:
    env = _get_env()
    response_text = req.get_response()
    if not response_text:
        raise HTTPException(422, "Provide either 'response' or 'action' field.")
    try:
        state, reward, terminated, truncated, info = env.step(response_text)
    except Exception as exc:
        logger.error("step error: %s", exc)
        raise HTTPException(500, f"Step failed: {exc}")
    return StepResponse(
        state=state,
        reward=round(float(reward), 4),
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state", tags=["Environment"])
async def get_state() -> dict:
    return _get_env()._build_obs()


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    rep = _get_env().get_metrics()
    return rep.to_dict()


@app.get("/metrics/{domain}", tags=["Metrics"])
async def get_domain_metrics(domain: str):
    if domain not in cfg.DOMAINS:
        raise HTTPException(404, f"Unknown domain '{domain}'. Valid: {cfg.DOMAINS}")
    rep = _get_env().get_metrics(domain=domain)
    return rep.to_dict()


@app.get("/fingerprint", tags=["Metrics"])
async def get_fingerprint() -> dict:
    env = _get_env()
    profiles = env.reward_history.get_domain_profiles()
    return {
        "domain_scores": {d: round(1.0 - r.ece, 3) for d, r in profiles.items()},
        "domain_ece": {d: round(r.ece, 3) for d, r in profiles.items()},
        "domain_accuracy": {d: round(r.accuracy, 3) for d, r in profiles.items()},
        "overall_ece": round(env.get_metrics().ece, 3),
    }


@app.get("/history", tags=["Metrics"])
async def get_history() -> dict:
    env = _get_env()
    df = env.reward_history.to_dataframe()
    records = df.tail(100).to_dict(orient="records") if len(df) > 0 else []
    return {"episodes": records, "total": len(df)}


@app.post("/advance_phase", tags=["Environment"])
async def advance_phase():
    env = _get_env()
    env.phase = min(getattr(env, "phase", 1) + 1, 4)
    return {"phase": env.phase, "message": f"Advanced to Phase {env.phase}"}


# ── Mount Gradio UI at /ui ────────────────────────────────────────────────────

try:
    import gradio as gr
    import importlib.util

    _ui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "app.py"
    )
    spec = importlib.util.spec_from_file_location("gradio_app", _ui_path)
    gradio_module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(gradio_module)
        if hasattr(gradio_module, "demo"):
            _gradio_demo = gradio_module.demo
        elif hasattr(gradio_module, "build_app"):
            _gradio_demo, _ = gradio_module.build_app()
        else:
            raise AttributeError("ui/app.py has neither 'demo' nor 'build_app'")
        app = gr.mount_gradio_app(app, _gradio_demo, path="/ui")
        print("✅  Gradio UI mounted at /ui")
    else:
        print("⚠️  Could not load ui/app.py spec")
except Exception as _e:
    print(f"⚠️  Gradio UI not mounted: {_e}")


# ── Direct runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
