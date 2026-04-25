import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import JSONResponse
from openenv.core.env_server import Environment, create_fastapi_app

from config import cfg
from env.parser import parse_response
from env.reward import compute_reward
from env.task_bank import TaskBank
from models import EchoAction, EchoObservation, EchoState


class EchoEnvironment(Environment):
    """
    ECHO: Epistemic Calibration via Hierarchical OpenEnv.

    Trains LLMs to accurately predict their own probability of being correct
    before answering. The model outputs:
    [<think>reasoning</think>] <confidence>0-100</confidence> <answer>text</answer>
    """

    def __init__(self):
        self.task_bank = TaskBank()
        self.task_bank.ensure_loaded()
        self._state = EchoState()
        self._current_task = None
        self._calibration_window = []  # sliding window for ECE
        self._phase = 1
        self._step_count = 0
        self.action_class = EchoAction
        self.observation_class = EchoObservation
        self.state_class = EchoState

    def reset(self) -> EchoObservation:
        """Sample a new question and return it as an observation."""
        self._step_count = 0
        # Pick difficulty based on phase
        if self._phase == 1:
            difficulties = ["easy"]
        elif self._phase == 2:
            difficulties = ["easy", "medium"]
        else:
            difficulties = ["easy", "medium", "hard"]

        difficulty = random.choice(difficulties)
        domain = random.choice(cfg.DOMAINS)
        self._current_task = self.task_bank.get_task(domain=domain, difficulty=difficulty)

        self._state = EchoState(
            current_question=self._current_task["question"],
            domain=self._current_task.get("domain", "general"),
            difficulty=difficulty,
            phase=self._phase,
        )

        return EchoObservation(
            question=self._current_task["question"],
            domain=self._current_task.get("domain", "general"),
            difficulty=difficulty,
            reward=0.0,
            accuracy=0.0,
            confidence=50,
            brier_score=0.25,
            ece=0.0,
            is_correct=False,
            feedback="New episode started. Predict your confidence and answer.",
            episode_step=0,
            total_steps=self._step_count,
        )

    def step(self, action: EchoAction) -> tuple[EchoObservation, float, bool, dict]:
        """
        Process model's response. Returns (observation, reward, done, info).
        """
        if self._current_task is None:
            self.reset()

        self._step_count += 1

        # Parse the model's response
        parse_result = parse_response(action.response)
        confidence = parse_result.confidence if parse_result.confidence is not None else 50

        # Compute reward
        reward_breakdown = compute_reward(
            confidence=confidence,
            predicted=parse_result.answer,
            ground_truth=self._current_task.get("answer", ""),
            aliases=self._current_task.get("answer_aliases", []),
            domain=self._current_task.get("domain", "general"),
        )
        reward = reward_breakdown.total
        is_correct = reward_breakdown.was_correct

        # Update calibration window
        self._calibration_window.append((confidence / 100.0, float(is_correct)))
        if len(self._calibration_window) > 100:
            self._calibration_window.pop(0)

        # Update state
        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.accuracy_history.append(float(is_correct))
        self._state.confidence_history.append(confidence)

        # Compute running ECE
        ece = self._compute_ece()
        self._state.ece_history.append(ece)

        # Build feedback
        status = "✓ Correct" if is_correct else "✗ Incorrect"
        feedback = (
            f"{status}. Confidence: {confidence}%. "
            f"Reward: {reward:.3f}. ECE: {ece:.3f}. "
            f"Brier: {(confidence/100 - float(is_correct))**2:.3f}"
        )

        obs = EchoObservation(
            question=self._current_task["question"],
            domain=self._current_task.get("domain", "general"),
            difficulty=self._state.difficulty,
            reward=reward,
            accuracy=float(is_correct),
            confidence=confidence,
            brier_score=(confidence / 100 - float(is_correct)) ** 2,
            ece=ece,
            is_correct=is_correct,
            thinking=getattr(parse_result, "thinking", ""),
            feedback=feedback,
            episode_step=self._step_count,
            total_steps=self._step_count,
        )

        done = False  # continuous environment
        info = {
            "accuracy_score": reward_breakdown.accuracy_score,
            "brier_reward": reward_breakdown.brier_reward_val,
            "overconfidence_penalty": reward_breakdown.overconfidence_penalty_val,
            "underconfidence_penalty": reward_breakdown.underconfidence_penalty_val,
            "reward": reward_breakdown.total,
            "was_correct": reward_breakdown.was_correct,
            "breakdown": reward_breakdown.breakdown_str,
            "ece": ece,
        }

        return obs, reward, done, info

    def state(self) -> EchoState:
        return self._state

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """Fuzzy answer matching."""
        if not predicted or not correct:
            return False
        pred = predicted.strip().lower()
        corr = correct.strip().lower()
        if pred == corr:
            return True
        if corr in pred or pred in corr:
            return True
        # Check multiple choice (A/B/C/D)
        if len(corr) == 1 and corr in "abcd":
            return pred.startswith(corr) or pred == corr
        return False

    def _compute_ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error from sliding window."""
        if len(self._calibration_window) < 5:
            return 0.0

        bins = [[] for _ in range(n_bins)]
        for conf, outcome in self._calibration_window:
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, outcome))

        ece = 0.0
        n = len(self._calibration_window)
        for b in bins:
            if b:
                avg_conf = sum(c for c, _ in b) / len(b)
                avg_acc = sum(o for _, o in b) / len(b)
                ece += (len(b) / n) * abs(avg_conf - avg_acc)
        return ece


# Create the OpenEnv app
env = EchoEnvironment()
app = create_fastapi_app(env)


# Add custom routes on top
@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "echo-openenv"}


@app.get("/metrics")
async def get_metrics():
    """Extended calibration metrics."""
    state = env.state()
    return JSONResponse(
        {
            "phase": state.phase,
            "step_count": state.step_count,
            "total_reward": state.total_reward,
            "ece_history": state.ece_history[-20:],
            "accuracy_history": state.accuracy_history[-20:],
            "confidence_history": state.confidence_history[-20:],
            "current_ece": state.ece_history[-1] if state.ece_history else 0.0,
            "current_accuracy": sum(state.accuracy_history[-20:])
            / max(len(state.accuracy_history[-20:]), 1),
        }
    )


@app.post("/advance_phase")
async def advance_phase():
    """Move to next curriculum phase."""
    env._phase = min(env._phase + 1, 4)
    env._state.phase = env._phase
    return {"phase": env._phase, "message": f"Advanced to Phase {env._phase}"}

# Mount Gradio UI inside FastAPI so both run on port 7860
try:
    import gradio as gr
    import importlib.util

    # Load the Gradio app from ui/app.py
    spec = importlib.util.spec_from_file_location(
        "gradio_app",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ui",
            "app.py",
        ),
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load ui/app.py")

    gradio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gradio_module)

    # Mount at /ui
    gradio_demo = gradio_module.demo  # assumes ui/app.py has: demo = gr.Blocks(...)
    app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
    print("Gradio UI mounted at /ui")
except Exception as e:
    print(f"Gradio UI not mounted: {e}")

# After mounting, the endpoints are:
# https://vikaspandey582003-echo-ultimate.hf.space/health  (API)
# https://vikaspandey582003-echo-ultimate.hf.space/reset   (API)
# https://vikaspandey582003-echo-ultimate.hf.space/step    (API)
# https://vikaspandey582003-echo-ultimate.hf.space/ui      (Gradio)
# https://vikaspandey582003-echo-ultimate.hf.space/docs    (Swagger)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
