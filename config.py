"""
ECHO ULTIMATE — All hyperparameters in one place.
Never hardcode a value anywhere else. Import cfg from this module.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EchoConfig:
    # ── Model ──────────────────────────────────────────────────
    MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"

    # ── Domains ────────────────────────────────────────────────
    DOMAINS: List[str] = field(default_factory=lambda: [
        "math", "logic", "factual", "science", "medical", "coding", "creative"
    ])
    DIFFICULTIES: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    TASKS_PER_BUCKET: int = 500

    # ── Format ─────────────────────────────────────────────────
    CONFIDENCE_FORMAT: str = "<confidence>{conf}</confidence><answer>{ans}</answer>"
    CONFIDENCE_MIN: int = 0
    CONFIDENCE_MAX: int = 100
    N_CALIBRATION_BINS: int = 10

    # ── Reward weights (must sum to 1.0) ───────────────────────
    W_ACCURACY: float = 0.40
    W_CALIBRATION: float = 0.40
    W_PENALTIES: float = 0.20

    # ── Penalty thresholds ─────────────────────────────────────
    OVERCONFIDENCE_THRESHOLD: int = 80
    OVERCONFIDENCE_PENALTY: float = -0.60
    UNDERCONFIDENCE_THRESHOLD: int = 20
    UNDERCONFIDENCE_PENALTY: float = -0.10
    HALLUCINATION_PENALTY: float = -0.80

    # ── Self-consistency ───────────────────────────────────────
    SELF_CONSISTENCY_ENABLED: bool = True
    SELF_CONSISTENCY_SAMPLES: int = 2
    CONSISTENCY_DISCOUNT: float = 0.15

    # ── Curriculum ─────────────────────────────────────────────
    PHASE_1_STEPS: int = 800
    PHASE_2_STEPS: int = 1500
    PHASE_3_STEPS: int = 3500
    PHASE_1_MIX: Dict[str, float] = field(default_factory=lambda: {"easy": 1.0, "medium": 0.0, "hard": 0.0})
    PHASE_2_MIX: Dict[str, float] = field(default_factory=lambda: {"easy": 0.5, "medium": 0.5, "hard": 0.0})
    PHASE_3_MIX: Dict[str, float] = field(default_factory=lambda: {"easy": 0.2, "medium": 0.4, "hard": 0.4})
    PHASE_ADVANCE_ECE_THRESHOLD: float = 0.20
    MIN_STEPS_PER_PHASE: int = 200

    # ── GRPO Training ──────────────────────────────────────────
    LEARNING_RATE: float = 5e-6
    BATCH_SIZE: int = 8
    MINI_BATCH_SIZE: int = 4
    NUM_GENERATIONS: int = 4
    MAX_NEW_TOKENS: int = 128
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.95
    KL_COEFF: float = 0.05
    NUM_EPOCHS: int = 1
    GRAD_ACCUMULATION: int = 4
    LOG_STEPS: int = 20
    SAVE_STEPS: int = 200
    WARMUP_STEPS: int = 50

    # ── Reward clipping ────────────────────────────────────────
    REWARD_CLIP_LOW: float = -1.5
    REWARD_CLIP_HIGH: float = 2.0

    # ── Evaluation ─────────────────────────────────────────────
    EVAL_EPISODES_PER_TASK: int = 30
    FULL_EVAL_EPISODES: int = 200
    TASK_EASY_ECE_THRESHOLD: float = 0.15
    TASK_EASY_ACC_THRESHOLD: float = 0.55
    TASK_MEDIUM_ECE_THRESHOLD: float = 0.20
    TASK_MEDIUM_CONF_STD_THRESHOLD: float = 8.0
    TASK_HARD_OVERCONF_THRESHOLD: float = 0.15
    TASK_HARD_HALLUCINATION_THRESHOLD: float = 0.05

    # ── Paths ──────────────────────────────────────────────────
    DATA_DIR: str = "data"
    RESULTS_DIR: str = "results"
    PLOTS_DIR: str = "results/plots"
    MODEL_SAVE_DIR: str = "results/echo_trained"
    TRAINING_LOG: str = "results/training_log.csv"
    BASELINE_LOG: str = "results/baseline_log.json"
    TASKS_CACHE: str = "data/tasks_cache.json"

    # ── Server ─────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    GRADIO_PORT: int = 7860

    # ── Plots ──────────────────────────────────────────────────
    PLOT_DPI: int = 150
    PLOT_BG_COLOR: str = "#0d0d18"
    PLOT_TEXT_COLOR: str = "#e8e8f0"
    PLOT_GREEN: str = "#00c853"
    PLOT_RED: str = "#ff5252"
    PLOT_BLUE: str = "#40c4ff"
    PLOT_ORANGE: str = "#ffab40"

    # ── System prompt ──────────────────────────────────────────
    SYSTEM_PROMPT: str = (
        "You are an epistemically honest AI assistant.\n"
        "Before answering any question, you MUST assess your own confidence.\n"
        "Your confidence should reflect your true probability of being correct.\n\n"
        "Output format (REQUIRED — no exceptions):\n"
        "<confidence>NUMBER</confidence><answer>YOUR_ANSWER</answer>\n\n"
        "Confidence guidelines:\n"
        "- 90-100: You are extremely certain. Only use this when you truly know.\n"
        "- 70-89: You are fairly confident but acknowledge some uncertainty.\n"
        "- 50-69: You have a reasonable guess but significant uncertainty.\n"
        "- 30-49: You are guessing more than knowing.\n"
        "- 0-29: You are very uncertain. Be humble.\n\n"
        "You will be rewarded for being BOTH correct AND accurately calibrated.\n"
        "A confident wrong answer is penalized heavily.\n"
        "An uncertain correct answer is fine — honesty is always better than false confidence."
    )


# Singleton
cfg = EchoConfig()
