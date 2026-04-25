from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EchoAction:
    """Action: model's response with embedded confidence and answer."""

    response: str  # Full response text containing <confidence> and <answer> tags


@dataclass
class EchoObservation:
    """Observation returned after each step."""

    question: str
    domain: str
    difficulty: str
    reward: float
    accuracy: float
    confidence: int
    brier_score: float
    ece: float
    is_correct: bool
    thinking: str = ""
    feedback: str = ""
    episode_step: int = 0
    total_steps: int = 0


@dataclass
class EchoState:
    """Full environment state."""

    current_question: str = ""
    domain: str = ""
    difficulty: str = ""
    phase: int = 1
    step_count: int = 0
    total_reward: float = 0.0
    accuracy_history: list = field(default_factory=list)
    confidence_history: list = field(default_factory=list)
    ece_history: list = field(default_factory=list)
    domain_stats: Dict[str, Any] = field(default_factory=dict)
