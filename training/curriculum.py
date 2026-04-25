"""
ECHO ULTIMATE — 3-Phase Curriculum Manager.
Phase advances when ECE < PHASE_ADVANCE_ECE_THRESHOLD.
"""

import logging
from config import cfg

logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Tracks training step count and manages curriculum phase transitions.
    Phases: 1 (easy only) → 2 (easy+medium) → 3 (all + adversarial).
    Never goes backward.
    """

    def __init__(self) -> None:
        self.current_phase = 1
        self.phase_history: list[tuple] = []   # (step, phase, ece)
        self._steps_in_phase = 0
        self._last_step = 0

    def should_advance(self, current_ece: float, current_step: int) -> bool:
        steps_since = current_step - self._last_step
        if self.current_phase >= 3:
            return False
        min_steps = cfg.MIN_STEPS_PER_PHASE
        ece_ok    = current_ece < cfg.PHASE_ADVANCE_ECE_THRESHOLD

        # Also force advance at scheduled boundaries
        phase_boundaries = [cfg.PHASE_1_STEPS, cfg.PHASE_1_STEPS + cfg.PHASE_2_STEPS]
        forced = current_step >= phase_boundaries[self.current_phase - 1]

        return (ece_ok and steps_since >= min_steps) or forced

    def advance_phase(self, step: int = 0, ece: float = 0.0) -> None:
        old = self.current_phase
        self.current_phase = min(3, self.current_phase + 1)
        self.phase_history.append((step, self.current_phase, ece))
        self._last_step = step
        self._steps_in_phase = 0
        logger.info(
            "🎓 Phase %d → %d at step %d (ECE=%.3f)", old, self.current_phase, step, ece
        )
        print(f"\n🎓 Phase {old} → {self.current_phase} at step {step} (ECE={ece:.3f})")

    def update(self, step: int, current_ece: float) -> bool:
        """Update state. Returns True if phase was advanced."""
        self._steps_in_phase += 1
        if self.should_advance(current_ece, step):
            self.advance_phase(step, current_ece)
            return True
        return False

    def get_current_mix(self) -> dict:
        mixes = [cfg.PHASE_1_MIX, cfg.PHASE_2_MIX, cfg.PHASE_3_MIX]
        return mixes[self.current_phase - 1]

    def get_phase_description(self) -> str:
        return {
            1: "Phase 1 — Easy tasks, difficulty labels shown — learning basic calibration",
            2: "Phase 2 — Easy+Medium, no difficulty labels — generalizing calibration",
            3: "Phase 3 — All difficulties, adversarial examples — mastering uncertainty",
        }[self.current_phase]

    def summary(self) -> dict:
        return {
            "current_phase": self.current_phase,
            "phase_history": self.phase_history,
            "description": self.get_phase_description(),
            "mix": self.get_current_mix(),
        }
