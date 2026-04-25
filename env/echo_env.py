"""
ECHO ULTIMATE — Main Gymnasium Environment.

Each episode = 1 question → 1 answer → 1 reward.
State includes running calibration metrics across all 7 domains.
"""

import logging
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import cfg
from env.parser import parse_response, format_prompt, ParseResult
from env.reward import compute_reward, RewardHistory, RewardBreakdown
from env.task_bank import TaskBank

logger = logging.getLogger(__name__)

_DOMAIN_INDEX = {d: i for i, d in enumerate(cfg.DOMAINS)}


class EchoEnv(gym.Env):
    """
    ECHO ULTIMATE Gymnasium environment.

    Observation: dict with task info + running calibration metrics.
    Action:      text string in <confidence>N</confidence><answer>X</answer> format.
    Reward:      weighted accuracy + Brier calibration + overconfidence penalties.

    Each episode terminates after exactly one step.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_bank: Optional[TaskBank] = None,
        reward_history: Optional[RewardHistory] = None,
        phase: int = 1,
        self_consistency: bool = False,
        generate_fn: Optional[Callable[[str], str]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.task_bank      = task_bank or TaskBank()
        self.task_bank.ensure_loaded()
        self.reward_history = reward_history or RewardHistory()
        self.phase          = phase
        self.self_consistency = self_consistency
        self.generate_fn    = generate_fn
        self.render_mode    = render_mode

        self._current_task: Optional[dict]       = None
        self._last_result:  Optional[RewardBreakdown] = None
        self._last_parsed:  Optional[ParseResult]     = None
        self._episode_step: int   = 0
        self._episode_reward: float = 0.0

        # Gymnasium spaces (informational for text-based env)
        self.action_space = spaces.Text(min_length=1, max_length=1024)
        self.observation_space = spaces.Dict({
            "task_id":              spaces.Text(min_length=1, max_length=128),
            "domain":               spaces.Text(min_length=1, max_length=32),
            "difficulty":           spaces.Text(min_length=1, max_length=16),
            "question":             spaces.Text(min_length=1, max_length=4096),
            "phase":                spaces.Discrete(4),
            "episode_step":         spaces.Discrete(3),
            "running_ece":          spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "running_accuracy":     spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "running_mean_confidence": spaces.Box(0, 100, shape=(1,), dtype=np.float32),
            "domain_ece":           spaces.Box(0, 1, shape=(len(cfg.DOMAINS),), dtype=np.float32),
        })

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        task_id = (options or {}).get("task_id")
        if task_id:
            task = self.task_bank.get_task_by_id(task_id) or \
                   self.task_bank.get_batch(1, self.phase)[0]
        elif (options or {}).get("adversarial"):
            task = self.task_bank.get_adversarial_batch(1)[0]
        else:
            task = self.task_bank.get_batch(1, self.phase)[0]

        self._current_task    = task
        self._episode_step    = 0
        self._episode_reward  = 0.0
        self._last_result     = None
        self._last_parsed     = None

        prompt = format_prompt(
            task["question"], task["domain"], task["difficulty"],
            show_difficulty=(self.phase == 1),
        )
        obs  = self._build_obs()
        info = {"task": task, "formatted_prompt": prompt}
        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        if self._current_task is None:
            logger.warning("step() called before reset() — auto-resetting")
            self.reset()

        task = self._current_task

        # Self-consistency check (demo mode only)
        if self.self_consistency and self.generate_fn is not None:
            from env.self_consistency import SelfConsistencyChecker
            checker = SelfConsistencyChecker()
            prompt  = format_prompt(task["question"], task["domain"], task["difficulty"])
            result  = checker.check(prompt, self.generate_fn)
            # Override confidence from consistency check
            action = cfg.CONFIDENCE_FORMAT.format(
                conf=result.final_confidence, ans=result.final_answer
            )

        parsed = parse_response(action)
        rb     = compute_reward(
            confidence=parsed.confidence,
            predicted=parsed.answer,
            ground_truth=task["answer"],
            aliases=task.get("answer_aliases", []),
            domain=task["domain"],
        )

        self.reward_history.append(
            confidence=parsed.confidence,
            was_correct=rb.was_correct,
            domain=task["domain"],
            difficulty=task["difficulty"],
            reward=rb.total,
            is_abstention=parsed.is_abstention,
        )

        self._last_result  = rb
        self._last_parsed  = parsed
        self._episode_step = 1
        self._episode_reward = rb.total

        obs = self._build_obs()
        info = {
            "accuracy":                  rb.accuracy_score,
            "brier_reward":              rb.brier_reward_val,
            "overconfidence_penalty":    rb.overconfidence_penalty_val,
            "underconfidence_penalty":   rb.underconfidence_penalty_val,
            "parsed_confidence":         parsed.confidence,
            "parsed_answer":             parsed.answer,
            "true_answer":               task["answer"],
            "was_correct":               rb.was_correct,
            "parse_success":             parsed.parse_success,
            "is_abstention":             parsed.is_abstention,
            "task_id":                   task["id"],
            "domain":                    task["domain"],
            "difficulty":                task["difficulty"],
            "breakdown":                 rb.breakdown_str,
        }

        if self.render_mode == "human":
            self.render()

        return obs, rb.total, True, False, info   # terminated=True (single step)

    def render(self) -> None:
        if self._current_task is None:
            print("[EchoEnv] No active episode.")
            return
        task = self._current_task
        rb   = self._last_result
        p    = self._last_parsed
        snap = self.reward_history.get_training_snapshot(last_n=100)

        icon = "✅" if (rb and rb.was_correct) else "❌"
        conf = p.confidence if p else "—"
        ans  = p.answer[:40] if p else "—"
        rew  = f"{rb.total:+.3f}" if rb else "—"
        ece  = f"{snap['ece']:.3f}"

        print(f"\n┌{'─'*37}┐")
        print(f"│ {'ECHO Episode Summary':<35} │")
        print(f"├{'─'*37}┤")
        print(f"│ {'Domain:':<12} {task['domain']} ({task['difficulty']}){'':<10}│"[:40])
        print(f"│ {'Q:':<5} {task['question'][:30]+'…':<32} │")
        print(f"│ {'Confidence:':<12} {conf}%{'':<22}│"[:40])
        print(f"│ {'Answer:':<12} {ans:<25} │"[:40])
        print(f"│ {'Correct:':<12} {icon:<25} │"[:40])
        print(f"│ {'Reward:':<12} {rew:<25} │"[:40])
        print(f"│ {'ECE (100ep):':<12} {ece:<25} │"[:40])
        print(f"└{'─'*37}┘")

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def get_metrics(self, domain: Optional[str] = None):
        return self.reward_history.get_calibration_report(domain=domain)

    def set_phase(self, phase: int) -> None:
        self.phase = max(1, min(3, phase))

    def get_formatted_prompt(self) -> str:
        if self._current_task is None:
            return ""
        t = self._current_task
        return format_prompt(t["question"], t["domain"], t["difficulty"],
                             show_difficulty=(self.phase == 1))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        task = self._current_task or {}
        snap = self.reward_history.get_training_snapshot(last_n=100)
        profiles = self.reward_history.get_domain_profiles()
        domain_ece = np.array(
            [profiles.get(d).ece if profiles.get(d) and profiles[d].n_samples > 0 else 0.5
             for d in cfg.DOMAINS],
            dtype=np.float32,
        )
        return {
            "task_id":               task.get("id", ""),
            "domain":                task.get("domain", ""),
            "difficulty":            task.get("difficulty", ""),
            "question":              task.get("question", ""),
            "phase":                 self.phase,
            "episode_step":          self._episode_step,
            "running_ece":           float(snap["ece"]),
            "running_accuracy":      float(snap["accuracy"]),
            "running_mean_confidence": float(snap["mean_confidence"]),
            "domain_ece":            [float(x) for x in domain_ece],
        }
