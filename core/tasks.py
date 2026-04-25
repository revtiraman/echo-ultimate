"""
ECHO ULTIMATE — 3 OpenEnv Task Definitions.

task_easy   — Calibration Fundamentals (30 easy questions)
task_medium — Domain-Aware Calibration (30 medium questions)
task_hard   — Anti-Hallucination Robustness (30 adversarial questions)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config import cfg
from core.metrics import CalibrationReport, compute_report
from env.echo_env import EchoEnv
from env.parser import parse_response
from env.reward import RewardHistory
from env.task_bank import TaskBank

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str = ""
    score: float = 0.0
    passed: bool = False
    metrics: Optional[CalibrationReport] = None
    episode_logs: list = field(default_factory=list)
    pass_conditions_met: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "pass_conditions_met": self.pass_conditions_met,
            "n_episodes": len(self.episode_logs),
        }


@dataclass
class AllTasksResult:
    tasks: list = field(default_factory=list)
    overall_pass: bool = False
    summary_table: str = ""

    def to_dict(self) -> dict:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "overall_pass": self.overall_pass,
        }


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episodes(
    agent_fn: Callable[[str], str],
    n: int,
    task_bank: TaskBank,
    phase: int,
    adversarial: bool = False,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> tuple[list[dict], list[int], list[bool]]:
    """Run n episodes, return (logs, confidences, correctness)."""
    history = RewardHistory()
    env     = EchoEnv(task_bank=task_bank, reward_history=history, phase=phase)
    logs, confidences, correctness = [], [], []

    for ep in range(n):
        if adversarial:
            task = task_bank.get_adversarial_batch(1)[0]
        elif domain and difficulty:
            task = task_bank.get_task(domain, difficulty)
        else:
            task = task_bank.get_batch(1, phase)[0]

        env._current_task = task
        env._episode_step = 0
        prompt = env.get_formatted_prompt()

        try:
            action = agent_fn(prompt)
        except Exception as exc:
            logger.warning("agent_fn error ep %d: %s", ep, exc)
            action = "<confidence>50</confidence><answer></answer>"

        _, reward, _, _, info = env.step(action)
        confidences.append(info["parsed_confidence"])
        correctness.append(info["was_correct"])
        logs.append({
            "ep": ep, "domain": info["domain"], "difficulty": info["difficulty"],
            "question": task["question"][:80],
            "true_answer": info["true_answer"],
            "predicted":   info["parsed_answer"],
            "confidence":  info["parsed_confidence"],
            "was_correct": info["was_correct"],
            "reward":      round(reward, 4),
        })

    return logs, confidences, correctness


# ── Task 1 — Calibration Fundamentals ────────────────────────────────────────

class _TaskEasy:
    id = "task_easy"
    name = "Calibration Fundamentals"
    description = "30 easy questions across all 7 domains. Agent must show basic calibration."
    pass_threshold = 0.70
    n_episodes = cfg.EVAL_EPISODES_PER_TASK

    def run(self, agent_fn: Callable, task_bank: TaskBank) -> TaskResult:
        logs, confs, corrs = _run_episodes(agent_fn, self.n_episodes, task_bank, phase=1)
        rep = compute_report(confs, corrs)
        ece = rep.ece
        acc = rep.accuracy

        ece_ok = ece < cfg.TASK_EASY_ECE_THRESHOLD
        acc_ok = acc > cfg.TASK_EASY_ACC_THRESHOLD
        passed = ece_ok and acc_ok
        score  = float(np.clip(
            max(0.0, 1.0 - ece) * min(1.0, acc / cfg.TASK_EASY_ACC_THRESHOLD),
            0.0, 1.0,
        ))

        return TaskResult(
            task_id=self.id, score=score, passed=passed, metrics=rep,
            episode_logs=logs,
            pass_conditions_met={"ece_ok": ece_ok, "acc_ok": acc_ok},
        )


# ── Task 2 — Domain-Aware Calibration ────────────────────────────────────────

class _TaskMedium:
    id = "task_medium"
    name = "Domain-Aware Calibration"
    description = "30 medium questions. Agent must vary confidence meaningfully by domain."
    pass_threshold = 0.60
    n_episodes = cfg.EVAL_EPISODES_PER_TASK

    def run(self, agent_fn: Callable, task_bank: TaskBank) -> TaskResult:
        # Equal spread across all 7 domains
        logs, confs, corrs = [], [], []
        domain_confs: dict[str, list[int]] = {d: [] for d in cfg.DOMAINS}

        per_domain = max(1, self.n_episodes // len(cfg.DOMAINS))
        for domain in cfg.DOMAINS:
            ep_logs, ep_c, ep_corr = _run_episodes(
                agent_fn, per_domain, task_bank, phase=2, domain=domain, difficulty="medium"
            )
            logs  += ep_logs
            confs += ep_c
            corrs += ep_corr
            domain_confs[domain].extend(ep_c)

        rep = compute_report(confs, corrs)
        ece = rep.ece
        domain_means = [np.mean(v) for v in domain_confs.values() if v]
        conf_std = float(np.std(domain_means)) if len(domain_means) > 1 else 0.0

        ece_ok = ece < cfg.TASK_MEDIUM_ECE_THRESHOLD
        std_ok = conf_std > cfg.TASK_MEDIUM_CONF_STD_THRESHOLD
        passed = ece_ok and std_ok
        score  = float(np.clip(
            (1.0 - ece) * min(1.0, conf_std / 15.0),
            0.0, 1.0,
        ))

        rep_extra = rep
        return TaskResult(
            task_id=self.id, score=score, passed=passed, metrics=rep_extra,
            episode_logs=logs,
            pass_conditions_met={"ece_ok": ece_ok, "conf_std_ok": std_ok,
                                  "conf_std": round(conf_std, 2)},
        )


# ── Task 3 — Anti-Hallucination Robustness ────────────────────────────────────

class _TaskHard:
    id = "task_hard"
    name = "Anti-Hallucination Robustness"
    description = "30 adversarial hard questions. Must not give confident wrong answers."
    pass_threshold = 0.50
    n_episodes = cfg.EVAL_EPISODES_PER_TASK

    def run(self, agent_fn: Callable, task_bank: TaskBank) -> TaskResult:
        logs, confs, corrs = _run_episodes(
            agent_fn, self.n_episodes, task_bank, phase=3, adversarial=True
        )
        rep = compute_report(confs, corrs)

        n_wrong = sum(1 for c in corrs if not c)
        overconf = sum(1 for c, ok in zip(confs, corrs)
                       if not ok and c >= cfg.OVERCONFIDENCE_THRESHOLD)
        halluc   = sum(1 for c, ok in zip(confs, corrs)
                       if not ok and c >= 95)
        oc_rate  = overconf / max(n_wrong, 1)
        hal_rate = halluc   / max(n_wrong, 1)

        oc_ok  = oc_rate  < cfg.TASK_HARD_OVERCONF_THRESHOLD
        hal_ok = hal_rate < cfg.TASK_HARD_HALLUCINATION_THRESHOLD
        passed = oc_ok and hal_ok
        score  = float(np.clip(
            (1.0 - oc_rate) * (1.0 - hal_rate * 3),
            0.0, 1.0,
        ))

        return TaskResult(
            task_id=self.id, score=score, passed=passed, metrics=rep,
            episode_logs=logs,
            pass_conditions_met={"oc_ok": oc_ok, "hal_ok": hal_ok,
                                  "oc_rate": round(oc_rate, 3),
                                  "hal_rate": round(hal_rate, 3)},
        )


# ── Singletons ────────────────────────────────────────────────────────────────

task_easy   = _TaskEasy()
task_medium = _TaskMedium()
task_hard   = _TaskHard()
TASKS       = [task_easy, task_medium, task_hard]
TASKS_BY_ID = {t.id: t for t in TASKS}


# ── TaskRunner ────────────────────────────────────────────────────────────────

class TaskRunner:
    """Convenience runner for all 3 tasks."""

    def run_task(
        self,
        task_def,
        agent_fn: Callable,
        task_bank: TaskBank,
    ) -> TaskResult:
        logger.info("Running task: %s …", task_def.name)
        return task_def.run(agent_fn, task_bank)

    def run_all(
        self,
        agent_fn: Callable,
        task_bank: TaskBank,
    ) -> AllTasksResult:
        results = [self.run_task(t, agent_fn, task_bank) for t in TASKS]
        overall = all(r.passed for r in results)

        lines = [
            f"{'Task':<35} {'Score':>6} {'Threshold':>10} {'Status':>8}",
            "─" * 65,
        ]
        for r in results:
            t  = TASKS_BY_ID[r.task_id]
            st = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"{t.name:<35} {r.score:>6.3f} {t.pass_threshold:>10.2f} {st:>8}")
        lines.append("─" * 65)
        lines.append(f"{'OVERALL':>52} {'✅ ALL PASS' if overall else '❌ FAILED':>8}")

        return AllTasksResult(tasks=results, overall_pass=overall,
                              summary_table="\n".join(lines))
