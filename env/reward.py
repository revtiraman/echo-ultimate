"""
ECHO ULTIMATE — All reward components.

Brier score formula: BS = (p - o)^2  where p = conf/100, o = 1 if correct
brier_reward = 1 - 2*BS  →  range [-1, 1]

Verification:
  conf=100, correct  → BS=0   → reward=+1.0 ✅
  conf=0,   wrong    → BS=0   → reward=+1.0 ✅
  conf=100, wrong    → BS=1   → reward=-1.0 ✅
  conf=50,  either   → BS=0.25 → reward=+0.5 ✅
"""

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg
from core.metrics import CalibrationReport, compute_report

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ── Number parsing ────────────────────────────────────────────────────────────

def _parse_num(text: str) -> Optional[float]:
    """Extract first number from text, handling commas and currency symbols."""
    if not text:
        return None
    cleaned = re.sub(r"[$€£¥,]", "", str(text))
    m = _NUM_RE.search(cleaned)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    return None


def _norm_choice(text: str) -> str:
    """Normalize a multiple-choice letter: '(A)', 'A.', 'A)' → 'A'."""
    if not text:
        return ""
    s = text.strip().upper()
    m = re.match(r"^\(?([A-Da-d])\)?\.?\s*", s)
    if m:
        return m.group(1).upper()
    return s[0] if s and s[0] in "ABCD" else s


def _fuzzy(a: str, b: str) -> float:
    """SequenceMatcher similarity ratio in [0, 1]."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── Accuracy reward ───────────────────────────────────────────────────────────

def accuracy_reward(
    predicted: str,
    ground_truth: str,
    answer_aliases: list[str],
    domain: str,
) -> float:
    """
    Domain-aware accuracy score in [0.0, 1.0].

    - math:    numeric tolerance (exact=1.0, ±1%=0.8, ±5%=0.5)
    - logic:   exact letter match after normalization
    - factual: alias list + substring matching
    - science/medical/coding/creative: fuzzy string matching
    """
    if not predicted:
        return 0.0

    try:
        if domain == "math":
            p = _parse_num(predicted)
            t = _parse_num(ground_truth)
            if p is None or t is None:
                return 0.0
            if p == t:
                return 1.0
            denom = abs(t) if t != 0 else 1.0
            rel = abs(p - t) / denom
            if rel <= 0.01:
                return 0.8
            if rel <= 0.05:
                return 0.5
            return 0.0

        elif domain == "logic":
            return 1.0 if _norm_choice(predicted) == _norm_choice(ground_truth) else 0.0

        elif domain in ("factual",):
            aliases = [ground_truth] + (answer_aliases or [])
            pred_low = predicted.strip().lower()
            for alias in aliases:
                if not alias:
                    continue
                al = alias.strip().lower()
                if pred_low == al:
                    return 1.0
            for alias in aliases:
                if not alias:
                    continue
                al = alias.strip().lower()
                if al in pred_low or pred_low in al:
                    return 0.5
            return 0.0

        elif domain in ("science", "medical"):
            # Multiple choice first
            pn = _norm_choice(predicted)
            tn = _norm_choice(ground_truth)
            if pn in "ABCD" and tn in "ABCD":
                return 1.0 if pn == tn else 0.0
            # Fuzzy fallback
            score = _fuzzy(predicted, ground_truth)
            if score > 0.85:
                return 1.0
            if score > 0.65:
                return 0.7
            if score > 0.45:
                return 0.4
            return 0.0

        elif domain in ("coding", "creative"):
            aliases = [ground_truth] + (answer_aliases or [])
            for alias in aliases:
                if not alias:
                    continue
                score = _fuzzy(predicted, alias)
                if score > 0.85:
                    return 1.0
                if score > 0.65:
                    return 0.7
                if score > 0.45:
                    return 0.4
            return 0.0

        else:
            return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0

    except Exception as exc:
        logger.warning("accuracy_reward error: %s", exc)
        return 0.0


# ── Brier reward ──────────────────────────────────────────────────────────────

def brier_reward(confidence: int, was_correct: bool) -> float:
    """
    Convert Brier score to reward signal in [-1, 1].

    BS = (p - o)^2   [0 = perfect, 1 = worst]
    reward = 1 - 2*BS
    """
    p = max(0.0, min(1.0, confidence / 100.0))
    o = 1.0 if was_correct else 0.0
    bs = (p - o) ** 2  # Brier score
    return float(1.0 - 2.0 * bs)


# ── Penalties ─────────────────────────────────────────────────────────────────

def overconfidence_penalty(confidence: int, was_correct: bool) -> float:
    """
    Graduated overconfidence penalty.
    conf >= 95 AND wrong → HALLUCINATION_PENALTY (-0.80)
    conf >= 80 AND wrong → OVERCONFIDENCE_PENALTY (-0.60)
    """
    if was_correct:
        return 0.0
    if confidence >= 95:
        return float(cfg.HALLUCINATION_PENALTY)
    if confidence >= cfg.OVERCONFIDENCE_THRESHOLD:
        return float(cfg.OVERCONFIDENCE_PENALTY)
    return 0.0


def underconfidence_penalty(confidence: int, was_correct: bool) -> float:
    """Small penalty for falsely humble correct answers."""
    if was_correct and confidence <= cfg.UNDERCONFIDENCE_THRESHOLD:
        return float(cfg.UNDERCONFIDENCE_PENALTY)
    return 0.0


# ── Combined reward ───────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """Full reward breakdown for one episode."""
    accuracy_score: float = 0.0
    brier_reward_val: float = 0.0
    overconfidence_penalty_val: float = 0.0
    underconfidence_penalty_val: float = 0.0
    total: float = 0.0
    was_correct: bool = False
    breakdown_str: str = ""


def compute_reward(
    confidence: int,
    predicted: str,
    ground_truth: str,
    aliases: list[str],
    domain: str,
) -> RewardBreakdown:
    """Compute full reward breakdown for one episode."""
    acc = accuracy_reward(predicted, ground_truth, aliases, domain)
    was_correct = acc >= 0.5

    br  = brier_reward(confidence, was_correct)
    oc  = overconfidence_penalty(confidence, was_correct)
    uc  = underconfidence_penalty(confidence, was_correct)

    raw = cfg.W_ACCURACY * acc + cfg.W_CALIBRATION * br + oc + uc
    total = float(np.clip(raw, cfg.REWARD_CLIP_LOW, cfg.REWARD_CLIP_HIGH))

    icon = "✅" if was_correct else "❌"
    breakdown_str = (
        f"{icon} acc={acc:.2f} brier={br:.2f} "
        f"oc_pen={oc:.2f} uc_pen={uc:.2f} → total={total:.3f}"
    )

    return RewardBreakdown(
        accuracy_score=acc,
        brier_reward_val=br,
        overconfidence_penalty_val=oc,
        underconfidence_penalty_val=uc,
        total=total,
        was_correct=was_correct,
        breakdown_str=breakdown_str,
    )


# ── RewardHistory ─────────────────────────────────────────────────────────────

class RewardHistory:
    """
    Rolling record of all episode outcomes.
    Feeds into calibration metrics and training logs.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def append(
        self,
        confidence: int,
        was_correct: bool,
        domain: str,
        difficulty: str,
        reward: float,
        is_abstention: bool = False,
    ) -> None:
        self._records.append({
            "confidence": confidence,
            "was_correct": was_correct,
            "domain": domain,
            "difficulty": difficulty,
            "reward": reward,
            "is_abstention": is_abstention,
        })

    def get_calibration_report(
        self, domain: Optional[str] = None
    ) -> CalibrationReport:
        records = self._records
        if domain:
            records = [r for r in records if r["domain"] == domain]
        if not records:
            return CalibrationReport(domain=domain)
        confs = [r["confidence"] for r in records]
        corrs = [r["was_correct"] for r in records]
        absts = [r["is_abstention"] for r in records]
        return compute_report(confs, corrs, absts, domain=domain)

    def get_domain_profiles(self) -> dict[str, CalibrationReport]:
        return {d: self.get_calibration_report(domain=d) for d in cfg.DOMAINS}

    def get_training_snapshot(self, last_n: int = 100) -> dict:
        records = self._records[-last_n:]
        if not records:
            return {
                "ece": 1.0, "accuracy": 0.0, "mean_confidence": 50.0,
                "overconfidence_rate": 0.5, "brier_score": 0.25, "mean_reward": 0.0,
            }
        confs = [r["confidence"] for r in records]
        corrs = [r["was_correct"] for r in records]
        rewards = [r["reward"] for r in records]
        rep = compute_report(confs, corrs)
        return {
            "ece": rep.ece,
            "accuracy": rep.accuracy,
            "mean_confidence": rep.mean_confidence,
            "overconfidence_rate": rep.overconfidence_rate,
            "brier_score": rep.brier_score,
            "mean_reward": float(np.mean(rewards)),
        }

    def to_dataframe(self) -> "pd.DataFrame":
        return pd.DataFrame(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()
