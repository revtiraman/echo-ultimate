"""
ECHO ULTIMATE — 5 calibration metrics implemented from scratch.

ECE, MCE, Brier Score, Sharpness, Resolution — all with mathematical comments.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


# ── CalibrationReport ─────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """Complete calibration profile for an agent over N episodes."""
    ece: float = 0.0
    mce: float = 0.0
    brier_score: float = 0.25
    sharpness: float = 0.0
    resolution: float = 0.0
    accuracy: float = 0.0
    mean_confidence: float = 50.0
    overconfidence_rate: float = 0.0
    underconfidence_rate: float = 0.0
    abstention_rate: float = 0.0
    bin_data: dict = field(default_factory=dict)
    n_samples: int = 0
    domain: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "sharpness": round(self.sharpness, 4),
            "resolution": round(self.resolution, 4),
            "accuracy": round(self.accuracy, 4),
            "mean_confidence": round(self.mean_confidence, 2),
            "overconfidence_rate": round(self.overconfidence_rate, 4),
            "underconfidence_rate": round(self.underconfidence_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "n_samples": self.n_samples,
            "domain": self.domain,
        }

    def summary_str(self) -> str:
        return (
            f"ECE={self.ece:.3f} | MCE={self.mce:.3f} | Brier={self.brier_score:.3f} | "
            f"Acc={self.accuracy:.1%} | MeanConf={self.mean_confidence:.0f}% | "
            f"OverconfRate={self.overconfidence_rate:.1%} | n={self.n_samples}"
        )


# ── Bin builder ───────────────────────────────────────────────────────────────

def _build_bins(
    confidences: list[int],
    correctness: list[bool],
    n_bins: int,
) -> dict[int, dict]:
    """
    Partition (confidence, outcome) pairs into equal-width bins [0,10), [10,20), …
    Returns dict keyed by bin center with accuracy, mean_conf, and count.
    """
    bins: dict[int, dict] = {}
    step = 100 // n_bins  # e.g. 10 for n_bins=10

    for bin_lower in range(0, 100, step):
        bin_upper = bin_lower + step
        center = bin_lower + step // 2
        indices = [
            i for i, c in enumerate(confidences)
            if bin_lower <= c < bin_upper
        ]
        if not indices:
            bins[center] = {"accuracy": 0.0, "mean_conf": center / 100.0, "count": 0}
            continue
        acc = float(np.mean([correctness[i] for i in indices]))
        mc  = float(np.mean([confidences[i] for i in indices])) / 100.0
        bins[center] = {"accuracy": acc, "mean_conf": mc, "count": len(indices)}

    return bins


# ── Metric functions ──────────────────────────────────────────────────────────

def ece(
    confidences: list[int],
    correctness: list[bool],
    n_bins: int = cfg.N_CALIBRATION_BINS,
) -> float:
    """
    Expected Calibration Error.

    ECE = Σ_{m=1}^{M} (|B_m| / n) * |acc(B_m) - conf(B_m)|

    where B_m = samples in bin m, acc = fraction correct, conf = mean confidence.
    Lower is better. Perfect calibration = 0.0.
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    bins = _build_bins(confidences, correctness, n_bins)
    ece_val = 0.0
    for b in bins.values():
        if b["count"] == 0:
            continue
        ece_val += (b["count"] / n) * abs(b["accuracy"] - b["mean_conf"])
    return float(ece_val)


def mce(
    confidences: list[int],
    correctness: list[bool],
    n_bins: int = cfg.N_CALIBRATION_BINS,
) -> float:
    """
    Maximum Calibration Error.

    MCE = max_m |acc(B_m) - conf(B_m)|

    Worst-case calibration error across all non-empty bins.
    """
    if not confidences:
        return 0.0
    bins = _build_bins(confidences, correctness, n_bins)
    gaps = [
        abs(b["accuracy"] - b["mean_conf"])
        for b in bins.values() if b["count"] > 0
    ]
    return float(max(gaps)) if gaps else 0.0


def brier_score(
    confidences: list[int],
    correctness: list[bool],
) -> float:
    """
    Brier Score.

    BS = (1/n) Σ (p_i - o_i)^2

    p_i = confidence_i / 100 (forecast probability)
    o_i = 1 if correct, 0 if wrong (outcome)
    Range [0, 1]. Lower = better.
    Perfect model = 0. Random (50%) = 0.25.
    Always guessing 1.0 on wrong answers = 1.0.
    """
    if not confidences:
        return 0.25
    scores = [
        (c / 100.0 - float(o)) ** 2
        for c, o in zip(confidences, correctness)
    ]
    return float(np.mean(scores))


def sharpness(confidences: list[int]) -> float:
    """
    Sharpness.

    Sharpness = (1/n) Σ (p_i - mean(p))^2

    Variance of predicted probabilities.
    Higher sharpness = more decisive predictions.
    Can be good (confident correct) or bad (confident wrong).
    """
    if not confidences:
        return 0.0
    probs = [c / 100.0 for c in confidences]
    return float(np.var(probs))


def resolution(
    confidences: list[int],
    correctness: list[bool],
    n_bins: int = cfg.N_CALIBRATION_BINS,
) -> float:
    """
    Resolution.

    Resolution = (1/n) Σ_m |B_m| * (acc(B_m) - overall_acc)^2

    Measures how much the binned confidence predictions differ from overall accuracy.
    Higher resolution = predictions contain more information beyond the base rate.
    """
    if not correctness:
        return 0.0
    n = len(correctness)
    overall_acc = float(np.mean(correctness))
    bins = _build_bins(confidences, correctness, n_bins)
    res = 0.0
    for b in bins.values():
        if b["count"] == 0:
            continue
        res += (b["count"] / n) * (b["accuracy"] - overall_acc) ** 2
    return float(res)


# ── Combined report ───────────────────────────────────────────────────────────

def compute_report(
    confidences: list[int],
    correctness: list[bool],
    abstentions: Optional[list[bool]] = None,
    domain: Optional[str] = None,
    n_bins: int = cfg.N_CALIBRATION_BINS,
) -> CalibrationReport:
    """
    Compute all 5 calibration metrics plus operational rates in one call.

    Args:
        confidences:  list of int [0, 100]
        correctness:  list of bool
        abstentions:  list of bool (True = agent said "I don't know")
        domain:       optional domain label for reporting
    """
    if not confidences:
        return CalibrationReport(n_samples=0, domain=domain)

    n = len(confidences)
    overall_acc = float(np.mean(correctness))

    # Overconfidence rate: fraction of WRONG answers with conf >= threshold
    wrong_mask = [not c for c in correctness]
    wrong_high = sum(
        1 for c, w in zip(confidences, wrong_mask)
        if w and c >= cfg.OVERCONFIDENCE_THRESHOLD
    )
    n_wrong = sum(wrong_mask)
    overconf_rate = wrong_high / max(n_wrong, 1)

    # Underconfidence rate: fraction of CORRECT answers with conf <= threshold
    correct_low = sum(
        1 for c, ok in zip(confidences, correctness)
        if ok and c <= cfg.UNDERCONFIDENCE_THRESHOLD
    )
    n_correct = sum(correctness)
    underconf_rate = correct_low / max(n_correct, 1)

    abst_rate = 0.0
    if abstentions:
        abst_rate = sum(abstentions) / n

    bins = _build_bins(confidences, correctness, n_bins)

    return CalibrationReport(
        ece=ece(confidences, correctness, n_bins),
        mce=mce(confidences, correctness, n_bins),
        brier_score=brier_score(confidences, correctness),
        sharpness=sharpness(confidences),
        resolution=resolution(confidences, correctness, n_bins),
        accuracy=overall_acc,
        mean_confidence=float(np.mean(confidences)),
        overconfidence_rate=overconf_rate,
        underconfidence_rate=underconf_rate,
        abstention_rate=abst_rate,
        bin_data=bins,
        n_samples=n,
        domain=domain,
    )
