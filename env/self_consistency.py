"""
ECHO ULTIMATE — Self-Consistency Confidence Checker.

Samples N answers for the same question. If answers disagree,
automatically reduces the stated confidence by CONSISTENCY_DISCOUNT.

This is a key innovation over the base ECHO environment.
In training: disabled (too slow, adds noise).
In demo: enabled (impressive, shows genuine uncertainty awareness).
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import cfg
from env.parser import parse_response, ParseResult

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Result of self-consistency checking for one question."""
    answers: list[str]           = field(default_factory=list)
    confidences: list[int]       = field(default_factory=list)
    final_answer: str            = ""
    final_confidence: int        = 50
    agreement_rate: float        = 1.0
    was_adjusted: bool           = False
    adjustment_amount: int       = 0
    parse_results: list          = field(default_factory=list)


class SelfConsistencyChecker:
    """
    Multi-sample confidence adjustment.

    Algorithm:
    1. Generate n_samples responses for the same prompt
    2. Parse each into (confidence, answer)
    3. Find majority-vote answer
    4. agreement_rate = fraction of samples matching majority
    5. If agreement_rate < 1.0:
         final_confidence = round(mean_confidence * (1 - CONSISTENCY_DISCOUNT))
       else:
         final_confidence = mean_confidence (unchanged)
    6. Return ConsistencyResult with final_answer and final_confidence
    """

    def __init__(self, n_samples: int = cfg.SELF_CONSISTENCY_SAMPLES) -> None:
        self.n_samples = n_samples
        self.discount = cfg.CONSISTENCY_DISCOUNT

    def check(
        self,
        prompt: str,
        generate_fn: Callable[[str], str],
        n_samples: Optional[int] = None,
    ) -> ConsistencyResult:
        """
        Run n_samples generations and return a consistency-adjusted result.

        Args:
            prompt:      formatted question prompt
            generate_fn: callable(prompt) -> raw LLM output string
            n_samples:   override default sample count
        """
        n = n_samples or self.n_samples
        parsed_list: list[ParseResult] = []
        answers = []
        confidences = []

        for i in range(n):
            try:
                raw = generate_fn(prompt)
                parsed = parse_response(raw)
            except Exception as exc:
                logger.warning("SelfConsistencyChecker sample %d failed: %s", i, exc)
                from env.parser import ParseResult as PR
                parsed = PR(confidence=50, answer="", raw="")

            parsed_list.append(parsed)
            answers.append(parsed.answer.strip().lower())
            confidences.append(parsed.confidence)

        if not answers:
            return ConsistencyResult(final_confidence=50, final_answer="")

        # Majority vote answer
        counter = Counter(answers)
        majority_answer_lower, majority_count = counter.most_common(1)[0]
        agreement_rate = majority_count / n

        # Find the original-cased answer for the majority
        final_answer = ""
        for pr in parsed_list:
            if pr.answer.strip().lower() == majority_answer_lower:
                final_answer = pr.answer
                break

        mean_conf = round(sum(confidences) / len(confidences))

        # Apply discount if answers disagree
        was_adjusted = agreement_rate < 1.0
        if was_adjusted:
            adjusted = round(mean_conf * (1.0 - self.discount))
            adjustment_amount = mean_conf - adjusted
            final_confidence = max(cfg.CONFIDENCE_MIN, adjusted)
        else:
            final_confidence = mean_conf
            adjustment_amount = 0

        return ConsistencyResult(
            answers=[pr.answer for pr in parsed_list],
            confidences=confidences,
            final_answer=final_answer,
            final_confidence=final_confidence,
            agreement_rate=agreement_rate,
            was_adjusted=was_adjusted,
            adjustment_amount=adjustment_amount,
            parse_results=parsed_list,
        )

    def format_explanation(self, result: ConsistencyResult) -> str:
        """Human-readable explanation of the consistency check result."""
        if not result.was_adjusted:
            return (
                f"✅ All {len(result.answers)} samples agreed → "
                f"confidence unchanged at {result.final_confidence}%"
            )
        return (
            f"⚠️  Samples disagreed (agreement={result.agreement_rate:.0%}) → "
            f"confidence reduced by {result.adjustment_amount}% "
            f"to {result.final_confidence}%\n"
            f"  Samples: {result.answers}"
        )
