"""ECHO ULTIMATE — Domain-specific answer graders (thin wrappers around reward.py)."""

from env.reward import accuracy_reward


def grade(predicted: str, task: dict) -> float:
    """Grade a predicted answer against a task dict. Returns float in [0, 1]."""
    return accuracy_reward(
        predicted=predicted,
        ground_truth=task.get("answer", ""),
        answer_aliases=task.get("answer_aliases", []),
        domain=task.get("domain", "factual"),
    )
