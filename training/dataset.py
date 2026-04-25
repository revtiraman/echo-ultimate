"""
ECHO ULTIMATE — GRPO Training Dataset Builder.
"""

import logging
from typing import Optional

from config import cfg
from env.parser import format_prompt
from env.task_bank import TaskBank

logger = logging.getLogger(__name__)


def build_grpo_dataset(
    task_bank: TaskBank,
    n_samples: int,
    phase: int,
    tokenizer=None,
) -> "datasets.Dataset":
    """
    Build a HuggingFace Dataset for GRPOTrainer.

    Each row:
        prompt, domain, difficulty, answer, answer_aliases, task_id, difficulty_score
    """
    from datasets import Dataset

    task_bank.ensure_loaded()
    tasks = task_bank.get_batch(n_samples, phase=phase)

    rows = {
        "prompt":          [],
        "domain":          [],
        "difficulty":      [],
        "answer":          [],
        "answer_aliases":  [],
        "task_id":         [],
        "difficulty_score": [],
    }

    for task in tasks:
        raw_prompt = format_prompt(
            task["question"], task["domain"], task["difficulty"],
            show_difficulty=(phase == 1),
        )
        # Apply chat template if tokenizer available
        if tokenizer is not None:
            try:
                messages = [
                    {"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Question: {task['question']}"},
                ]
                raw_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass   # fall back to raw format

        rows["prompt"].append(raw_prompt)
        rows["domain"].append(task["domain"])
        rows["difficulty"].append(task["difficulty"])
        rows["answer"].append(task["answer"])
        rows["answer_aliases"].append(task.get("answer_aliases", [task["answer"]]))
        rows["task_id"].append(task["id"])
        rows["difficulty_score"].append(task.get("difficulty_score", 0.5))

    return Dataset.from_dict(rows)


def build_curriculum_datasets(
    task_bank: TaskBank,
    tokenizer=None,
) -> tuple:
    """
    Build all 3 phase datasets.
    Returns (phase1_dataset, phase2_dataset, phase3_dataset).
    """
    phase1 = build_grpo_dataset(
        task_bank, cfg.PHASE_1_STEPS * cfg.BATCH_SIZE, phase=1, tokenizer=tokenizer
    )
    phase2 = build_grpo_dataset(
        task_bank, cfg.PHASE_2_STEPS * cfg.BATCH_SIZE, phase=2, tokenizer=tokenizer
    )
    phase3 = build_grpo_dataset(
        task_bank, cfg.PHASE_3_STEPS * cfg.BATCH_SIZE, phase=3, tokenizer=tokenizer
    )
    return phase1, phase2, phase3
