"""
ECHO ULTIMATE — Phase 4: Adversarial Self-Play.

After Phase 3, the model generates its own hard calibration questions targeting
its weakest domains, then trains on them for an additional 500 steps.
This is a research feature — all errors are caught and logged without crashing.
"""

import json
import logging
import re
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from config import cfg

logger = logging.getLogger(__name__)

_WEAK_DOMAIN_DEFAULT = ["medical", "coding", "science"]


@dataclass
class AdversarialQuestion:
    question: str
    domain: str
    difficulty: str = "adversarial"
    generated_by: str = "self-play"


def generate_adversarial_questions(
    model,
    tokenizer,
    weak_domains: List[str],
    n_questions: int = 200,
    config=None,
) -> List[dict]:
    """
    Model generates questions in domains where it is overconfident.
    Returns a list of task dicts compatible with TaskBank format.
    """
    config = config or cfg
    questions = []
    per_domain = max(1, n_questions // len(weak_domains))

    for domain in weak_domains:
        prompt = (
            f"Generate {per_domain} challenging {domain} questions where an AI might be "
            f"overconfident. Each should have a clear, non-obvious correct answer.\n"
            f"Format each as:\nQ: [question]\nA: [correct answer]\n---\n"
            f"Generate {per_domain} questions now:\n"
        )
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            pairs = generated.split("---")
            for pair in pairs:
                q_match = re.search(r"Q:\s*(.+?)(?=A:|$)", pair, re.DOTALL)
                a_match = re.search(r"A:\s*(.+?)(?=Q:|---$|$)", pair, re.DOTALL)
                if q_match and a_match:
                    q_text = q_match.group(1).strip().replace("\n", " ")
                    a_text = a_match.group(1).strip().replace("\n", " ")
                    if q_text and a_text:
                        questions.append({
                            "id":               f"adversarial_{domain}_{len(questions):05d}",
                            "domain":           domain,
                            "difficulty":       "adversarial",
                            "difficulty_score": 0.10,
                            "question":         q_text,
                            "answer":           a_text,
                            "answer_aliases":   [a_text],
                            "source_dataset":   "self_play",
                            "metadata":         {"generated_by": "echo_phase4"},
                        })
        except Exception as exc:
            logger.error("Phase 4 generation failed for domain %s: %s", domain, exc)

    logger.info("Phase 4: generated %d adversarial questions", len(questions))
    return questions[:n_questions]


def _get_weak_domains(reward_history) -> List[str]:
    """Return the 3 domains with the highest ECE (most miscalibrated)."""
    if reward_history is None:
        return _WEAK_DOMAIN_DEFAULT

    try:
        profiles = reward_history.get_domain_profiles()
        if not profiles:
            return _WEAK_DOMAIN_DEFAULT
        sorted_domains = sorted(
            [(d, p.ece) for d, p in profiles.items() if p.n_samples > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        weak = [d for d, _ in sorted_domains[:3]]
        return weak if weak else _WEAK_DOMAIN_DEFAULT
    except Exception:
        return _WEAK_DOMAIN_DEFAULT


def run_phase_4(trainer, model, tokenizer, reward_history, config=None) -> List[dict]:
    """
    Run adversarial self-play phase after Phase 3.
    Generates questions targeting weak domains, saves them, and trains 500 more steps.
    """
    config = config or cfg
    logger.info("=== PHASE 4: ADVERSARIAL SELF-PLAY ===")
    print("\n🧪  Phase 4: Adversarial Self-Play")

    try:
        weak_domains = _get_weak_domains(reward_history)
        print(f"    Targeting weak domains: {weak_domains}")

        questions = generate_adversarial_questions(
            model, tokenizer, weak_domains, n_questions=200, config=config
        )
        print(f"    Generated {len(questions)} adversarial questions")

        # Save for inspection / reuse
        out_path = "adversarial_questions.json"
        with open(out_path, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"    Saved to {out_path}")

        if not questions:
            logger.warning("Phase 4: no questions generated — skipping extra training")
            return questions

        # Build a small dataset from the adversarial questions and run 500 more steps
        try:
            from training.dataset import build_grpo_dataset
            from env.task_bank import TaskBank

            # Inject questions into a temporary TaskBank and rebuild dataset
            tmp_bank = TaskBank()
            tmp_bank.ensure_loaded()
            for q in questions:
                d = q["domain"]
                if d in tmp_bank._tasks:
                    tmp_bank._tasks[d]["hard"].append(q)

            adv_dataset = build_grpo_dataset(
                tmp_bank,
                n_samples=min(500 * config.BATCH_SIZE, len(questions) * 4),
                phase=3,
                tokenizer=tokenizer,
            )
            trainer.train_dataset = adv_dataset
            trainer.args.max_steps = (trainer.state.global_step or 0) + 500
            print("    Training 500 steps on adversarial questions…")
            trainer.train(resume_from_checkpoint=False)
            print("    Phase 4 complete ✅")
        except Exception as exc:
            logger.error("Phase 4 extra training failed: %s", exc)

        return questions

    except Exception as exc:
        logger.error("Phase 4 run_phase_4 error: %s", exc)
        return []
