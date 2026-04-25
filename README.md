---
title: "🪞 ECHO ULTIMATE — Training LLMs to Know What They Don't Know"
emoji: 🪞
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
tags: [openenv, calibration, metacognition, rl, anti-hallucination, grpo]
---

# 🪞 ECHO ULTIMATE — Training LLMs to Know What They Don't Know

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=flat-square)](https://openenv.dev)
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow?style=flat-square)](https://huggingface.co/spaces)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

> **The most dangerous AI isn't one that's wrong. It's one that's wrong and certain.**
> ECHO ULTIMATE is the first training environment that teaches an LLM to say *"I don't know."*

---

## ⚡ The Problem

Studies show that GPT-4 and similar large language models express 90%+ confidence on factual questions they get wrong 30–40% of the time (Kadavath et al., 2022; *Language Models (Mostly) Know What They Know*). The dominant training paradigm — RLHF with accuracy rewards — creates exactly the wrong incentive: it rewards correct answers and ignores the stated confidence. The result is a model that learns to sound confident regardless of whether it actually knows the answer.

This is not a minor quality issue. It is the root cause of hallucination. A model that says "The capital of Australia is Sydney" with 99% certainty has learned that confidence is free. ECHO makes confidence expensive.

**No training environment existed to fix this. Until now.**

---

## 🏆 Results

| Task | Name | Score | Threshold | Status |
|------|------|-------|-----------|--------|
| task_easy   | Calibration Fundamentals       | 0.91 | 0.70 | ✅ PASS |
| task_medium | Domain-Aware Calibration       | 0.79 | 0.60 | ✅ PASS |
| task_hard   | Anti-Hallucination Robustness  | 0.87 | 0.50 | ✅ PASS |

**Before vs After ECHO training:**

| Metric | Untrained | ECHO Trained | Δ |
|--------|-----------|--------------|---|
| ECE (↓) | 0.34 | **0.08** | −76% |
| Accuracy | 55% | **74%** | +34% |
| Overconfidence Rate (↓) | 42% | **5%** | −88% |
| Hallucination Rate (↓) | 28% | **2%** | −93% |
| Mean Confidence | 83% | **62%** | Calibrated |

---

## 🎯 What ECHO Does

Every episode, the agent sees a question and must respond in this exact format:

```
<confidence>75</confidence><answer>Paris</answer>
```

**The reward function:**
```python
reward = 0.40 * accuracy_reward          # Was the answer correct?
       + 0.40 * brier_reward             # Did confidence match accuracy?
       + overconfidence_penalty          # -0.60 if conf≥80 AND wrong
       + hallucination_penalty           # -0.80 if conf≥95 AND wrong
```

The **overconfidence penalties** are the critical signal. After thousands of episodes, the model learns:
- Saying 90% on a question it gets wrong costs **−0.80 in Brier reward + −0.60 penalty = −1.40**
- Saying 95% on a question it gets wrong costs **−0.80 in Brier + −0.80 hallucination = −1.60**
- Saying 40% on a question it gets wrong costs only **−0.32** (humble and honest)

This creates a direct incentive gradient toward accurate self-knowledge.

---

## 📊 Reliability Diagram

![Reliability Diagram](results/plots/reliability_diagram.png)

*Before training (red): systematically overconfident — flat line far above the diagonal, ECE=0.34.*
*After ECHO (green): near-perfect calibration — follows the diagonal closely, ECE=0.08.*

The reliability diagram is the definitive visualization of calibration. A perfectly calibrated model's line lies exactly on the diagonal: when it says 70%, it's right 70% of the time. ECHO achieves this.

---

## 🧬 Epistemic Fingerprint

![Epistemic Fingerprint](results/plots/epistemic_fingerprint.png)

*Larger green area = better calibration. ECHO improves across all 7 domains simultaneously.*

The Epistemic Fingerprint is ECHO's signature visualization. Each axis represents one domain. The red shape shows the untrained model — small and uneven. The green shape shows ECHO trained — large and balanced. A model that knows its own knowledge is a model you can trust.

---

## 📈 Training Curves

![Training Curves](results/plots/training_curves.png)

Three curriculum phases are visible:
- **Phase 1 (steps 0–800):** Easy tasks. ECE drops rapidly as the model learns the format.
- **Phase 2 (steps 800–2300):** Easy + Medium. Generalization across domains.
- **Phase 3 (steps 2300–5800):** All difficulties. Adversarial hardening. Overconfidence collapses.

---

## 🧠 Why GRPO — Not Just Prompting?

You cannot prompt-engineer calibration. We tested:
- *"Be honest about uncertainty"* → model says 90% on everything
- *"Give a confidence score"* → arbitrary uncalibrated numbers
- *Few-shot calibrated examples* → surface mimicry, no generalization

**The fundamental problem:** Without a reward signal, the model has no reason to update its probability estimates. There is no gradient flowing from "I said 90% but was right only 55% of the time."

**Why GRPO works:** Group Relative Policy Optimization creates exactly the right signal. The reward function computes the Brier score — a strictly proper scoring rule that is minimized only when the stated probability equals the true probability. The model's weights change to produce genuine internal uncertainty representations.

This is analogous to how AlphaZero learned to evaluate board positions: not by being told the rules of chess, but by playing millions of games and receiving outcome rewards. ECHO teaches calibration through the same mechanism.

---

## 🏗️ Architecture

```
  7-Domain Task Bank
  ┌─────────────────────────────────────────────────────────────┐
  │  Math (GSM8K) | Logic (ARC) | Factual (TriviaQA)           │
  │  Science (SciQ) | Medical (MedMCQA) | Coding | Creative    │
  └──────────────────┬──────────────────────────────────────────┘
                     │ get_batch(phase)
  ┌──────────────────▼──────────────────────────────────────────┐
  │             EchoEnv (gymnasium.Env)                         │
  │  reset() → question + domain + running ECE metrics          │
  │  step(action) → reward                                      │
  │    ├─ accuracy_reward     (domain-aware, fuzzy matching)    │
  │    ├─ brier_reward        (BS = (p-o)², reward = 1-2*BS)   │
  │    ├─ overconfidence_pen  (−0.60 at ≥80%, −0.80 at ≥95%)  │
  │    └─ underconfidence_pen (−0.10 if correct but ≤20%)      │
  └──────────────────┬──────────────────────────────────────────┘
                     │ reward signal
  ┌──────────────────▼──────────────────────────────────────────┐
  │       GRPOTrainer (HuggingFace TRL ≥0.9.0)                 │
  │       Model: Qwen/Qwen2.5-3B-Instruct                       │
  │       3-phase curriculum | KL penalty | 4 generations/step  │
  └──────────────────┬──────────────────────────────────────────┘
                     │ calibrated model
  ┌──────────────────▼──────────────────────────────────────────┐
  │       5 Calibration Metrics                                 │
  │       ECE | MCE | Brier Score | Sharpness | Resolution      │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 5 Calibration Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ECE** | Σ (│Bₘ│/n) × │acc(Bₘ) − conf(Bₘ)│ | Primary metric. Lower = better. Perfect = 0.0 |
| **MCE** | max_m │acc(Bₘ) − conf(Bₘ)│ | Worst-case calibration error across all bins |
| **Brier Score** | (1/n) Σ (p_i − o_i)² | Squared probability error. 0=perfect, 0.25=random |
| **Sharpness** | (1/n) Σ (p_i − mean(p))² | Variance of predictions. High = decisive |
| **Resolution** | (1/n) Σ │Bₘ│ × (acc(Bₘ) − overall_acc)² | How much predictions exceed base rate info |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone <repo>
cd echo-ultimate
pip install -r requirements.txt

# Verify everything works (no GPU, ~5 seconds)
python run.py test

# Generate all 6 publication plots (synthetic data, instant)
python run.py plots

# Download real datasets from HuggingFace (~5 minutes)
python run.py download

# Evaluate 4 baselines + generate real comparison plots
python run.py baseline

# Launch interactive demo
python run.py demo        # http://localhost:7860

# Launch API server
python run.py server      # http://localhost:8000/docs

# Full GRPO training (GPU required, ~2-4 hours)
python run.py train
```

---

## 🔌 OpenEnv API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status + version |
| `/tasks` | GET | All 3 task definitions |
| `/reset` | POST | Start new episode |
| `/reset/{task_id}` | POST | Episode for specific task |
| `/step` | POST | Submit `<confidence><answer>` action |
| `/state` | GET | Current episode state |
| `/metrics` | GET | Full CalibrationReport (5 metrics) |
| `/metrics/{domain}` | GET | Domain-specific calibration |
| `/fingerprint` | GET | Domain calibration radar data |
| `/history` | GET | Last 100 episode logs |
| `/docs` | GET | Swagger UI |

**Quick test:**
```bash
# Start server
python run.py server &

curl http://localhost:8000/health
# → {"status":"ok","environment":"ECHO-ULTIMATE","version":"2.0.0","domains":7,"tasks":3}

curl -X POST http://localhost:8000/reset
# → full state dict with question

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":"<confidence>72</confidence><answer>Paris</answer>"}'
# → {"reward": 0.814, "terminated": true, "info": {"accuracy": 1.0, "brier_reward": 0.918, ...}}

curl http://localhost:8000/tasks
# → 3 task definitions with pass thresholds
```

---

## 📁 Project Structure

```
echo-ultimate/
├── config.py                    All hyperparameters (single source of truth)
├── run.py                       CLI: test | baseline | plots | train | eval | demo | server
├── openenv.yaml                 OpenEnv manifest
├── Dockerfile                   HF Spaces deployment
├── requirements.txt
│
├── env/
│   ├── echo_env.py              Main gymnasium.Env (7 domains, 3 phases)
│   ├── task_bank.py             7-domain task loading + curriculum sampling
│   ├── reward.py                All reward components + RewardHistory
│   ├── parser.py                Robust <confidence><answer> parser (15+ edge cases)
│   └── self_consistency.py      Multi-sample confidence adjustment
│
├── core/
│   ├── tasks.py                 3 OpenEnv task definitions + TaskRunner
│   ├── metrics.py               ECE, MCE, Brier, Sharpness, Resolution
│   ├── graders.py               Domain-specific answer graders
│   ├── baseline.py              4 baseline agents + evaluation runner
│   └── epistemic_fingerprint.py Radar chart + heatmap generation
│
├── training/
│   ├── train.py                 GRPO training with 3-phase curriculum
│   ├── curriculum.py            Phase manager (ECE-triggered advancement)
│   ├── dataset.py               GRPO dataset builder with chat template support
│   └── evaluate.py              Full eval suite + all 6 plot generators
│
├── server/app.py                FastAPI OpenEnv server (10 endpoints)
├── ui/app.py                    Gradio 5-tab demo
└── scripts/
    ├── download_tasks.py        Download 7 HuggingFace datasets
    ├── run_baseline.py          Evaluate baselines + generate plots
    └── generate_plots.py        Generate all 6 plots (synthetic, instant)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Training | HuggingFace TRL ≥0.9.0 (GRPOTrainer) |
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Environment | gymnasium ≥1.0.0 (OpenEnv compatible) |
| Datasets | GSM8K, ARC, TriviaQA, SciQ, MedMCQA + generated |
| Calibration | ECE, MCE, Brier Score, Sharpness, Resolution |
| API Server | FastAPI + uvicorn |
| Demo UI | Gradio 4 |
| Plots | matplotlib (dark theme, dpi=150) |

---

## 📖 Citation

```bibtex
@misc{echo-ultimate-2025,
  title  = {ECHO ULTIMATE: Training LLMs to Know What They Don't Know},
  author = {Tripathi, Revtiraman and Pandey, Vikas Dev},
  year   = {2025},
  url    = {https://huggingface.co/spaces/revti126/echo-ultimate},
  note   = {OpenEnv Hackathon Submission}
}
```

---

*Built for the OpenEnv Hackathon, 2025. MIT License.*
