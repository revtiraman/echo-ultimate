"""
Publish ECHO ULTIMATE as a HuggingFace Space (Docker SDK, Python 3.11).

Usage:
  python scripts/publish_space.py --token YOUR_HF_TOKEN
  python scripts/publish_space.py --token YOUR_HF_TOKEN --repo your-username/echo-ultimate
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPACE_README = """\
---
title: ECHO ULTIMATE
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
license: apache-2.0
---

# ECHO ULTIMATE
### Metacognitive Calibration RL Environment

**The first open-source RL environment for training LLMs to know what they don't know.**

ECHO ULTIMATE teaches language models to accurately predict their own confidence —
solving the overconfidence problem that makes LLMs unreliable in high-stakes settings.

## What's Inside

| Tab | Feature |
|-----|---------|
| 🎯 Live Challenge | Answer questions with a confidence slider — see your calibration score in real time |
| 🤖 ECHO vs AI | Side-by-side comparison: calibrated ECHO vs overconfident baseline |
| 🧬 Epistemic Fingerprint | Radar chart of per-domain calibration accuracy |
| 📊 Training Evidence | All 6 plots from GRPO training — ECE curves, reward curves, reliability diagrams |
| 🏆 Official Evaluation | Run the 3 OpenEnv benchmark tasks |
| ⚡ Live Training | Watch ECE drop in real-time as GRPO trains |

## How It Works

ECHO uses **GRPO (Group Relative Policy Optimization)** with a custom reward function:

```
R = accuracy_reward − overconfidence_penalty
```

The agent learns to output `<confidence>75</confidence><answer>Paris</answer>` —
pairing every answer with a calibrated probability estimate.

## EchoBench Dataset

The 7-domain benchmark: [Vikaspandey582003/echobench](https://huggingface.co/datasets/Vikaspandey582003/echobench)

| Domain | Source |
|--------|--------|
| Math | GSM8K |
| Logic | AI2-ARC |
| Factual | TriviaQA |
| Science | SciQ |
| Medical | MedMCQA |
| Coding | Synthetic |
| Creative | Synthetic |

## Citation

```bibtex
@misc{echo-ultimate-2025,
  title  = {ECHO ULTIMATE: Metacognitive Calibration RL Environment},
  author = {Tripathi, Revtiraman and Pandey, Vikas Dev},
  year   = {2025},
  url    = {https://huggingface.co/spaces/Vikaspandey582003/echo-ultimate},
  note   = {OpenEnv Hackathon 2025}
}
```
"""

# Dockerfile written into the Space — uses Python 3.11 to avoid audioop/pydub issue
_SPACE_DOCKERFILE = """\
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl git && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data results/plots

# Pre-generate all plots so Gradio loads instantly (falls back silently on failure)
RUN python scripts/generate_plots.py || echo "Plot pre-generation skipped"

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
"""

_IGNORE = {
    "__pycache__", ".git", ".gitignore", "data", "results",
    "echo_lora_adapter", "adversarial_questions.json",
    ".env", "node_modules", ".DS_Store",
}


def _should_skip(p: Path) -> bool:
    for part in p.parts:
        if part in _IGNORE or part.startswith("."):
            return True
    return p.suffix == ".pyc"


def build_space_dir(src: Path, dst: Path):
    """Copy project into dst, inject Space README, Dockerfile, and requirements."""
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if _should_skip(rel):
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

    # Inject Space-specific files (override project versions)
    (dst / "README.md").write_text(_SPACE_README, encoding="utf-8")
    (dst / "Dockerfile").write_text(_SPACE_DOCKERFILE, encoding="utf-8")

    # Use the lighter space_requirements.txt as requirements.txt
    space_req = src / "space_requirements.txt"
    if space_req.exists():
        shutil.copy2(space_req, dst / "requirements.txt")

    print(f"  Space dir prepared: {dst}")
    return dst


def publish(repo_id: str, token: str, src: Path):
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"Creating Space: {repo_id} (Docker SDK)")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print("  Repo created (or already exists)")
    except Exception as exc:
        print(f"  Note: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        space_dir = build_space_dir(src, Path(tmp) / "space")

        print("Uploading files to Space…")
        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["*.pyc", "__pycache__"],
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n✅  Space published: {url}")
    print("    Docker build takes ~5-10 minutes on HuggingFace.")
    return url


def main():
    parser = argparse.ArgumentParser(description="Publish ECHO ULTIMATE to HuggingFace Spaces.")
    parser.add_argument("--token", required=True, help="HuggingFace API write token")
    parser.add_argument("--repo", default="Vikaspandey582003/echo-ultimate",
                        help="Space repo ID (default: Vikaspandey582003/echo-ultimate)")
    args, _ = parser.parse_known_args()

    src = Path(__file__).parent.parent.resolve()
    publish(args.repo, args.token, src)


if __name__ == "__main__":
    main()
