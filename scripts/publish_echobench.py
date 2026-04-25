"""
EchoBench Publisher
Converts ECHO task bank to HuggingFace Dataset and publishes to the Hub.

Usage:
  python scripts/publish_echobench.py --token YOUR_HF_TOKEN
  python scripts/publish_echobench.py --token YOUR_HF_TOKEN --repo your-username/echobench
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tasks_from_bank():
    """Load all tasks from ECHO's task bank."""
    from env.task_bank import TaskBank
    from config import cfg

    bank = TaskBank()
    print("Loading task bank (downloads datasets if not cached)…")
    bank.ensure_loaded()

    all_tasks = []
    for domain in cfg.DOMAINS:
        for difficulty in cfg.DIFFICULTIES:
            bucket = bank._tasks.get(domain, {}).get(difficulty, [])
            all_tasks.extend(bucket)
            print(f"  {domain}/{difficulty}: {len(bucket)} tasks")

    print(f"\nTotal tasks: {len(all_tasks)}")
    return all_tasks


def tasks_to_hf_dataset(tasks):
    """Convert task dicts to HuggingFace DatasetDict split by domain."""
    from datasets import Dataset, DatasetDict

    records = []
    for task in tasks:
        records.append({
            "id":               str(task.get("id", "")),
            "domain":           str(task.get("domain", "")),
            "difficulty":       str(task.get("difficulty", "")),
            "difficulty_score": float(task.get("difficulty_score", 0.5)),
            "question":         str(task.get("question", "")),
            "answer":           str(task.get("answer", "")),
            "answer_aliases":   [str(a) for a in task.get("answer_aliases", [])],
            "source_dataset":   str(task.get("source_dataset", "")),
        })

    splits = {}
    domains = sorted({r["domain"] for r in records})
    for domain in domains:
        subset = [r for r in records if r["domain"] == domain]
        splits[domain] = Dataset.from_list(subset)
        print(f"  Split '{domain}': {len(subset)} rows")

    splits["all"] = Dataset.from_list(records)
    print(f"  Split 'all':    {len(records)} rows")
    return DatasetDict(splits)


_DATASET_CARD = """\
---
license: apache-2.0
task_categories:
- question-answering
- text-classification
language:
- en
tags:
- calibration
- metacognition
- llm-evaluation
- grpo
- openenv
size_categories:
- 10K<n<100K
---

# EchoBench

**The first public benchmark for LLM metacognitive calibration.**

EchoBench contains questions across 7 domains for training and evaluating
whether language models accurately predict their own probability of being correct.

## Domains

| Domain | Source | Description |
|--------|--------|-------------|
| Math | GSM8K | Grade-school math word problems |
| Logic | AI2-ARC | Multiple-choice science reasoning |
| Factual | TriviaQA | Open-domain factual questions |
| Science | SciQ | Multiple-choice science questions |
| Medical | MedMCQA | Medical licensing exam questions |
| Coding | Synthetic | Code output/complexity prediction |
| Creative | Synthetic | Wordplay, synonyms, literary devices |

## Usage

```python
from datasets import load_dataset

# Load all tasks
ds = load_dataset("revti126/echobench", "all")

# Load a specific domain
math_ds = load_dataset("revti126/echobench", "math")
print(math_ds["train"][0])
```

## Task Format

Each row contains:
- `id` — unique task identifier (`math_easy_00042`)
- `domain` — one of math/logic/factual/science/medical/coding/creative
- `difficulty` — easy / medium / hard
- `difficulty_score` — float 0.0 (hardest) → 1.0 (easiest)
- `question` — the question text
- `answer` — canonical correct answer
- `answer_aliases` — all accepted answer strings
- `source_dataset` — originating HuggingFace dataset

## Citation

```bibtex
@misc{echobench-2025,
  title  = {EchoBench: A Benchmark for LLM Metacognitive Calibration},
  author = {Tripathi, Revtiraman and Pandey, Vikas Dev},
  year   = {2025},
  url    = {https://huggingface.co/datasets/revti126/echobench},
  note   = {Created for ECHO ULTIMATE — OpenEnv Hackathon 2025}
}
```

*Part of the [ECHO ULTIMATE](https://huggingface.co/spaces/revti126/echo-ultimate) project.*
"""


def publish_to_hub(dataset_dict, repo_id: str, token: str):
    """Push dataset to HuggingFace Hub and upload the dataset card."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"\nCreating repository: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:
        print(f"  Note: {exc}")

    print("Pushing dataset…")
    dataset_dict.push_to_hub(repo_id, token=token)

    print("Uploading dataset card…")
    api.upload_file(
        path_or_fileobj=_DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n✅  EchoBench published: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Publish ECHO task bank as EchoBench HuggingFace dataset."
    )
    parser.add_argument("--token",  required=True, help="HuggingFace API write token")
    parser.add_argument("--repo",   default="revti126/echobench",
                        help="HuggingFace repo ID (default: revti126/echobench)")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print("=== EchoBench Publisher ===\n")

    tasks       = load_tasks_from_bank()
    if not tasks:
        print("❌  No tasks loaded. Run `python run.py download` first.")
        sys.exit(1)

    dataset_dict = tasks_to_hf_dataset(tasks)
    url          = publish_to_hub(dataset_dict, args.repo, args.token)

    print(f"\n=== Done ===")
    print(f"Dataset URL: {url}")
    print(f"Add to README.md and openenv.yaml:")
    print(f"  dataset: {args.repo}")


if __name__ == "__main__":
    main()
