"""
ECHO ULTIMATE — 7-domain Task Bank.
Loads from HuggingFace datasets, caches to data/, falls back to synthetic tasks.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_num(text: str) -> Optional[str]:
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def _task(domain, difficulty, idx, question, answer, aliases=None, source="synthetic", meta=None):
    diff_score = {"easy": 0.85, "medium": 0.55, "hard": 0.25}[difficulty]
    return {
        "id": f"{domain}_{difficulty}_{idx:05d}",
        "domain": domain,
        "difficulty": difficulty,
        "difficulty_score": diff_score,
        "question": question.replace("\n", " ").replace("\r", " ").strip(),
        "answer": str(answer),
        "answer_aliases": aliases or [str(answer)],
        "source_dataset": source,
        "metadata": meta or {},
    }


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_math():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
    tasks = {"easy": [], "medium": [], "hard": []}
    for i, row in enumerate(ds):
        sol = row["answer"]
        ans = _last_num(sol.split("####")[-1]) or "0"
        ans = ans.replace(",", "").strip()
        steps = len(re.findall(r"[.!?]", sol))
        if steps <= 3:
            diff = "easy"
        elif steps <= 6:
            diff = "medium"
        else:
            diff = "hard"
        tasks[diff].append(_task("math", diff, i, row["question"], ans,
                                 aliases=[ans], source="gsm8k"))
        if i >= cfg.TASKS_PER_BUCKET * 3:
            break
    return tasks


def _load_logic():
    from datasets import load_dataset
    tasks = {"easy": [], "medium": [], "hard": []}
    for cfg_name, diff in [("ARC-Easy", "easy"), ("ARC-Challenge", "hard")]:
        ds = load_dataset("ai2_arc", cfg_name, split="train", trust_remote_code=True)
        for i, row in enumerate(ds):
            labels = row["choices"]["label"]
            texts  = row["choices"]["text"]
            opts   = " | ".join(f"{l}: {t}" for l, t in zip(labels, texts))
            q = f"{row['question']}\nChoices: {opts}"
            a = row["answerKey"].strip().upper()
            tasks[diff].append(_task("logic", diff, i, q, a, source=f"arc_{diff}"))
            if i >= cfg.TASKS_PER_BUCKET:
                break
    # medium = subset of easy with extra distractor framing
    for i, t in enumerate(tasks["easy"][:cfg.TASKS_PER_BUCKET]):
        t2 = dict(t)
        t2["id"] = f"logic_medium_{i:05d}"
        t2["difficulty"] = "medium"
        t2["difficulty_score"] = 0.55
        t2["question"] = "Think carefully: " + t2["question"]
        tasks["medium"].append(t2)
    return tasks


def _load_factual():
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    tasks = {"easy": [], "medium": [], "hard": []}
    for i, row in enumerate(ds):
        q   = row["question"]
        ad  = row["answer"]
        ans = ad.get("value", "") if isinstance(ad, dict) else str(ad)
        aliases = ad.get("aliases", [ans]) if isinstance(ad, dict) else [ans]
        if not ans:
            continue
        diff = "easy" if len(ans) <= 10 else ("medium" if len(ans) <= 25 else "hard")
        tasks[diff].append(_task("factual", diff, i, q, ans,
                                 aliases=[a for a in aliases if a], source="trivia_qa"))
        if i >= cfg.TASKS_PER_BUCKET * 3:
            break
    return tasks


def _load_science():
    from datasets import load_dataset
    tasks = {"easy": [], "medium": [], "hard": []}
    try:
        ds = load_dataset("sciq", split="train", trust_remote_code=True)
        for i, row in enumerate(ds):
            q = row["question"]
            correct = row["correct_answer"]
            distractors = [row.get(f"distractor{j}", "") for j in range(1, 4)]
            all_opts = [correct] + [d for d in distractors if d]
            random.shuffle(all_opts)
            labels = ["A", "B", "C", "D"][:len(all_opts)]
            opts = " | ".join(f"{l}: {t}" for l, t in zip(labels, all_opts))
            correct_label = labels[all_opts.index(correct)]
            full_q = f"{q}\nChoices: {opts}"
            diff = ["easy", "medium", "hard"][i % 3]
            tasks[diff].append(_task("science", diff, i, full_q, correct_label,
                                     source="sciq"))
            if i >= cfg.TASKS_PER_BUCKET * 3:
                break
    except Exception as e:
        logger.warning("sciq load failed: %s", e)
    return tasks


def _load_medical():
    from datasets import load_dataset
    tasks = {"easy": [], "medium": [], "hard": []}
    try:
        ds = load_dataset("medmcqa", split="train", trust_remote_code=True)
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        topic_diff = {"anatomy": "easy", "medicine": "medium",
                      "surgery": "hard", "pharmacology": "hard"}
        for i, row in enumerate(ds):
            q = row.get("question", "")
            opts = " | ".join(f"{l}: {row.get(f'op{k}','')}"
                              for l, k in zip("ABCD", "abcd"))
            full_q = f"{q}\nChoices: {opts}"
            ans_idx = row.get("cop", 0)
            ans = label_map.get(ans_idx, "A")
            topic = str(row.get("subject_name", "")).lower()
            diff = next((v for k, v in topic_diff.items() if k in topic), "medium")
            tasks[diff].append(_task("medical", diff, i, full_q, ans, source="medmcqa"))
            if i >= cfg.TASKS_PER_BUCKET * 3:
                break
    except Exception as e:
        logger.warning("medmcqa load failed: %s", e)
    return tasks


def _load_coding():
    tasks = {"easy": [], "medium": [], "hard": []}
    easy_q = [
        ("What does print(1 + 1) output?", "2"),
        ("What does print(type(42)) output?", "<class 'int'>"),
        ("What does print('hello'[0]) output?", "h"),
        ("What does print(len([1,2,3])) output?", "3"),
        ("What does print(2 ** 8) output?", "256"),
        ("What does print(10 % 3) output?", "1"),
        ("What does bool(0) return?", "False"),
        ("What does print(round(3.7)) output?", "4"),
    ]
    medium_q = [
        ("def f(x): return x*x\nWhat does f(5) return?", "25"),
        ("x = [1,2,3]; x.append(4); what is len(x)?", "4"),
        ("What is the output of: print(list(range(3)))?", "[0, 1, 2]"),
        ("d = {'a':1}; d['b']=2; what is len(d)?", "2"),
        ("What does 'abc'.upper() return?", "ABC"),
    ]
    hard_q = [
        ("What is the time complexity of binary search?", "O(log n)"),
        ("What is the time complexity of merge sort?", "O(n log n)"),
        ("What design pattern separates object creation from use?", "Factory"),
        ("In Python, what is a generator?", "lazy iterator"),
    ]
    for i, (q, a) in enumerate(easy_q):
        tasks["easy"].append(_task("coding", "easy", i, q, a))
    for i, (q, a) in enumerate(medium_q):
        tasks["medium"].append(_task("coding", "medium", i, q, a))
    for i, (q, a) in enumerate(hard_q):
        tasks["hard"].append(_task("coding", "hard", i, q, a,
                                   aliases=[a, a.lower()]))
    return tasks


def _load_creative():
    tasks = {"easy": [], "medium": [], "hard": []}
    easy_q = [
        ("What rhymes with 'cat'?", "bat", ["bat","hat","mat","rat","sat","fat","pat"]),
        ("What rhymes with 'night'?", "light", ["light","right","fight","might","sight"]),
        ("What color do you get mixing red and blue?", "purple", ["purple","violet"]),
        ("What is the opposite of 'hot'?", "cold", ["cold","cool","frigid"]),
        ("Name an animal that lives in the ocean.", "whale", ["whale","shark","dolphin","fish","octopus"]),
    ]
    medium_q = [
        ("What is a word meaning 'happy' that starts with J?", "joyful", ["joyful","jovial","jubilant"]),
        ("Name a synonym for 'large' starting with 'G'.", "gigantic", ["gigantic","grand","great"]),
        ("What poetic device is used in 'the wind whispered'?", "personification", ["personification"]),
    ]
    hard_q = [
        ("Name the literary device where a part represents the whole.", "synecdoche", ["synecdoche"]),
        ("What is a nine-line poem with specific rhyme scheme called?", "spenserian sonnet", ["spenserian sonnet","spenserian"]),
        ("What rhetorical device uses 'but wait' to return to an earlier point?", "analepsis", ["analepsis","flashback"]),
    ]
    for i, (q, a, al) in enumerate(easy_q):
        tasks["easy"].append(_task("creative", "easy", i, q, a, aliases=al))
    for i, (q, a, al) in enumerate(medium_q):
        tasks["medium"].append(_task("creative", "medium", i, q, a, aliases=al))
    for i, (q, a, al) in enumerate(hard_q):
        tasks["hard"].append(_task("creative", "hard", i, q, a, aliases=al))
    return tasks


# ── Synthetic fallbacks (always available) ────────────────────────────────────

def _synthetic_all() -> dict:
    return {
        "math":     _load_coding(),   # reuse as placeholder
        "logic":    {"easy": [_task("logic","easy",0,"All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?\nChoices: A: Yes | B: No | C: Maybe | D: Cannot determine","A")], "medium": [], "hard": []},
        "factual":  {"easy": [_task("factual","easy",0,"What is the capital of France?","Paris",["Paris"])], "medium": [], "hard": []},
        "science":  {"easy": [_task("science","easy",0,"What is H2O?\nChoices: A: Water | B: Salt | C: Air | D: Fire","A")], "medium": [], "hard": []},
        "medical":  {"easy": [_task("medical","easy",0,"How many chambers does the human heart have?\nChoices: A: 2 | B: 3 | C: 4 | D: 6","C")], "medium": [], "hard": []},
        "coding":   _load_coding(),
        "creative": _load_creative(),
    }


# ── Adversarial bank ──────────────────────────────────────────────────────────

_ADVERSARIAL = [
    _task("factual","hard",9001,"How many bones does an adult human body have?","206",["206"],"adversarial"),
    _task("factual","hard",9002,"What is the capital of Australia?","Canberra",["Canberra"],"adversarial"),
    _task("math","hard",9003,"A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?","0.05",["0.05","5 cents","$0.05"],"adversarial"),
    _task("factual","hard",9004,"In what year did the Berlin Wall fall?","1989",["1989"],"adversarial"),
    _task("science","hard",9005,"What is the boiling point of water at sea level in Celsius?","100",["100","100°C"],"adversarial"),
    _task("math","hard",9006,"If you have 3 apples and take away 2, how many do you have?","2",["2"],"adversarial"),
    _task("factual","hard",9007,"Who wrote Hamlet?","William Shakespeare",["William Shakespeare","Shakespeare"],"adversarial"),
    _task("science","hard",9008,"How many planets are in our solar system?","8",["8"],"adversarial"),
    _task("coding","hard",9009,"What does the following return: not not True","True",["True"],"adversarial"),
    _task("math","hard",9010,"What is 15% of 200?","30",["30"],"adversarial"),
]


# ── TaskBank class ────────────────────────────────────────────────────────────

class TaskBank:
    """
    Manages loading, caching, and curriculum-aware sampling of tasks
    across 7 domains and 3 difficulty levels.
    """

    def __init__(self, data_dir: str = cfg.DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, dict[str, list]] = {
            d: {"easy": [], "medium": [], "hard": []} for d in cfg.DOMAINS
        }
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def download_all(self) -> None:
        """Download all datasets and cache to data/tasks_cache.json."""
        loaders = {
            "math": _load_math, "logic": _load_logic, "factual": _load_factual,
            "science": _load_science, "medical": _load_medical,
            "coding": _load_coding, "creative": _load_creative,
        }
        for domain, loader in loaders.items():
            logger.info("Loading %s…", domain)
            try:
                self._tasks[domain] = loader()
            except Exception as exc:
                logger.warning("%s load failed: %s — using synthetic", domain, exc)
                synth = _synthetic_all()
                self._tasks[domain] = synth.get(domain, {"easy": [], "medium": [], "hard": []})
        self._loaded = True
        self._save_cache()

    def load_all(self) -> None:
        """Load from cache or fall back to synthetic."""
        if self._try_load_cache():
            return
        logger.warning("No cache — using synthetic tasks. Run download_all() for full data.")
        synth = _synthetic_all()
        for domain in cfg.DOMAINS:
            self._tasks[domain] = synth.get(domain, {"easy": [], "medium": [], "hard": []})
        # Also load coding and creative (always available)
        self._tasks["coding"]   = _load_coding()
        self._tasks["creative"] = _load_creative()
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_all()

    def get_task(
        self, domain: str, difficulty: str, exclude_ids: list[str] = []
    ) -> dict:
        """Return a random task from the given domain and difficulty."""
        self.ensure_loaded()
        pool = self._tasks.get(domain, {}).get(difficulty, [])
        if not pool:
            pool = list(_synthetic_all().get(domain, {}).get(difficulty, []))
        if not pool:
            pool = list(_synthetic_all()["coding"]["easy"])
        available = [t for t in pool if t["id"] not in exclude_ids]
        return dict(random.choice(available if available else pool))

    def get_batch(
        self, n: int, phase: int, mix_ratios: Optional[dict] = None
    ) -> list[dict]:
        """Return n tasks for the given curriculum phase."""
        self.ensure_loaded()
        if mix_ratios is None:
            mix_ratios = [cfg.PHASE_1_MIX, cfg.PHASE_2_MIX, cfg.PHASE_3_MIX][phase - 1]
        domains = cfg.DOMAINS
        batch = []
        for _ in range(n):
            r = random.random()
            cum = 0.0
            chosen_diff = "easy"
            for diff in ["easy", "medium", "hard"]:
                cum += mix_ratios.get(diff, 0.0)
                if r <= cum:
                    chosen_diff = diff
                    break
            domain = random.choice(domains)
            batch.append(self.get_task(domain, chosen_diff))
        return batch

    def get_adversarial_batch(self, n: int) -> list[dict]:
        """Return n adversarial tasks designed to trigger overconfidence."""
        self.ensure_loaded()
        pool = list(_ADVERSARIAL)
        if not pool:
            return self.get_batch(n, phase=3)
        return [dict(random.choice(pool)) for _ in range(n)]

    def stats(self) -> None:
        """Print domain × difficulty × count table."""
        self.ensure_loaded()
        header = f"{'Domain':<12}" + "".join(f"  {d:<8}" for d in cfg.DIFFICULTIES) + "  Total"
        print(header)
        print("─" * len(header))
        for domain in cfg.DOMAINS:
            counts = {d: len(self._tasks[domain][d]) for d in cfg.DIFFICULTIES}
            row = f"{domain:<12}" + "".join(f"  {counts[d]:<8}" for d in cfg.DIFFICULTIES)
            row += f"  {sum(counts.values())}"
            print(row)

    def get_task_by_id(self, task_id: str) -> Optional[dict]:
        self.ensure_loaded()
        for domain in cfg.DOMAINS:
            for diff in cfg.DIFFICULTIES:
                for t in self._tasks[domain][diff]:
                    if t["id"] == task_id:
                        return dict(t)
        return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _save_cache(self) -> None:
        cache = Path(cfg.TASKS_CACHE)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(self._tasks, f)
        logger.info("Saved task cache → %s", cache)

    def _try_load_cache(self) -> bool:
        cache = Path(cfg.TASKS_CACHE)
        if not cache.exists():
            return False
        try:
            with open(cache) as f:
                self._tasks = json.load(f)
            self._loaded = True
            logger.info("Loaded task bank from cache")
            return True
        except Exception as exc:
            logger.warning("Cache load failed: %s", exc)
            return False
