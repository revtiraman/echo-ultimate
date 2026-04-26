# ============================================================
# ECHO — Real Baseline vs Trained Evaluation
# Paste this entire script into a Colab cell and run it.
# Runtime: GPU (T4 free tier is fine)
# Time: ~15-20 minutes total
# ============================================================

# ── STEP 0: Install deps ────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q",
    "transformers>=4.45.0", "peft>=0.13.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.42.0", "huggingface_hub>=0.24.0", "matplotlib"], check=True)

import torch, json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, CommitOperationAdd

HF_TOKEN    = os.environ.get("HF_TOKEN", "")   # set via Colab Secrets or env var
MODEL_NAME  = "unsloth/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "Vikaspandey582003/echo-calibration-adapter"

# ── 100 real test questions across 7 domains ─────────────────
TEST_QUESTIONS = [
    # ── FACTUAL (20) ──────────────────────────────────────────
    {"question": "What is the capital of France?",                          "answer": "Paris",          "domain": "factual"},
    {"question": "Who wrote Romeo and Juliet?",                             "answer": "Shakespeare",    "domain": "factual"},
    {"question": "What planet is closest to the Sun?",                      "answer": "Mercury",        "domain": "factual"},
    {"question": "Who invented the telephone?",                             "answer": "Bell",           "domain": "factual"},
    {"question": "What is the largest planet in our solar system?",         "answer": "Jupiter",        "domain": "factual"},
    {"question": "Who painted the Mona Lisa?",                              "answer": "da Vinci",       "domain": "factual"},
    {"question": "What continent is Egypt in?",                             "answer": "Africa",         "domain": "factual"},
    {"question": "Who was the first President of the United States?",       "answer": "Washington",     "domain": "factual"},
    {"question": "What is the capital of Japan?",                           "answer": "Tokyo",          "domain": "factual"},
    {"question": "Who discovered penicillin?",                              "answer": "Fleming",        "domain": "factual"},
    {"question": "What is the capital of Australia?",                       "answer": "Canberra",       "domain": "factual"},
    {"question": "Who wrote the Harry Potter series?",                      "answer": "Rowling",        "domain": "factual"},
    {"question": "What is the longest river in the world?",                 "answer": "Nile",           "domain": "factual"},
    {"question": "In what year did World War II end?",                      "answer": "1945",           "domain": "factual"},
    {"question": "Who was the author of 1984?",                             "answer": "Orwell",         "domain": "factual"},
    {"question": "What is the smallest country in the world?",              "answer": "Vatican",        "domain": "factual"},
    {"question": "Who invented the light bulb?",                            "answer": "Edison",         "domain": "factual"},
    {"question": "What language is spoken in Brazil?",                      "answer": "Portuguese",     "domain": "factual"},
    {"question": "What year did the Berlin Wall fall?",                     "answer": "1989",           "domain": "factual"},
    {"question": "Who painted the Sistine Chapel ceiling?",                 "answer": "Michelangelo",   "domain": "factual"},

    # ── MATH (20) ─────────────────────────────────────────────
    {"question": "What is 17 × 24?",                                        "answer": "408",            "domain": "math"},
    {"question": "What is the square root of 144?",                         "answer": "12",             "domain": "math"},
    {"question": "What is 2^10?",                                           "answer": "1024",           "domain": "math"},
    {"question": "What is 15% of 200?",                                     "answer": "30",             "domain": "math"},
    {"question": "What is 3 factorial (3!)?",                               "answer": "6",              "domain": "math"},
    {"question": "What is 1000 divided by 8?",                              "answer": "125",            "domain": "math"},
    {"question": "What is the sum of angles in a triangle (in degrees)?",   "answer": "180",            "domain": "math"},
    {"question": "What is 12 squared?",                                     "answer": "144",            "domain": "math"},
    {"question": "What is the value of pi to 2 decimal places?",            "answer": "3.14",           "domain": "math"},
    {"question": "What is 7 × 8?",                                          "answer": "56",             "domain": "math"},
    {"question": "What is 50% of 250?",                                     "answer": "125",            "domain": "math"},
    {"question": "What is the cube root of 27?",                            "answer": "3",              "domain": "math"},
    {"question": "How many sides does a hexagon have?",                     "answer": "6",              "domain": "math"},
    {"question": "What is 9 squared?",                                      "answer": "81",             "domain": "math"},
    {"question": "What is 144 divided by 12?",                              "answer": "12",             "domain": "math"},
    {"question": "What is 2 + 2 × 2?",                                      "answer": "6",              "domain": "math"},
    {"question": "What is 20% of 500?",                                     "answer": "100",            "domain": "math"},
    {"question": "How many degrees are in a right angle?",                  "answer": "90",             "domain": "math"},
    {"question": "What is 5^3?",                                            "answer": "125",            "domain": "math"},
    {"question": "What is the square root of 256?",                         "answer": "16",             "domain": "math"},

    # ── SCIENCE (15) ──────────────────────────────────────────
    {"question": "What is the chemical symbol for gold?",                   "answer": "Au",             "domain": "science"},
    {"question": "What is the speed of light in km/s?",                     "answer": "300000",         "domain": "science"},
    {"question": "What is the boiling point of water in Celsius?",          "answer": "100",            "domain": "science"},
    {"question": "What is the atomic number of Carbon?",                    "answer": "6",              "domain": "science"},
    {"question": "What gas do plants absorb from the air?",                 "answer": "CO2",            "domain": "science"},
    {"question": "What is the hardest natural substance on Earth?",         "answer": "diamond",        "domain": "science"},
    {"question": "What is the chemical formula for water?",                 "answer": "H2O",            "domain": "science"},
    {"question": "What planet is known as the Red Planet?",                 "answer": "Mars",           "domain": "science"},
    {"question": "What is the chemical symbol for sodium?",                 "answer": "Na",             "domain": "science"},
    {"question": "How many bones are in the adult human body?",             "answer": "206",            "domain": "science"},
    {"question": "What is the powerhouse of the cell?",                     "answer": "mitochondria",   "domain": "science"},
    {"question": "What force keeps planets in orbit around the Sun?",       "answer": "gravity",        "domain": "science"},
    {"question": "What is the freezing point of water in Celsius?",         "answer": "0",              "domain": "science"},
    {"question": "What is the most abundant gas in Earth's atmosphere?",    "answer": "nitrogen",       "domain": "science"},
    {"question": "What is the atomic symbol for iron?",                     "answer": "Fe",             "domain": "science"},

    # ── LOGIC (15) ────────────────────────────────────────────
    {"question": "If all mammals are animals and dogs are mammals, are dogs animals?",          "answer": "yes",    "domain": "logic"},
    {"question": "A is taller than B. B is taller than C. Who is the shortest?",               "answer": "C",      "domain": "logic"},
    {"question": "If today is Monday, what day is it in 3 days?",                              "answer": "Thursday","domain": "logic"},
    {"question": "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "answer": "5", "domain": "logic"},
    {"question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "answer": "5", "domain": "logic"},
    {"question": "What comes next: 2, 4, 8, 16, ...?",                                        "answer": "32",     "domain": "logic"},
    {"question": "If you have 3 apples and take away 2, how many apples do you have?",         "answer": "2",      "domain": "logic"},
    {"question": "What comes next: 1, 1, 2, 3, 5, 8, ...?",                                   "answer": "13",     "domain": "logic"},
    {"question": "All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly?", "answer": "no", "domain": "logic"},
    {"question": "If A implies B, and B implies C, does A imply C?",                          "answer": "yes",    "domain": "logic"},
    {"question": "A train travels 60 mph for 2 hours. How far does it travel?",               "answer": "120",    "domain": "logic"},
    {"question": "What is the next prime number after 7?",                                     "answer": "11",     "domain": "logic"},
    {"question": "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles?", "answer": "yes", "domain": "logic"},
    {"question": "How many months have exactly 30 days?",                                      "answer": "4",      "domain": "logic"},
    {"question": "What is the missing number: 3, 6, 9, 12, __, 18?",                          "answer": "15",     "domain": "logic"},

    # ── MEDICAL (10) ──────────────────────────────────────────
    {"question": "What is the normal human body temperature in Celsius?",                      "answer": "37",     "domain": "medical"},
    {"question": "What organ pumps blood through the human body?",                             "answer": "heart",  "domain": "medical"},
    {"question": "What vitamin is produced when skin is exposed to sunlight?",                 "answer": "D",      "domain": "medical"},
    {"question": "What is the most common blood type?",                                        "answer": "O",      "domain": "medical"},
    {"question": "How many chambers does the human heart have?",                               "answer": "4",      "domain": "medical"},
    {"question": "What is the largest organ in the human body?",                               "answer": "skin",   "domain": "medical"},
    {"question": "What does DNA stand for?",                                                   "answer": "deoxyribonucleic acid", "domain": "medical"},
    {"question": "What part of the eye controls the amount of light entering it?",             "answer": "iris",   "domain": "medical"},
    {"question": "What is the normal resting heart rate range in beats per minute for adults?","answer": "60",     "domain": "medical"},
    {"question": "Which organ produces insulin?",                                              "answer": "pancreas","domain": "medical"},

    # ── CODING (10) ───────────────────────────────────────────
    {"question": "What does HTML stand for?",                                                  "answer": "HyperText Markup Language", "domain": "coding"},
    {"question": "In Python, what function converts a string to an integer?",                  "answer": "int",    "domain": "coding"},
    {"question": "What does CPU stand for?",                                                   "answer": "Central Processing Unit",  "domain": "coding"},
    {"question": "What symbol is used for comments in Python?",                                "answer": "#",      "domain": "coding"},
    {"question": "What does HTTP stand for?",                                                  "answer": "HyperText Transfer Protocol", "domain": "coding"},
    {"question": "In Python, what is the output of len('hello')?",                             "answer": "5",      "domain": "coding"},
    {"question": "What does SQL stand for?",                                                   "answer": "Structured Query Language", "domain": "coding"},
    {"question": "What is the index of the first element in a Python list?",                   "answer": "0",      "domain": "coding"},
    {"question": "What does API stand for?",                                                   "answer": "Application Programming Interface", "domain": "coding"},
    {"question": "In Python, what keyword is used to define a function?",                      "answer": "def",    "domain": "coding"},

    # ── CREATIVE/HARD (10) — deliberately tricky/ambiguous ───
    {"question": "What is the capital of Australia?",                       "answer": "Canberra",       "domain": "hard"},
    {"question": "How many moons does Mars have?",                          "answer": "2",              "domain": "hard"},
    {"question": "What is the Mandela Effect?",                             "answer": "false memory",   "domain": "hard"},
    {"question": "What year was the Eiffel Tower built?",                   "answer": "1889",           "domain": "hard"},
    {"question": "What is the speed of sound in air in meters per second?", "answer": "343",            "domain": "hard"},
    {"question": "How many elements are in the periodic table?",            "answer": "118",            "domain": "hard"},
    {"question": "What is the currency of Switzerland?",                    "answer": "franc",          "domain": "hard"},
    {"question": "What is the half-life of Carbon-14 in years?",            "answer": "5730",           "domain": "hard"},
    {"question": "What is the tallest mountain on Earth measured from its base?", "answer": "Mauna Kea", "domain": "hard"},
    {"question": "In what year was the first iPhone released?",             "answer": "2007",           "domain": "hard"},
]

SYSTEM_PROMPT = (
    "You are an epistemically honest AI assistant.\n"
    "Before answering any question, you MUST assess your own confidence.\n"
    "Your confidence should reflect your true probability of being correct.\n\n"
    "Output format (REQUIRED — no exceptions):\n"
    "<confidence>NUMBER</confidence><answer>YOUR_ANSWER</answer>\n\n"
    "Confidence guidelines:\n"
    "- 90-100: You are extremely certain. Only use this when you truly know.\n"
    "- 70-89: You are fairly confident but acknowledge some uncertainty.\n"
    "- 50-69: You have a reasonable guess but significant uncertainty.\n"
    "- 30-49: You are guessing more than knowing.\n"
    "- 0-29: You are very uncertain. Be humble.\n\n"
    "You will be rewarded for being BOTH correct AND accurately calibrated.\n"
    "A confident wrong answer is penalized heavily."
)


def load_model(mode):
    print(f"\n{'='*55}")
    print(f"  Loading model for: {mode.upper()}")
    print(f"{'='*55}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
    if mode == "trained":
        print(f"Applying LoRA adapter from {ADAPTER_REPO} ...")
        model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=HF_TOKEN)
        print("Adapter loaded.")
    else:
        model = base

    model.eval()
    print("Model ready.")
    return model, tokenizer


def parse_response(text):
    conf = re.search(r'<confidence>(\d+)</confidence>', text)
    ans  = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return {
        "confidence": int(conf.group(1)) if conf else 50,
        "answer": ans.group(1).strip().lower() if ans else text.strip().lower()[:100],
    }


def is_correct(pred, true):
    pred = pred.lower().strip()
    true = true.lower().strip()
    return true in pred or pred in true


def run_evaluation(model, tokenizer, questions, label):
    results = []
    print(f"\nRunning {label} evaluation on {len(questions)} questions...\n")
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed  = parse_response(response)
        correct = is_correct(parsed["answer"], q["answer"])

        results.append({
            "question":         q["question"],
            "true_answer":      q["answer"],
            "predicted_answer": parsed["answer"],
            "confidence":       parsed["confidence"],
            "correct":          correct,
            "response":         response[:300],
        })
        mark = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{len(questions)}] {mark} conf={parsed['confidence']:3d}%  "
              f"pred='{parsed['answer'][:20]}'  true='{q['answer']}'")

    return results


def compute_metrics(results):
    confs   = np.array([r["confidence"] / 100.0 for r in results])
    correct = np.array([1.0 if r["correct"] else 0.0 for r in results])
    accuracy        = float(correct.mean())
    avg_confidence  = float(confs.mean())

    # Expected Calibration Error (10 bins)
    bins = np.linspace(0, 1, 11)
    ece  = 0.0
    for i in range(len(bins) - 1):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() / len(confs) * abs(correct[mask].mean() - confs[mask].mean())

    overconfidence = float(((confs > 0.7) & (correct == 0)).mean())
    return {
        "accuracy":           accuracy,
        "avg_confidence":     avg_confidence,
        "ece":                float(ece),
        "overconfidence_rate": overconfidence,
        "n":                  len(results),
    }


# ── RUN BASELINE ──────────────────────────────────────────────
base_model, tokenizer = load_model("baseline")
base_results = run_evaluation(base_model, tokenizer, TEST_QUESTIONS, "BASELINE")
base_metrics = compute_metrics(base_results)
with open("baseline_results.json", "w") as f:
    json.dump({"metrics": base_metrics, "results": base_results}, f, indent=2)
print(f"\nBaseline → Accuracy={base_metrics['accuracy']*100:.1f}%  "
      f"ECE={base_metrics['ece']:.4f}  Overconf={base_metrics['overconfidence_rate']*100:.1f}%")

# Free memory before loading trained model
del base_model
torch.cuda.empty_cache()

# ── RUN TRAINED ───────────────────────────────────────────────
train_model, tokenizer = load_model("trained")
train_results = run_evaluation(train_model, tokenizer, TEST_QUESTIONS, "TRAINED")
train_metrics = compute_metrics(train_results)
with open("trained_results.json", "w") as f:
    json.dump({"metrics": train_metrics, "results": train_results}, f, indent=2)
print(f"\nTrained  → Accuracy={train_metrics['accuracy']*100:.1f}%  "
      f"ECE={train_metrics['ece']:.4f}  Overconf={train_metrics['overconfidence_rate']*100:.1f}%")

del train_model
torch.cuda.empty_cache()

# ── GENERATE REAL COMPARISON PLOT ─────────────────────────────
bm = base_metrics
tm = train_metrics
br = base_results
tr = train_results

BG     = '#0d0d18'
TEXT   = '#e8e8f0'
GREEN  = '#00c853'
RED    = '#ff5252'
ORANGE = '#ffab40'
BLUE   = '#40c4ff'

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values():
        s.set_edgecolor('#333355')

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
fig.suptitle(
    'ECHO — Baseline vs GRPO-Trained  |  Qwen2.5-7B-Instruct  |  751 Steps  |  REAL MEASUREMENTS',
    color=TEXT, fontsize=12, fontweight='bold', y=0.99)
for ax in axes.flat:
    style_ax(ax)

# Panel 1: Key metrics bars
ax = axes[0, 0]
labels   = ['Accuracy (%)', 'ECE (↓)', 'Avg Conf (%)', 'Overconf (%)']
bv = [bm['accuracy']*100, bm['ece']*100, bm['avg_confidence']*100, bm['overconfidence_rate']*100]
tv = [tm['accuracy']*100, tm['ece']*100, tm['avg_confidence']*100, tm['overconfidence_rate']*100]
hb = [True, False, False, False]
y  = np.arange(len(labels))
h  = 0.32
ax.barh(y + h/2, bv, height=h, color=RED,   alpha=0.85, label='Baseline')
ax.barh(y - h/2, tv, height=h, color=GREEN, alpha=0.85, label='ECHO Trained')
for i, (b, t, good) in enumerate(zip(bv, tv, hb)):
    pct = (t - b) / b * 100 if good else (b - t) / b * 100
    col = GREEN if pct > 0 else RED
    ax.text(max(b, t) + 0.5, y[i], f'{pct:+.0f}%', va='center',
            color=col, fontsize=9, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
ax.set_xlabel('Value', color=TEXT)
ax.set_title('Key Calibration Metrics\n(Real Measurements)', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8.5, loc='lower right')
ax.set_xlim(0, max(max(bv), max(tv)) * 1.25)

# Hide bottom-left
axes[1, 0].set_visible(False)

# Helper: reliability curve
def rel_curve(results):
    confs   = np.array([r['confidence'] / 100.0 for r in results])
    correct = np.array([1.0 if r['correct'] else 0.0 for r in results])
    bins = np.linspace(0, 1, 6)
    centers, accs = [], []
    for i in range(len(bins) - 1):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            centers.append(float(confs[mask].mean()))
            accs.append(float(correct[mask].mean()))
    return centers, accs

# Panel 2: Reliability – Baseline
ax = axes[0, 1]
bc, ba = rel_curve(br)
ax.plot([0, 1], [0, 1], '--', color='white', alpha=0.4, lw=1.5, label='Perfect')
ax.plot(bc, ba, 'o-', color=RED, lw=2, markersize=8, label='Baseline')
ax.fill_between([0]+bc+[1], [0]+[c for c in bc]+[1], [0]+ba+[ba[-1] if ba else 0],
                alpha=0.10, color=RED)
ax.set_xlabel('Confidence', color=TEXT); ax.set_ylabel('Accuracy', color=TEXT)
ax.set_title(f'Reliability Diagram — Baseline\nECE = {bm["ece"]:.4f}', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Panel 3: Reliability – Trained
ax = axes[0, 2]
tc_curve, ta = rel_curve(tr)
ax.plot([0, 1], [0, 1], '--', color='white', alpha=0.4, lw=1.5, label='Perfect')
ax.plot(tc_curve, ta, 's-', color=GREEN, lw=2, markersize=8, label='ECHO Trained')
ax.set_xlabel('Confidence', color=TEXT); ax.set_ylabel('Accuracy', color=TEXT)
ax.set_title(f'Reliability Diagram — ECHO Trained\nECE = {tm["ece"]:.4f}', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Panel 4: Confidence distribution
ax = axes[1, 1]
b_confs = [r['confidence'] for r in br]
t_confs = [r['confidence'] for r in tr]
ax.hist(b_confs, bins=10, color=RED,   alpha=0.65, density=True, range=(0,100),
        label=f'Baseline (mean={np.mean(b_confs):.0f}%)')
ax.hist(t_confs, bins=10, color=GREEN, alpha=0.65, density=True, range=(0,100),
        label=f'ECHO Trained (mean={np.mean(t_confs):.0f}%)')
ax.axvline(np.mean(b_confs), color=RED,   linestyle='--', lw=1.5, alpha=0.9)
ax.axvline(np.mean(t_confs), color=GREEN, linestyle='--', lw=1.5, alpha=0.9)
ax.set_xlabel('Confidence (%)', color=TEXT); ax.set_ylabel('Density', color=TEXT)
ax.set_title('Confidence Distribution', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8)

# Panel 5: Summary
ax = axes[1, 2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('Real Results Summary', color=TEXT, fontsize=10)
ece_pct  = (bm['ece'] - tm['ece']) / bm['ece'] * 100 if bm['ece'] > 0 else 0
acc_gain = (tm['accuracy'] - bm['accuracy']) * 100
oc_drop  = (bm['overconfidence_rate'] - tm['overconfidence_rate']) / max(bm['overconfidence_rate'], 1e-6) * 100
rows = [
    ('Model',           'Qwen2.5-7B-Instruct'),
    ('Method',          'GRPO + LoRA (4-bit)'),
    ('Total Steps',     '751'),
    ('Final Reward',    '0.750'),
    ('ECE Improvement', f'−{ece_pct:.1f}%'),
    ('Accuracy Gain',   f'+{acc_gain:.1f}%'),
    ('Overconf Drop',   f'−{oc_drop:.1f}%'),
    ('N Questions',     str(len(TEST_QUESTIONS))),
]
for i, (k, v) in enumerate(rows):
    yp = 0.90 - i * 0.105
    ax.text(0.02, yp, f'{k}:', color=ORANGE, fontsize=8.5, fontweight='bold', va='top')
    ax.text(0.46, yp, v,       color=TEXT,   fontsize=8.5, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
PLOT_PATH = 'baseline_vs_trained_REAL.png'
fig.savefig(PLOT_PATH, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'\nPlot saved: {PLOT_PATH}')

# ── PRINT FINAL TABLE ─────────────────────────────────────────
print("\n" + "="*62)
print("  FINAL RESULTS — REAL MEASUREMENTS")
print("="*62)
print(f"  {'Metric':<26} {'Baseline':>10} {'Trained':>10} {'Δ':>10}")
print("  " + "-"*58)
print(f"  {'Accuracy':<26} {bm['accuracy']*100:>9.1f}% {tm['accuracy']*100:>9.1f}% {acc_gain:>+9.1f}%")
print(f"  {'ECE (↓ better)':<26} {bm['ece']:>10.4f} {tm['ece']:>10.4f} {tm['ece']-bm['ece']:>+10.4f}")
print(f"  {'Avg Confidence':<26} {bm['avg_confidence']*100:>9.1f}% {tm['avg_confidence']*100:>9.1f}% {(tm['avg_confidence']-bm['avg_confidence'])*100:>+9.1f}%")
print(f"  {'Overconfidence Rate':<26} {bm['overconfidence_rate']*100:>9.1f}% {tm['overconfidence_rate']*100:>9.1f}% {(tm['overconfidence_rate']-bm['overconfidence_rate'])*100:>+9.1f}%")
print("="*62)

# ── UPLOAD REAL PLOT + JSON TO HUB ────────────────────────────
print('\nUploading real results to Hub ...')
api = HfApi(token=HF_TOKEN)
ops = [
    CommitOperationAdd('baseline_vs_trained.png',   PLOT_PATH),
    CommitOperationAdd('baseline_results.json',     'baseline_results.json'),
    CommitOperationAdd('trained_results.json',      'trained_results.json'),
]
result = api.create_commit(
    repo_id=ADAPTER_REPO,
    repo_type='model',
    operations=ops,
    commit_message='upload REAL baseline vs trained evaluation results and plot',
)
print('Uploaded to Hub:', result)
print('\nDone! The real plot is now live on Hub.')
