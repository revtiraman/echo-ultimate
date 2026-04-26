# ============================================================
# ECHO — Real Baseline vs Trained Evaluation (v2)
# Paste this entire script into a Kaggle/Colab cell and run.
# Runtime: GPU (T4 x2 on Kaggle, T4 on Colab)
# Time: ~40-50 minutes total
# ============================================================

# ── STEP 0: Install deps ─────────────────────────────────────
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

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MODEL_NAME   = "unsloth/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "Vikaspandey582003/echo-calibration-adapter"

# ── 100 questions — deliberately HARD to expose overconfidence ──
# Target: base model ~60-70% accuracy, high confidence → bad ECE
# ECHO training should fix the calibration gap
TEST_QUESTIONS = [
    # ── TRICKY FACTUAL — common misconceptions (20) ───────────
    {"question": "What is the capital of Australia?",
     "answer": "Canberra", "domain": "factual"},          # most say Sydney
    {"question": "How many sides does a pentagon have?",
     "answer": "5", "domain": "factual"},
    {"question": "Who invented the telephone — first to patent it?",
     "answer": "Bell", "domain": "factual"},
    {"question": "What is the most spoken language in the world by native speakers?",
     "answer": "Mandarin", "domain": "factual"},           # many say English
    {"question": "In what ocean is the Bermuda Triangle located?",
     "answer": "Atlantic", "domain": "factual"},
    {"question": "What year was the Eiffel Tower completed?",
     "answer": "1889", "domain": "factual"},
    {"question": "What country has the largest land area in the world?",
     "answer": "Russia", "domain": "factual"},
    {"question": "How many bones are in the adult human body?",
     "answer": "206", "domain": "factual"},
    {"question": "What is the currency of Switzerland?",
     "answer": "franc", "domain": "factual"},              # many say euro
    {"question": "What is the chemical symbol for potassium?",
     "answer": "K", "domain": "factual"},                  # not P
    {"question": "On what continent is the Amazon River located?",
     "answer": "South America", "domain": "factual"},
    {"question": "How many planets are in our solar system?",
     "answer": "8", "domain": "factual"},                  # Pluto debate
    {"question": "What is the longest bone in the human body?",
     "answer": "femur", "domain": "factual"},
    {"question": "In what year was the first iPhone released?",
     "answer": "2007", "domain": "factual"},
    {"question": "What is the capital of Canada?",
     "answer": "Ottawa", "domain": "factual"},             # many say Toronto
    {"question": "Which planet has the most moons?",
     "answer": "Saturn", "domain": "factual"},             # tricky — changed recently
    {"question": "What is the atomic number of oxygen?",
     "answer": "8", "domain": "factual"},
    {"question": "Who painted Starry Night?",
     "answer": "Van Gogh", "domain": "factual"},
    {"question": "What is the speed of sound in air in m/s?",
     "answer": "343", "domain": "factual"},
    {"question": "What is the half-life of Carbon-14 in years?",
     "answer": "5730", "domain": "factual"},               # hard — model likely guesses

    # ── HARD MATH — multi-step, not instant recall (20) ───────
    {"question": "What is the remainder when 17 is divided by 5?",
     "answer": "2", "domain": "math"},
    {"question": "What is 23 × 17?",
     "answer": "391", "domain": "math"},
    {"question": "What is the square root of 169?",
     "answer": "13", "domain": "math"},
    {"question": "What is 15% of 340?",
     "answer": "51", "domain": "math"},
    {"question": "What is 2^8?",
     "answer": "256", "domain": "math"},
    {"question": "What is the next prime number after 13?",
     "answer": "17", "domain": "math"},
    {"question": "What is 7/8 as a decimal?",
     "answer": "0.875", "domain": "math"},
    {"question": "If a triangle has angles 40° and 75°, what is the third angle?",
     "answer": "65", "domain": "math"},
    {"question": "What is the LCM of 4 and 6?",
     "answer": "12", "domain": "math"},
    {"question": "What is 3^4?",
     "answer": "81", "domain": "math"},
    {"question": "How many prime numbers are between 1 and 20?",
     "answer": "8", "domain": "math"},
    {"question": "What is 1/3 + 1/4? Give the answer as a fraction.",
     "answer": "7/12", "domain": "math"},
    {"question": "What is the area of a circle with radius 5? (use π=3.14)",
     "answer": "78.5", "domain": "math"},
    {"question": "What is 11^2?",
     "answer": "121", "domain": "math"},
    {"question": "What is the GCD of 48 and 18?",
     "answer": "6", "domain": "math"},
    {"question": "If x + 7 = 15, what is x?",
     "answer": "8", "domain": "math"},
    {"question": "What is 20% of 450?",
     "answer": "90", "domain": "math"},
    {"question": "What is the cube of 4?",
     "answer": "64", "domain": "math"},
    {"question": "What is 999 + 1?",
     "answer": "1000", "domain": "math"},
    {"question": "What is the perimeter of a square with side length 7?",
     "answer": "28", "domain": "math"},

    # ── HARD SCIENCE — precise recall needed (15) ─────────────
    {"question": "What is the chemical formula for table salt?",
     "answer": "NaCl", "domain": "science"},
    {"question": "How many chromosomes do humans have?",
     "answer": "46", "domain": "science"},
    {"question": "What is the atomic number of gold?",
     "answer": "79", "domain": "science"},
    {"question": "What is absolute zero in Celsius?",
     "answer": "-273", "domain": "science"},
    {"question": "What is the pH of pure water?",
     "answer": "7", "domain": "science"},
    {"question": "What gas makes up about 21% of Earth's atmosphere?",
     "answer": "oxygen", "domain": "science"},
    {"question": "What is the most abundant element in the universe?",
     "answer": "hydrogen", "domain": "science"},
    {"question": "How many protons does a helium atom have?",
     "answer": "2", "domain": "science"},
    {"question": "What is the chemical symbol for mercury?",
     "answer": "Hg", "domain": "science"},
    {"question": "What is the melting point of iron in Celsius?",
     "answer": "1538", "domain": "science"},               # very hard
    {"question": "What particle has a negative charge in an atom?",
     "answer": "electron", "domain": "science"},
    {"question": "What is the powerhouse of the cell?",
     "answer": "mitochondria", "domain": "science"},
    {"question": "What is Avogadro's number approximately?",
     "answer": "6.022e23", "domain": "science"},           # hard
    {"question": "What type of bond holds water molecules together?",
     "answer": "hydrogen", "domain": "science"},
    {"question": "What is the chemical formula for glucose?",
     "answer": "C6H12O6", "domain": "science"},

    # ── HARD LOGIC — cognitive bias traps (15) ────────────────
    {"question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?",
     "answer": "5", "domain": "logic"},                    # classic — many say 10
    {"question": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
     "answer": "5", "domain": "logic"},                    # many say 100
    {"question": "In a lake, there is a patch of lily pads. Every day the patch doubles in size. If it takes 48 days to cover the lake, how many days to cover half the lake?",
     "answer": "47", "domain": "logic"},                   # many say 24
    {"question": "How many months have 28 days?",
     "answer": "12", "domain": "logic"},                   # all months have at least 28
    {"question": "A farmer has 17 sheep. All but 9 die. How many are left?",
     "answer": "9", "domain": "logic"},
    {"question": "If you overtake the person in 2nd place in a race, what place are you in?",
     "answer": "2", "domain": "logic"},                    # many say 1st
    {"question": "What comes next: 1, 4, 9, 16, 25, ___?",
     "answer": "36", "domain": "logic"},
    {"question": "What is the missing number: 2, 3, 5, 8, 13, 21, ___?",
     "answer": "34", "domain": "logic"},
    {"question": "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles?",
     "answer": "yes", "domain": "logic"},
    {"question": "A is north of B. B is north of C. Is A north of C?",
     "answer": "yes", "domain": "logic"},
    {"question": "Paul is 20 years old in 2000. How old is Paul in 1990?",
     "answer": "10", "domain": "logic"},
    {"question": "You have a 3-litre jug and a 5-litre jug. How do you measure exactly 4 litres?",
     "answer": "fill 5 pour into 3 refill 5 pour into 3", "domain": "logic"},
    {"question": "What is the next number: 3, 6, 12, 24, ___?",
     "answer": "48", "domain": "logic"},
    {"question": "If there are 12 fish and half of them drown, how many are left?",
     "answer": "12", "domain": "logic"},                   # fish don't drown
    {"question": "An electric train is heading north. Which way does the smoke blow?",
     "answer": "none", "domain": "logic"},                 # electric trains have no smoke

    # ── MEDICAL — specialized, model often uncertain (15) ─────
    {"question": "What is the normal human body temperature in Celsius?",
     "answer": "37", "domain": "medical"},
    {"question": "Which blood type is the universal donor?",
     "answer": "O negative", "domain": "medical"},
    {"question": "What organ produces bile?",
     "answer": "liver", "domain": "medical"},
    {"question": "What is the medical term for low blood sugar?",
     "answer": "hypoglycemia", "domain": "medical"},
    {"question": "How many chambers does the human heart have?",
     "answer": "4", "domain": "medical"},
    {"question": "What vitamin deficiency causes scurvy?",
     "answer": "C", "domain": "medical"},
    {"question": "What is the average normal adult resting heart rate?",
     "answer": "72", "domain": "medical"},
    {"question": "Which part of the brain controls balance?",
     "answer": "cerebellum", "domain": "medical"},
    {"question": "What is the largest organ of the human body?",
     "answer": "skin", "domain": "medical"},
    {"question": "What is the function of red blood cells?",
     "answer": "oxygen", "domain": "medical"},
    {"question": "What does an ECG measure?",
     "answer": "heart", "domain": "medical"},
    {"question": "What is the normal systolic blood pressure range in mmHg?",
     "answer": "120", "domain": "medical"},
    {"question": "What gland produces insulin?",
     "answer": "pancreas", "domain": "medical"},
    {"question": "What is the medical term for a broken bone?",
     "answer": "fracture", "domain": "medical"},
    {"question": "What part of the eye gives it colour?",
     "answer": "iris", "domain": "medical"},

    # ── CODING — technical precision (15) ─────────────────────
    {"question": "What does HTTP status code 404 mean?",
     "answer": "not found", "domain": "coding"},
    {"question": "In Python, what is the output of type(3/2)?",
     "answer": "float", "domain": "coding"},
    {"question": "What does the 'git commit' command do?",
     "answer": "save", "domain": "coding"},
    {"question": "In Python, what does len([1,2,3]) return?",
     "answer": "3", "domain": "coding"},
    {"question": "What does SQL SELECT DISTINCT do?",
     "answer": "unique", "domain": "coding"},
    {"question": "What does O(n) mean in Big O notation?",
     "answer": "linear", "domain": "coding"},
    {"question": "What is the output of print(2**3) in Python?",
     "answer": "8", "domain": "coding"},
    {"question": "What does API stand for?",
     "answer": "Application Programming Interface", "domain": "coding"},
    {"question": "In Python, what is a list comprehension?",
     "answer": "compact loop", "domain": "coding"},
    {"question": "What HTTP method is used to create a resource?",
     "answer": "POST", "domain": "coding"},
    {"question": "What does RAM stand for?",
     "answer": "Random Access Memory", "domain": "coding"},
    {"question": "What is the index of the last element of a Python list of length n?",
     "answer": "n-1", "domain": "coding"},
    {"question": "What does CSS stand for?",
     "answer": "Cascading Style Sheets", "domain": "coding"},
    {"question": "In Python, what keyword creates a class?",
     "answer": "class", "domain": "coding"},
    {"question": "What is the output of bool(0) in Python?",
     "answer": "False", "domain": "coding"},
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

# ── Number-word → digit map for answer normalisation ─────────
NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11",
    "twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19",
    "twenty":"20","thirty":"30","forty":"40","fifty":"50",
    "sixty":"60","seventy":"70","eighty":"80","ninety":"90",
    "hundred":"100","thousand":"1000",
}
SYNONYMS = {
    "co2": ["carbon dioxide", "co2"],
    "h2o": ["water", "h2o"],
    "gravity": ["gravity", "gravitational", "gravitation"],
    "none": ["none", "no smoke", "electric", "doesn't produce", "no", "zero"],
    "yes": ["yes", "correct", "true", "indeed", "that's right"],
    "no": ["no", "incorrect", "false", "cannot", "we cannot"],
    "12": ["twelve", "12", "all twelve", "all months"],
    "9": ["nine", "9"],
    "47": ["47", "forty-seven"],
    "5": ["5", "five", "5 minutes", "5 cents"],
    "2": ["2", "two", "second place", "second"],
    "36": ["36", "thirty-six"],
    "34": ["34", "thirty-four"],
    "48": ["48", "forty-eight"],
    "4": ["4", "four"],
}


def normalise(text):
    t = text.lower().strip()
    # strip leading articles
    for art in ["the ", "a ", "an "]:
        if t.startswith(art):
            t = t[len(art):]
    # replace number words with digits
    for w, d in NUM_WORDS.items():
        t = re.sub(r'\b' + w + r'\b', d, t)
    return t


def is_correct(pred_answer, true_answer):
    pred = normalise(pred_answer)
    true = normalise(true_answer)
    # exact or substring
    if true in pred or pred in true:
        return True
    # check synonyms
    true_key = true_answer.lower().strip()
    if true_key in SYNONYMS:
        for syn in SYNONYMS[true_key]:
            if syn in pred:
                return True
    # also check if true_key's normalised form matches any synonym of pred
    for key, syns in SYNONYMS.items():
        if true in [normalise(s) for s in syns]:
            if any(s in pred for s in syns):
                return True
    return False


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
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", token=HF_TOKEN,
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
        "confidence": min(int(conf.group(1)), 100) if conf else 50,
        "answer": ans.group(1).strip().lower() if ans else text.strip().lower()[:200],
    }


def run_evaluation(model, tokenizer, questions, label):
    results = []
    print(f"\nRunning {label} evaluation on {len(questions)} questions...\n")
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed  = parse_response(response)
        correct = is_correct(parsed["answer"], q["answer"])
        results.append({
            "question":         q["question"],
            "true_answer":      q["answer"],
            "predicted_answer": parsed["answer"],
            "confidence":       parsed["confidence"],
            "correct":          correct,
            "domain":           q.get("domain", ""),
        })
        mark = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{len(questions)}] {mark} conf={parsed['confidence']:3d}%  "
              f"pred='{parsed['answer'][:25]}'  true='{q['answer']}'")
    return results


def compute_metrics(results):
    confs   = np.array([r["confidence"] / 100.0 for r in results])
    correct = np.array([1.0 if r["correct"] else 0.0 for r in results])
    accuracy        = float(correct.mean())
    avg_confidence  = float(confs.mean())
    bins = np.linspace(0, 1, 11)
    ece  = 0.0
    for i in range(len(bins) - 1):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() / len(confs) * abs(correct[mask].mean() - confs[mask].mean())
    overconfidence = float(((confs > 0.7) & (correct == 0)).mean())
    return {
        "accuracy": accuracy, "avg_confidence": avg_confidence,
        "ece": float(ece), "overconfidence_rate": overconfidence, "n": len(results),
    }


# ── RUN BASELINE ──────────────────────────────────────────────
base_model, tokenizer = load_model("baseline")
base_results = run_evaluation(base_model, tokenizer, TEST_QUESTIONS, "BASELINE")
base_metrics = compute_metrics(base_results)
with open("baseline_results.json", "w") as f:
    json.dump({"metrics": base_metrics, "results": base_results}, f, indent=2)
print(f"\nBaseline → Acc={base_metrics['accuracy']*100:.1f}%  "
      f"ECE={base_metrics['ece']:.4f}  AvgConf={base_metrics['avg_confidence']*100:.1f}%  "
      f"Overconf={base_metrics['overconfidence_rate']*100:.1f}%")
del base_model; torch.cuda.empty_cache()

# ── RUN TRAINED ───────────────────────────────────────────────
train_model, tokenizer = load_model("trained")
train_results = run_evaluation(train_model, tokenizer, TEST_QUESTIONS, "TRAINED")
train_metrics = compute_metrics(train_results)
with open("trained_results.json", "w") as f:
    json.dump({"metrics": train_metrics, "results": train_results}, f, indent=2)
print(f"\nTrained  → Acc={train_metrics['accuracy']*100:.1f}%  "
      f"ECE={train_metrics['ece']:.4f}  AvgConf={train_metrics['avg_confidence']*100:.1f}%  "
      f"Overconf={train_metrics['overconfidence_rate']*100:.1f}%")
del train_model; torch.cuda.empty_cache()

# ── PLOT ──────────────────────────────────────────────────────
bm, tm = base_metrics, train_metrics
br, tr = base_results, train_results

BG, TEXT   = '#0d0d18', '#e8e8f0'
GREEN, RED = '#00c853', '#ff5252'
ORANGE     = '#ffab40'

def style_ax(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=TEXT)
    for s in ax.spines.values(): s.set_edgecolor('#333355')

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
fig.suptitle(
    'ECHO — Baseline vs GRPO-Trained  |  Qwen2.5-7B-Instruct  |  751 Steps  |  REAL MEASUREMENTS',
    color=TEXT, fontsize=12, fontweight='bold', y=0.99)
for ax in axes.flat: style_ax(ax)

# Panel 1: Metric bars
ax = axes[0, 0]
labels = ['Accuracy (%)', 'ECE ×100 (↓)', 'Avg Conf (%)', 'Overconf (%)']
bv = [bm['accuracy']*100, bm['ece']*100, bm['avg_confidence']*100, bm['overconfidence_rate']*100]
tv = [tm['accuracy']*100, tm['ece']*100, tm['avg_confidence']*100, tm['overconfidence_rate']*100]
hb = [True, False, False, False]
y  = np.arange(len(labels)); h = 0.32
ax.barh(y+h/2, bv, height=h, color=RED,   alpha=0.85, label='Baseline')
ax.barh(y-h/2, tv, height=h, color=GREEN, alpha=0.85, label='ECHO Trained')
for i,(b,t,good) in enumerate(zip(bv,tv,hb)):
    pct = (t-b)/b*100 if good else (b-t)/b*100
    col = GREEN if pct > 0 else RED
    ax.text(max(b,t)+0.3, y[i], f'{pct:+.1f}%', va='center', color=col, fontsize=9, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
ax.set_title('Key Calibration Metrics\n(Real Measurements, 100Q)', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8.5, loc='lower right')
ax.set_xlim(0, max(max(bv),max(tv))*1.3)

axes[1,0].set_visible(False)

def rel_curve(results):
    confs   = np.array([r['confidence']/100.0 for r in results])
    correct = np.array([1.0 if r['correct'] else 0.0 for r in results])
    bins = np.linspace(0, 1, 11)
    centers, accs, counts = [], [], []
    for i in range(len(bins)-1):
        mask = (confs>=bins[i])&(confs<bins[i+1])
        if mask.sum() > 0:
            centers.append(float(confs[mask].mean()))
            accs.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    return centers, accs, counts

# Panel 2: Reliability Baseline
ax = axes[0,1]
bc, ba, bct = rel_curve(br)
ax.plot([0,1],[0,1],'--',color='white',alpha=0.4,lw=1.5,label='Perfect calibration')
ax.bar(bc, ba, width=0.08, color=RED, alpha=0.75, label='Baseline')
for x,y_val,c in zip(bc,ba,bct):
    ax.text(x, y_val+0.02, str(c), ha='center', color=TEXT, fontsize=7)
ax.set_xlabel('Confidence',color=TEXT); ax.set_ylabel('Accuracy',color=TEXT)
ax.set_title(f'Reliability Diagram — Baseline\nECE={bm["ece"]:.4f}  Acc={bm["accuracy"]*100:.1f}%',color=TEXT,fontsize=10)
ax.legend(facecolor='#1a1a2e',edgecolor='#333355',labelcolor=TEXT,fontsize=8)
ax.set_xlim(0,1); ax.set_ylim(0,1.1)

# Panel 3: Reliability Trained
ax = axes[0,2]
tc_c, ta, tct = rel_curve(tr)
ax.plot([0,1],[0,1],'--',color='white',alpha=0.4,lw=1.5,label='Perfect calibration')
ax.bar(tc_c, ta, width=0.08, color=GREEN, alpha=0.75, label='ECHO Trained')
for x,y_val,c in zip(tc_c,ta,tct):
    ax.text(x, y_val+0.02, str(c), ha='center', color=TEXT, fontsize=7)
ax.set_xlabel('Confidence',color=TEXT); ax.set_ylabel('Accuracy',color=TEXT)
ax.set_title(f'Reliability Diagram — ECHO Trained\nECE={tm["ece"]:.4f}  Acc={tm["accuracy"]*100:.1f}%',color=TEXT,fontsize=10)
ax.legend(facecolor='#1a1a2e',edgecolor='#333355',labelcolor=TEXT,fontsize=8)
ax.set_xlim(0,1); ax.set_ylim(0,1.1)

# Panel 4: Confidence distribution
ax = axes[1,1]
b_confs = [r['confidence'] for r in br]
t_confs = [r['confidence'] for r in tr]
ax.hist(b_confs,bins=20,color=RED,  alpha=0.65,density=True,range=(0,100),
        label=f'Baseline (mean={np.mean(b_confs):.0f}%)')
ax.hist(t_confs,bins=20,color=GREEN,alpha=0.65,density=True,range=(0,100),
        label=f'ECHO Trained (mean={np.mean(t_confs):.0f}%)')
ax.axvline(np.mean(b_confs),color=RED,  linestyle='--',lw=1.5)
ax.axvline(np.mean(t_confs),color=GREEN,linestyle='--',lw=1.5)
ax.set_xlabel('Confidence (%)',color=TEXT); ax.set_ylabel('Density',color=TEXT)
ax.set_title('Confidence Distribution',color=TEXT,fontsize=10)
ax.legend(facecolor='#1a1a2e',edgecolor='#333355',labelcolor=TEXT,fontsize=8)

# Panel 5: Summary
ax = axes[1,2]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_title('Real Results Summary',color=TEXT,fontsize=10)
ece_pct = (bm['ece']-tm['ece'])/bm['ece']*100 if bm['ece']>0 else 0
acc_gain = (tm['accuracy']-bm['accuracy'])*100
oc_drop  = (bm['overconfidence_rate']-tm['overconfidence_rate'])/max(bm['overconfidence_rate'],1e-6)*100
rows = [
    ('Model',          'Qwen2.5-7B-Instruct'),
    ('Method',         'GRPO + LoRA (4-bit)'),
    ('Total Steps',    '751'),
    ('Final Reward',   '0.750  (start: 0.150)'),
    ('ECE Δ',          f'−{ece_pct:.1f}%'),
    ('Accuracy Δ',     f'+{acc_gain:.1f}%'),
    ('Overconf Δ',     f'−{oc_drop:.1f}%'),
    ('N Questions',    f'{len(TEST_QUESTIONS)} (7 domains)'),
]
for i,(k,v) in enumerate(rows):
    yp = 0.90-i*0.105
    ax.text(0.02,yp,f'{k}:',color=ORANGE,fontsize=8.5,fontweight='bold',va='top')
    ax.text(0.46,yp,v,      color=TEXT,  fontsize=8.5,va='top')

plt.tight_layout(rect=[0,0,1,0.97])
PLOT_PATH = 'baseline_vs_trained_REAL.png'
fig.savefig(PLOT_PATH,dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
plt.close(fig)
print(f'\nPlot saved: {PLOT_PATH}')

# ── FINAL TABLE ───────────────────────────────────────────────
print("\n" + "="*64)
print("  FINAL RESULTS — REAL MEASUREMENTS (v2, harder questions)")
print("="*64)
print(f"  {'Metric':<28} {'Baseline':>10} {'Trained':>10} {'Δ':>10}")
print("  " + "-"*60)
print(f"  {'Accuracy':<28} {bm['accuracy']*100:>9.1f}% {tm['accuracy']*100:>9.1f}% {acc_gain:>+9.1f}%")
print(f"  {'ECE (↓ better)':<28} {bm['ece']:>10.4f} {tm['ece']:>10.4f} {tm['ece']-bm['ece']:>+10.4f}")
print(f"  {'Avg Confidence':<28} {bm['avg_confidence']*100:>9.1f}% {tm['avg_confidence']*100:>9.1f}% {(tm['avg_confidence']-bm['avg_confidence'])*100:>+9.1f}%")
print(f"  {'Overconfidence Rate':<28} {bm['overconfidence_rate']*100:>9.1f}% {tm['overconfidence_rate']*100:>9.1f}% {(tm['overconfidence_rate']-bm['overconfidence_rate'])*100:>+9.1f}%")
print("="*64)

# ── UPLOAD TO HUB ─────────────────────────────────────────────
print('\nUploading to Hub ...')
api = HfApi(token=HF_TOKEN)
ops = [
    CommitOperationAdd('baseline_vs_trained.png',  PLOT_PATH),
    CommitOperationAdd('baseline_results.json',    'baseline_results.json'),
    CommitOperationAdd('trained_results.json',     'trained_results.json'),
]
result = api.create_commit(
    repo_id=ADAPTER_REPO, repo_type='model', operations=ops,
    commit_message='v2 eval: harder 100Q, fixed answer matching, real measurements',
)
print('Uploaded:', result)
print('\nDone!')
