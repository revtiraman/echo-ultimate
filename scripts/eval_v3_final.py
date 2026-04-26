# ============================================================
# ECHO Eval v3 — ONE SCRIPT, NO OOM
# Strategy: load base model once → eval baseline → apply LoRA
# adapter on top (no reload) → eval trained → plot → upload
# ============================================================
import os
# ── SET YOUR TOKEN HERE ──────────────────────────────────────
os.environ["HF_TOKEN"] = "PASTE_YOUR_HF_TOKEN_HERE"  # replace before running

import subprocess
subprocess.run(["pip", "install", "-q",
    "transformers>=4.45.0", "peft>=0.13.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.42.0", "huggingface_hub>=0.24.0", "matplotlib"], check=True)

import torch, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, CommitOperationAdd

HF_TOKEN     = os.environ["HF_TOKEN"]
MODEL_NAME   = "unsloth/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "Vikaspandey582003/echo-calibration-adapter"
OUT_DIR      = "/kaggle/working"

# ── Test Questions (v3 — 40Q, 5 calibration failure modes) ───
TEST_QUESTIONS = [
    # 1. PRECISION-NUMERIC (8)
    {"question": "What is the boiling point of mercury in Celsius, rounded to the nearest whole degree?",
     "answer": "357", "domain": "precision_numeric"},
    {"question": "What is the boiling point of liquid nitrogen in Celsius, rounded to the nearest whole degree?",
     "answer": "-196", "domain": "precision_numeric"},
    {"question": "How many elements are currently in the periodic table?",
     "answer": "118", "domain": "precision_numeric"},
    {"question": "What is the atomic number of einsteinium?",
     "answer": "99", "domain": "precision_numeric"},
    {"question": "What is the melting point of gold in Celsius, rounded to the nearest whole degree?",
     "answer": "1064", "domain": "precision_numeric"},
    {"question": "What is the speed of light in vacuum, in km per second, rounded to the nearest thousand?",
     "answer": "300000", "domain": "precision_numeric"},
    {"question": "What is the atomic number of uranium?",
     "answer": "92", "domain": "precision_numeric"},
    {"question": "How many naturally occurring amino acids are encoded by the standard genetic code?",
     "answer": "20", "domain": "precision_numeric"},

    # 2. COUNTERINTUITIVE (8)
    {"question": "What is the largest desert in the world by area?",
     "answer": "Antarctica", "domain": "counterintuitive"},
    {"question": "Which planet in our solar system has the highest average surface temperature?",
     "answer": "Venus", "domain": "counterintuitive"},
    {"question": "What is the most abundant gas in Earth's atmosphere by volume?",
     "answer": "nitrogen", "domain": "counterintuitive"},
    {"question": "Which is the only mammal capable of true sustained flight?",
     "answer": "bat", "domain": "counterintuitive"},
    {"question": "Which country has the most time zones, including all overseas territories?",
     "answer": "France", "domain": "counterintuitive"},
    {"question": "Of Lake Superior and Lake Baikal, which holds more water by volume?",
     "answer": "Baikal", "domain": "counterintuitive"},
    {"question": "What is the smallest sovereign country in the world by land area?",
     "answer": "Vatican", "domain": "counterintuitive"},
    {"question": "Is Earth closer to the Sun at perihelion or aphelion?",
     "answer": "perihelion", "domain": "counterintuitive"},

    # 3. GPQA-LITE (8)
    {"question": "A 2 kg object falls from a height of 10 m. Ignoring air resistance and using g = 10 m/s^2, what is its kinetic energy in Joules just before impact?",
     "answer": "200", "domain": "gpqa_lite"},
    {"question": "A sound wave has a frequency of 500 Hz and travels at 340 m/s. What is its wavelength in centimetres, rounded to the nearest whole number?",
     "answer": "68", "domain": "gpqa_lite"},
    {"question": "An ideal gas at constant pressure occupies 2 litres at 300 K. What volume in litres does it occupy at 600 K?",
     "answer": "4", "domain": "gpqa_lite"},
    {"question": "A radioactive isotope has a half-life of 10 days. What fraction of the original sample remains after 30 days? Express as a simple fraction.",
     "answer": "1/8", "domain": "gpqa_lite"},
    {"question": "Two 5-ohm resistors are connected in series across a battery. What is the total resistance in ohms?",
     "answer": "10", "domain": "gpqa_lite"},
    {"question": "If the pH of a solution decreases from 7 to 4, by what factor does the hydrogen ion concentration increase?",
     "answer": "1000", "domain": "gpqa_lite"},
    {"question": "A converging lens has a focal length of 10 cm. An object is placed 20 cm from the lens. At what distance from the lens, in cm, is the image formed?",
     "answer": "20", "domain": "gpqa_lite"},
    {"question": "DNA polymerase synthesizes new DNA strands in which direction? Answer with the format X' to Y'.",
     "answer": "5' to 3'", "domain": "gpqa_lite"},

    # 4. OBSCURE-HISTORICAL (8)
    {"question": "In what year was the Peace of Westphalia signed?",
     "answer": "1648", "domain": "obscure_historical"},
    {"question": "In what year did Constantinople fall to the Ottomans?",
     "answer": "1453", "domain": "obscure_historical"},
    {"question": "Who was the first Roman emperor?",
     "answer": "Augustus", "domain": "obscure_historical"},
    {"question": "In what year was the Rosetta Stone discovered?",
     "answer": "1799", "domain": "obscure_historical"},
    {"question": "Magellan died during his expedition. Who actually completed the first circumnavigation of the globe by ship?",
     "answer": "Elcano", "domain": "obscure_historical"},
    {"question": "In what year was the Magna Carta signed?",
     "answer": "1215", "domain": "obscure_historical"},
    {"question": "Who was the first United States Secretary of the Treasury?",
     "answer": "Hamilton", "domain": "obscure_historical"},
    {"question": "In what year did Krakatoa undergo its catastrophic eruption?",
     "answer": "1883", "domain": "obscure_historical"},

    # 5. UNIT-AWARE (8)
    {"question": "What is normal human body temperature in degrees Fahrenheit?",
     "answer": "98.6", "domain": "unit_aware"},
    {"question": "What is the freezing point of water in degrees Fahrenheit at 1 atmosphere?",
     "answer": "32", "domain": "unit_aware"},
    {"question": "What is the boiling point of water in Kelvin at 1 atmosphere, rounded to the nearest whole number?",
     "answer": "373", "domain": "unit_aware"},
    {"question": "How many feet are in one statute mile?",
     "answer": "5280", "domain": "unit_aware"},
    {"question": "What is one atmosphere of pressure in pascals, rounded to the nearest hundred?",
     "answer": "101300", "domain": "unit_aware"},
    {"question": "What is the gravitational acceleration on Earth's surface in feet per second squared, rounded to the nearest whole number?",
     "answer": "32", "domain": "unit_aware"},
    {"question": "Approximately how many kilometres is the average Earth-Moon distance, rounded to the nearest thousand?",
     "answer": "384000", "domain": "unit_aware"},
    {"question": "What is absolute zero in degrees Fahrenheit, rounded to the nearest whole degree?",
     "answer": "-460", "domain": "unit_aware"},
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

NUMERIC_DOMAINS = {"precision_numeric", "unit_aware", "gpqa_lite"}


def normalise_latex(text):
    m = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', text)
    return f"{m.group(1)}/{m.group(2)}" if m else text


def is_correct_numeric(pred_text, true_value, tol_pct=2.0):
    cleaned = normalise_latex(pred_text.replace(',', ''))
    m = re.search(r'-?\d+\.?\d*', cleaned)
    if not m:
        return False
    try:
        pred = float(m.group())
        true = float(str(true_value).replace(',', ''))
        if true == 0:
            return abs(pred) < 0.01
        return abs(pred - true) / abs(true) <= tol_pct / 100.0
    except ValueError:
        return False


def is_correct(pred_answer, true_answer, domain=""):
    pred = normalise_latex(pred_answer.lower().strip())
    true = true_answer.lower().strip()
    if domain in NUMERIC_DOMAINS:
        if is_correct_numeric(pred, true):
            return True
    if true in pred or pred in true:
        return True
    if '/' in true:
        return true in pred
    return False


def parse_response(text):
    conf = re.search(r'<confidence>(\d+)</confidence>', text)
    ans  = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return {
        "confidence": min(int(conf.group(1)), 100) if conf else 50,
        "answer": ans.group(1).strip().lower() if ans else text.strip().lower()[:200],
    }


def run_evaluation(model, tokenizer, label):
    results = []
    print(f"\n{'='*55}\n  Running {label} — {len(TEST_QUESTIONS)} questions\n{'='*55}\n")
    for i, q in enumerate(TEST_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["question"]},
        ]
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed   = parse_response(response)
        correct  = is_correct(parsed["answer"], q["answer"], q.get("domain", ""))
        results.append({
            "question": q["question"], "true_answer": q["answer"],
            "predicted_answer": parsed["answer"], "confidence": parsed["confidence"],
            "correct": correct, "domain": q.get("domain", ""),
        })
        mark = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/40] {mark} conf={parsed['confidence']:3d}%  "
              f"pred='{parsed['answer'][:28]}'  true='{q['answer']}'")
    return results


def compute_metrics(results):
    confs   = np.array([r["confidence"] / 100.0 for r in results])
    correct = np.array([1.0 if r["correct"] else 0.0 for r in results])
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        mask = (confs >= lo) & (confs < lo + 0.1)
        if mask.sum() > 0:
            ece += mask.sum() / len(confs) * abs(correct[mask].mean() - confs[mask].mean())
    return {
        "accuracy":            float(correct.mean()),
        "avg_confidence":      float(confs.mean()),
        "ece":                 float(ece),
        "overconfidence_rate": float(((confs > 0.7) & (correct == 0)).mean()),
        "n": len(results),
    }


def domain_breakdown(results):
    return {d: compute_metrics([r for r in results if r["domain"] == d])
            for d in sorted(set(r["domain"] for r in results))}


# ════════════════════════════════════════════════════════════
# STEP 1 — Load base model ONCE (4-bit, ~5GB VRAM)
# ════════════════════════════════════════════════════════════
print("Loading base model in 4-bit (loads once, used for both evals)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
base_model.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ════════════════════════════════════════════════════════════
# STEP 2 — Baseline eval (raw base model, no adapter)
# ════════════════════════════════════════════════════════════
base_results = run_evaluation(base_model, tokenizer, "BASELINE")
base_metrics = compute_metrics(base_results)
with open(f"{OUT_DIR}/baseline_results_v3.json", "w") as f:
    json.dump({"metrics": base_metrics, "results": base_results}, f, indent=2)
print(f"\nBaseline → Acc={base_metrics['accuracy']*100:.1f}%  "
      f"ECE={base_metrics['ece']:.4f}  AvgConf={base_metrics['avg_confidence']*100:.1f}%  "
      f"Overconf={base_metrics['overconfidence_rate']*100:.1f}%")

# ════════════════════════════════════════════════════════════
# STEP 3 — Apply LoRA adapter ON TOP (no model reload!)
# ════════════════════════════════════════════════════════════
print(f"\nApplying LoRA adapter from {ADAPTER_REPO} ...")
trained_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
trained_model.eval()
print(f"Adapter applied. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ════════════════════════════════════════════════════════════
# STEP 4 — Trained eval (same base + adapter)
# ════════════════════════════════════════════════════════════
trained_results = run_evaluation(trained_model, tokenizer, "TRAINED")
trained_metrics = compute_metrics(trained_results)
with open(f"{OUT_DIR}/trained_results_v3.json", "w") as f:
    json.dump({"metrics": trained_metrics, "results": trained_results}, f, indent=2)
print(f"\nTrained  → Acc={trained_metrics['accuracy']*100:.1f}%  "
      f"ECE={trained_metrics['ece']:.4f}  AvgConf={trained_metrics['avg_confidence']*100:.1f}%  "
      f"Overconf={trained_metrics['overconfidence_rate']*100:.1f}%")

bm, tm = base_metrics, trained_metrics
br, tr = base_results, trained_results
b_domain = domain_breakdown(br)
t_domain = domain_breakdown(tr)

# ════════════════════════════════════════════════════════════
# STEP 5 — Print final table
# ════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print(f"  FINAL RESULTS — v3 Hard Set (40Q, 5 failure modes)")
print(f"{'='*68}")
print(f"  {'Metric':<28} {'Baseline':>10} {'Trained':>10} {'Δ':>12}")
print(f"  {'-'*62}")
acc_d  = (tm['accuracy']            - bm['accuracy'])            * 100
ece_d  =  tm['ece']                 - bm['ece']
conf_d = (tm['avg_confidence']      - bm['avg_confidence'])      * 100
oc_d   = (tm['overconfidence_rate'] - bm['overconfidence_rate']) * 100
print(f"  {'Accuracy':<28} {bm['accuracy']*100:>9.1f}% {tm['accuracy']*100:>9.1f}% {acc_d:>+10.1f}%")
print(f"  {'ECE (↓ better)':<28} {bm['ece']:>10.4f} {tm['ece']:>10.4f} {ece_d:>+12.4f}")
print(f"  {'Avg Confidence':<28} {bm['avg_confidence']*100:>9.1f}% {tm['avg_confidence']*100:>9.1f}% {conf_d:>+10.1f}%")
print(f"  {'Overconfidence Rate':<28} {bm['overconfidence_rate']*100:>9.1f}% {tm['overconfidence_rate']*100:>9.1f}% {oc_d:>+10.1f}%")
print(f"{'='*68}")
print(f"\n  Domain breakdown:")
for d in sorted(b_domain.keys()):
    ba = b_domain[d]['accuracy']*100; ta = t_domain[d]['accuracy']*100
    be = b_domain[d]['ece'];          te = t_domain[d]['ece']
    print(f"  {d:<22}  acc {ba:.0f}% → {ta:.0f}%   ECE {be:.3f} → {te:.3f}")

# ════════════════════════════════════════════════════════════
# STEP 6 — Generate plot
# ════════════════════════════════════════════════════════════
BG, TEXT   = '#0d0d18', '#e8e8f0'
GREEN, RED = '#00c853', '#ff5252'
ORANGE     = '#ffab40'

def style_ax(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=TEXT)
    for s in ax.spines.values(): s.set_edgecolor('#333355')

fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=BG)
fig.suptitle(
    'ECHO — Baseline vs GRPO-Trained  |  Qwen2.5-7B  |  751 Steps  |  40Q Hard Set (5 Calibration Failure Modes)',
    color=TEXT, fontsize=11, fontweight='bold', y=0.99)
for ax in axes.flat:
    style_ax(ax)

# Panel 1: Overall metrics
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
ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=9)
ax.set_title('Overall Calibration Metrics', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8.5, loc='lower right')
ax.set_xlim(0, max(max(bv), max(tv)) * 1.35)
axes[1, 0].set_visible(False)

# Reliability diagrams
def rel_bars(results):
    confs   = np.array([r['confidence']/100.0 for r in results])
    correct = np.array([1.0 if r['correct'] else 0.0 for r in results])
    cs, acs, ns = [], [], []
    for lo in np.linspace(0, 0.9, 10):
        mask = (confs >= lo) & (confs < lo + 0.1)
        if mask.sum() > 0:
            cs.append(float(confs[mask].mean()))
            acs.append(float(correct[mask].mean()))
            ns.append(int(mask.sum()))
    return cs, acs, ns

for ax, results, color, label, metrics in [
    (axes[0,1], br, RED,   f'Baseline  ECE={bm["ece"]:.4f}', bm),
    (axes[0,2], tr, GREEN, f'Trained   ECE={tm["ece"]:.4f}', tm),
]:
    c, a, n = rel_bars(results)
    ax.plot([0,1],[0,1],'--',color='white',alpha=0.4,lw=1.5,label='Perfect')
    ax.bar(c, a, width=0.08, color=color, alpha=0.78, label=label)
    for x, y_, cnt in zip(c, a, n):
        ax.text(x, min(y_+0.03, 0.97), str(cnt), ha='center', color=TEXT, fontsize=7)
    ax.set_xlabel('Confidence', color=TEXT); ax.set_ylabel('Accuracy', color=TEXT)
    ax.set_title(f'Reliability Diagram\n{label}  Acc={metrics["accuracy"]*100:.1f}%', color=TEXT, fontsize=10)
    ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)

# Domain accuracy
ax = axes[1, 1]
domains = sorted(b_domain.keys())
x = np.arange(len(domains)); h = 0.32
ax.barh(x+h/2, [b_domain[d]['accuracy']*100 for d in domains], height=h, color=RED,   alpha=0.85, label='Baseline')
ax.barh(x-h/2, [t_domain[d]['accuracy']*100 for d in domains], height=h, color=GREEN, alpha=0.85, label='Trained')
ax.set_yticks(x); ax.set_yticklabels([d.replace('_',' ') for d in domains], color=TEXT, fontsize=8.5)
ax.set_xlabel('Accuracy (%)', color=TEXT)
ax.set_title('Accuracy by Failure Mode', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8.5)
ax.set_xlim(0, 120)

# Summary
ax = axes[1, 2]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_title('Summary', color=TEXT, fontsize=10)
ece_pct = (bm['ece']-tm['ece'])/bm['ece']*100 if bm['ece'] > 0 else 0
rows = [
    ('Model',          'Qwen2.5-7B-Instruct'),
    ('Method',         'GRPO + LoRA (4-bit)'),
    ('Training Steps', '751'),
    ('Final Reward',   '0.750  (start: 0.150)'),
    ('Test Set',       '40Q, 5 calibration modes'),
    ('ECE Δ',          f'{ece_pct:+.1f}%'),
    ('Accuracy Δ',     f'{acc_d:+.1f}%'),
    ('Overconf Δ',     f'{oc_d:+.1f}%'),
]
for i,(k,v) in enumerate(rows):
    yp = 0.90 - i*0.105
    ax.text(0.02, yp, f'{k}:', color=ORANGE, fontsize=8.5, fontweight='bold', va='top')
    ax.text(0.46, yp, v,       color=TEXT,   fontsize=8.5, va='top')

plt.tight_layout(rect=[0,0,1,0.97])
PLOT = f"{OUT_DIR}/baseline_vs_trained_v3.png"
fig.savefig(PLOT, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'\nPlot saved: {PLOT}')

# ════════════════════════════════════════════════════════════
# STEP 7 — Upload to Hub
# ════════════════════════════════════════════════════════════
print('Uploading to Hub...')
api = HfApi(token=HF_TOKEN)
ops = [
    CommitOperationAdd('baseline_vs_trained.png',      PLOT),
    CommitOperationAdd('baseline_results_v3.json',     f"{OUT_DIR}/baseline_results_v3.json"),
    CommitOperationAdd('trained_results_v3.json',      f"{OUT_DIR}/trained_results_v3.json"),
]
r = api.create_commit(repo_id=ADAPTER_REPO, repo_type='model', operations=ops,
    commit_message='v3 eval: 40Q hard set, 5 failure modes, single-load no OOM')
print('Uploaded:', r)
print('\n✓ All done!')
