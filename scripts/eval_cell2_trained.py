# ============================================================
# ECHO Eval v3 — CELL 2: TRAINED + PLOT + UPLOAD
# Run ONLY after restarting the kernel from Cell 1.
# baseline_results_v3.json must exist in /kaggle/working/
# ============================================================
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

# ── Same test questions as Cell 1 ─────────────────────────────
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


def is_correct_numeric(pred_text, true_value, tol_pct=2.0):
    m = re.search(r'-?\d+\.?\d*', pred_text.replace(',', ''))
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
    pred = pred_answer.lower().strip()
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


def run_evaluation(model, tokenizer, questions):
    results = []
    print(f"\nRunning TRAINED on {len(questions)} questions...\n")
    for i, q in enumerate(questions):
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
        correct  = is_correct(parsed["answer"], q["answer"], q.get("domain",""))
        results.append({
            "question": q["question"], "true_answer": q["answer"],
            "predicted_answer": parsed["answer"], "confidence": parsed["confidence"],
            "correct": correct, "domain": q.get("domain",""),
        })
        mark = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{len(questions)}] {mark} conf={parsed['confidence']:3d}%  "
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
    domains = sorted(set(r["domain"] for r in results))
    out = {}
    for d in domains:
        sub = [r for r in results if r["domain"] == d]
        out[d] = compute_metrics(sub)
    return out


# ── Load baseline from Cell 1 ─────────────────────────────────
with open("/kaggle/working/baseline_results_v3.json") as f:
    base_data = json.load(f)
bm = base_data["metrics"]
br = base_data["results"]
print(f"Loaded baseline: Acc={bm['accuracy']*100:.1f}%  ECE={bm['ece']:.4f}")

# ── Load + run trained ────────────────────────────────────────
print("\nLoading trained model (4-bit + LoRA adapter)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                   device_map="auto", token=HF_TOKEN)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
model.eval()
print("Adapter loaded.")

tr = run_evaluation(model, tokenizer, TEST_QUESTIONS)
tm = compute_metrics(tr)

with open("/kaggle/working/trained_results_v3.json", "w") as f:
    json.dump({"metrics": tm, "results": tr}, f, indent=2)

# ── Domain breakdown ──────────────────────────────────────────
b_domain = domain_breakdown(br)
t_domain = domain_breakdown(tr)

print(f"\n{'='*68}")
print(f"  FINAL RESULTS — v3 HARD SET (Real Measurements)")
print(f"{'='*68}")
print(f"  {'Metric':<28} {'Baseline':>10} {'Trained':>10} {'Δ':>12}")
print(f"  {'-'*62}")
print(f"  {'Accuracy':<28} {bm['accuracy']*100:>9.1f}% {tm['accuracy']*100:>9.1f}% {(tm['accuracy']-bm['accuracy'])*100:>+10.1f}%")
print(f"  {'ECE (↓ better)':<28} {bm['ece']:>10.4f} {tm['ece']:>10.4f} {tm['ece']-bm['ece']:>+12.4f}")
print(f"  {'Avg Confidence':<28} {bm['avg_confidence']*100:>9.1f}% {tm['avg_confidence']*100:>9.1f}% {(tm['avg_confidence']-bm['avg_confidence'])*100:>+10.1f}%")
print(f"  {'Overconfidence Rate':<28} {bm['overconfidence_rate']*100:>9.1f}% {tm['overconfidence_rate']*100:>9.1f}% {(tm['overconfidence_rate']-bm['overconfidence_rate'])*100:>+10.1f}%")
print(f"{'='*68}")
print(f"\n  Domain breakdown:")
for d in sorted(b_domain.keys()):
    ba, ta = b_domain[d]['accuracy'], t_domain[d]['accuracy']
    be, te = b_domain[d]['ece'],      t_domain[d]['ece']
    print(f"  {d:<22}  acc {ba*100:.0f}%→{ta*100:.0f}%  ECE {be:.3f}→{te:.3f}")

# ── Generate plot ─────────────────────────────────────────────
BG, TEXT   = '#0d0d18', '#e8e8f0'
GREEN, RED = '#00c853', '#ff5252'
ORANGE     = '#ffab40'
BLUE       = '#40c4ff'

def style_ax(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=TEXT)
    for s in ax.spines.values(): s.set_edgecolor('#333355')

fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=BG)
fig.suptitle(
    'ECHO — Baseline vs GRPO-Trained  |  Qwen2.5-7B  |  751 Steps  |  Hard Calibration Set (40Q, 5 failure modes)',
    color=TEXT, fontsize=11, fontweight='bold', y=0.99)
for ax in axes.flat: style_ax(ax)

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
    ax.text(max(b,t)+0.2, y[i], f'{pct:+.1f}%', va='center', color=col, fontsize=9, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=9)
ax.set_title('Overall Calibration Metrics', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8.5, loc='lower right')
ax.set_xlim(0, max(max(bv),max(tv))*1.35)
axes[1,0].set_visible(False)

# Panel 2 & 3: Reliability diagrams
def rel_curve(results):
    confs   = np.array([r['confidence']/100.0 for r in results])
    correct = np.array([1.0 if r['correct'] else 0.0 for r in results])
    centers, accs, cnts = [], [], []
    for lo in np.linspace(0, 0.9, 10):
        mask = (confs>=lo)&(confs<lo+0.1)
        if mask.sum()>0:
            centers.append(float(confs[mask].mean()))
            accs.append(float(correct[mask].mean()))
            cnts.append(int(mask.sum()))
    return centers, accs, cnts

for ax, results, color, title, metrics in [
    (axes[0,1], br, RED,   f'Baseline  ECE={bm["ece"]:.4f}  Acc={bm["accuracy"]*100:.1f}%', bm),
    (axes[0,2], tr, GREEN, f'ECHO Trained  ECE={tm["ece"]:.4f}  Acc={tm["accuracy"]*100:.1f}%', tm),
]:
    c, a, cnt = rel_curve(results)
    ax.plot([0,1],[0,1],'--',color='white',alpha=0.4,lw=1.5)
    ax.bar(c, a, width=0.08, color=color, alpha=0.78)
    for x,y_,n in zip(c,a,cnt):
        ax.text(x, min(y_+0.03,0.97), str(n), ha='center', color=TEXT, fontsize=7)
    ax.set_xlabel('Confidence',color=TEXT); ax.set_ylabel('Accuracy',color=TEXT)
    ax.set_title(f'Reliability Diagram\n{title}', color=TEXT, fontsize=10)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)

# Panel 4: Domain accuracy comparison
ax = axes[1,1]
domains = sorted(b_domain.keys())
x = np.arange(len(domains)); h = 0.32
ba_vals = [b_domain[d]['accuracy']*100 for d in domains]
ta_vals = [t_domain[d]['accuracy']*100 for d in domains]
ax.barh(x+h/2, ba_vals, height=h, color=RED,   alpha=0.85, label='Baseline')
ax.barh(x-h/2, ta_vals, height=h, color=GREEN, alpha=0.85, label='ECHO Trained')
ax.set_yticks(x)
ax.set_yticklabels([d.replace('_',' ') for d in domains], color=TEXT, fontsize=8)
ax.set_xlabel('Accuracy (%)', color=TEXT)
ax.set_title('Accuracy by Domain / Failure Mode', color=TEXT, fontsize=10)
ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor=TEXT, fontsize=8)
ax.set_xlim(0, 115)

# Panel 5: Summary
ax = axes[1,2]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_title('Summary', color=TEXT, fontsize=10)
ece_pct  = (bm['ece']-tm['ece'])/bm['ece']*100 if bm['ece']>0 else 0
acc_gain = (tm['accuracy']-bm['accuracy'])*100
oc_drop  = (bm['overconfidence_rate']-tm['overconfidence_rate'])/max(bm['overconfidence_rate'],1e-6)*100
rows = [
    ('Model',         'Qwen2.5-7B-Instruct'),
    ('Method',        'GRPO + LoRA (4-bit)'),
    ('Training Steps','751'),
    ('Final Reward',  '0.750  (start: 0.150)'),
    ('Test Set',      '40Q, 5 calibration modes'),
    ('ECE Δ',         f'{ece_pct:+.1f}%'),
    ('Accuracy Δ',    f'{acc_gain:+.1f}%'),
    ('Overconf Δ',    f'{oc_drop:+.1f}%'),
]
for i,(k,v) in enumerate(rows):
    yp = 0.90-i*0.105
    ax.text(0.02,yp,f'{k}:',color=ORANGE,fontsize=8.5,fontweight='bold',va='top')
    ax.text(0.46,yp,v,      color=TEXT,  fontsize=8.5,va='top')

plt.tight_layout(rect=[0,0,1,0.97])
PLOT = '/kaggle/working/baseline_vs_trained_v3.png'
fig.savefig(PLOT, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'\nPlot saved: {PLOT}')

# ── Upload to Hub ─────────────────────────────────────────────
print('Uploading to Hub...')
api = HfApi(token=HF_TOKEN)
ops = [
    CommitOperationAdd('baseline_vs_trained.png',        PLOT),
    CommitOperationAdd('baseline_results_v3.json',       '/kaggle/working/baseline_results_v3.json'),
    CommitOperationAdd('trained_results_v3.json',        '/kaggle/working/trained_results_v3.json'),
]
r = api.create_commit(repo_id=ADAPTER_REPO, repo_type='model', operations=ops,
    commit_message='v3 eval: hard 40Q calibration set, 5 failure modes, tolerance scoring')
print('Uploaded:', r)
print('\nAll done!')
