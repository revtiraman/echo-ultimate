# ============================================================
# ECHO Eval v3 — CELL 1: BASELINE
# Run this cell first. After it prints "DONE", go to:
#   Run → Restart & clear cell outputs
# Then run eval_cell2_trained.py
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q",
    "transformers>=4.45.0", "peft>=0.13.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.42.0", "huggingface_hub>=0.24.0"], check=True)

import torch, json, re, os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"

# ── v3 Test Questions ─────────────────────────────────────────
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
    # tolerance-based for numeric domains
    if domain in NUMERIC_DOMAINS:
        if is_correct_numeric(pred, true):
            return True
    # substring match
    if true in pred or pred in true:
        return True
    # fraction match (e.g. "1/8")
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
    print(f"\nRunning BASELINE on {len(questions)} questions...\n")
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
        "accuracy":           float(correct.mean()),
        "avg_confidence":     float(confs.mean()),
        "ece":                float(ece),
        "overconfidence_rate": float(((confs > 0.7) & (correct == 0)).mean()),
        "n": len(results),
    }


# ── Load + run ────────────────────────────────────────────────
print("Loading baseline model (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                              device_map="auto", token=HF_TOKEN)
model.eval()

results = run_evaluation(model, tokenizer, TEST_QUESTIONS)
metrics = compute_metrics(results)

with open("/kaggle/working/baseline_results_v3.json", "w") as f:
    json.dump({"metrics": metrics, "results": results}, f, indent=2)

print(f"\n{'='*60}")
print(f"  BASELINE DONE")
print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
print(f"  ECE:       {metrics['ece']:.4f}")
print(f"  AvgConf:   {metrics['avg_confidence']*100:.1f}%")
print(f"  Overconf:  {metrics['overconfidence_rate']*100:.1f}%")
print(f"{'='*60}")
print("\n>>> RESTART KERNEL NOW, then run Cell 2 <<<")
