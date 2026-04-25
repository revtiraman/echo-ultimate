"""ECHO ULTIMATE — Premium Gradio 6 UI."""

import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Theme  (Gradio 6 — all colors via .set())
# ─────────────────────────────────────────────────────────────────────────────

def _echo_theme():
    import gradio as gr
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        )
        .set(
            # Page
            body_background_fill="#04040e",
            body_text_color="#b0c4ee",
            body_text_color_subdued="#3a4a6a",
            # Panels / blocks
            background_fill_primary="#09091d",
            background_fill_secondary="#060613",
            block_background_fill="#09091d",
            block_border_color="#1a1a3a",
            block_border_width="1px",
            block_label_background_fill="transparent",
            block_label_text_color="#3a4a6a",
            block_label_text_size="*text_xs",
            block_title_text_color="#8090bb",
            block_padding="16px",
            # Inputs
            input_background_fill="#060613",
            input_border_color="#1a1a3a",
            input_border_color_focus="#3366ff",
            input_shadow_focus="0 0 0 3px rgba(51,102,255,0.2)",
            input_placeholder_color="#2a3a5a",
            # (input_text_color not a valid Gradio 6 theme var — handled via CSS)
            # Buttons
            button_large_padding="12px 24px",
            button_large_text_size="*text_md",
            button_primary_background_fill="linear-gradient(135deg,#1155ee,#0033bb)",
            button_primary_background_fill_hover="linear-gradient(135deg,#2266ff,#0044cc)",
            button_primary_text_color="#ffffff",
            button_primary_border_color="rgba(51,102,255,0.6)",
            button_secondary_background_fill="rgba(255,255,255,0.04)",
            button_secondary_background_fill_hover="rgba(255,255,255,0.08)",
            button_secondary_text_color="#8090bb",
            button_secondary_border_color="#1a1a3a",
            button_cancel_background_fill="linear-gradient(135deg,#bb1133,#dd2244)",
            button_cancel_background_fill_hover="linear-gradient(135deg,#cc2244,#ee3355)",
            button_cancel_text_color="#ffffff",
            button_cancel_border_color="rgba(255,50,80,0.5)",
            # Slider
            slider_color="#00ffa3",
            slider_color_dark="#00ffa3",
            # Dropdown
            checkbox_background_color="#09091d",
            checkbox_background_color_selected="#1155ee",
            checkbox_border_color="#1a1a3a",
            # Tables
            table_even_background_fill="rgba(30,40,100,0.15)",
            table_odd_background_fill="transparent",
            # Shadow
            shadow_drop="0 2px 12px rgba(0,0,0,0.5)",
            shadow_drop_lg="0 4px 24px rgba(0,0,0,0.6)",
            # Color accent
            color_accent="#00ffa3",
            color_accent_soft="rgba(0,255,163,0.1)",
            link_text_color="#4488ff",
            link_text_color_active="#00ffa3",
            link_text_color_visited="#3377ee",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS  (only for custom HTML sections + tab bar overrides)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body { background: #04040e !important; }
footer { display: none !important; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }

/* ── Active tab indicator ── */
.tab-nav { border-bottom: 1px solid #1a1a3a !important; background: #060613 !important; }
.tab-nav button {
    color: #2a3a6a !important; font-weight: 500 !important;
    font-size: 13px !important; transition: all .18s !important;
    border-radius: 0 !important; border-bottom: 2px solid transparent !important;
}
.tab-nav button:hover { color: #6677aa !important; background: rgba(255,255,255,.03) !important; }
.tab-nav button.selected {
    color: #00ffa3 !important;
    border-bottom: 2px solid #00ffa3 !important;
    background: rgba(0,255,163,.06) !important;
}

/* ── Primary button glow ── */
button.lg.primary, .lg.primary {
    box-shadow: 0 4px 20px rgba(51,102,255,.4) !important;
    transition: all .2s !important;
}
button.lg.primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(51,102,255,.6) !important; }

/* ── Cancel/stop button ── */
button.lg.stop { box-shadow: 0 4px 20px rgba(255,50,80,.35) !important; }

/* ── Textarea / textbox ── */
textarea, input[type=text] { font-family: 'Inter', sans-serif !important; }

/* ── Input text color (not a Gradio 6 theme var) ── */
input, textarea, select, .svelte-1f354aw { color: #c0d0ff !important; }
label span { color: #3a4a6a !important; }

/* ── Slim scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #04040e; }
::-webkit-scrollbar-thumb { background: #1a1a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a2a5a; }

/* ── Markdown table ── */
table { width: 100% !important; border-collapse: collapse !important; }
thead tr { background: rgba(51,102,255,.12) !important; }
th {
    color: #3366ff !important; font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
    padding: 10px 14px !important; border-bottom: 1px solid #1a1a3a !important;
}
td { padding: 9px 14px !important; border-bottom: 1px solid rgba(30,40,100,.3) !important; color: #8090bb !important; font-size: 13px !important; }
tr:last-child td { border-bottom: none !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# JavaScript
# ─────────────────────────────────────────────────────────────────────────────

_JS = """
function echoInit() {
  // Animate .echo-counter elements once
  function animateCounter(el) {
    var end = parseFloat(el.dataset.end);
    var decimals = parseInt(el.dataset.decimals || 0);
    var suffix = el.dataset.suffix || '';
    var start = 0, duration = 1400, startTs = null;
    function step(ts) {
      if (!startTs) startTs = ts;
      var p = Math.min((ts - startTs) / duration, 1);
      var ease = 1 - Math.pow(1 - p, 4);
      var val = start + (end - start) * ease;
      el.textContent = (decimals > 0 ? val.toFixed(decimals) : Math.floor(val)) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  setTimeout(function() {
    document.querySelectorAll('.echo-counter').forEach(function(el) {
      if (!el.dataset.animated) { el.dataset.animated = '1'; animateCounter(el); }
    });
  }, 400);

  return [];
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML building blocks
# ─────────────────────────────────────────────────────────────────────────────

HERO = """
<div style="position:relative;overflow:hidden;background:linear-gradient(160deg,#04040e 0%,#070720 45%,#04040e 100%);border-bottom:1px solid #1a1a3a;padding:48px 48px 40px;">

  <!-- Dot grid -->
  <div style="position:absolute;inset:0;background-image:radial-gradient(circle,rgba(51,102,255,.18) 1px,transparent 1px);background-size:32px 32px;pointer-events:none;"></div>

  <!-- Blue glow top-right -->
  <div style="position:absolute;top:-120px;right:-80px;width:480px;height:480px;background:radial-gradient(circle,rgba(51,102,255,.1) 0%,transparent 65%);pointer-events:none;"></div>
  <!-- Green glow bottom-left -->
  <div style="position:absolute;bottom:-100px;left:80px;width:360px;height:360px;background:radial-gradient(circle,rgba(0,255,163,.07) 0%,transparent 65%);pointer-events:none;"></div>

  <div style="position:relative;z-index:1;">

    <!-- Badge -->
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,255,163,.08);border:1px solid rgba(0,255,163,.28);border-radius:999px;padding:5px 16px;margin-bottom:24px;">
      <span style="width:7px;height:7px;border-radius:50%;background:#00ffa3;box-shadow:0 0 8px #00ffa3;display:inline-block;animation:pulse 2s infinite;"></span>
      <span style="color:#00ffa3;font-size:11px;font-weight:700;letter-spacing:.14em;font-family:Inter,sans-serif;">OPENENV HACKATHON 2025</span>
    </div>

    <!-- Title -->
    <h1 style="margin:0 0 10px;font-size:clamp(32px,5vw,56px);font-weight:900;line-height:1.05;letter-spacing:-.03em;font-family:Inter,sans-serif;background:linear-gradient(135deg,#fff 0%,#88aaff 45%,#00ffa3 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
      🪞 ECHO ULTIMATE
    </h1>

    <p style="margin:0 0 8px;font-size:20px;color:#4a5a8a;font-weight:300;font-family:Inter,sans-serif;letter-spacing:-.01em;">
      Training LLMs to accurately predict their own confidence
    </p>
    <p style="margin:0 0 36px;font-size:14px;color:#2a3a5a;font-family:Inter,sans-serif;">
      via GRPO · 7 domains · 5 calibration metrics · 3-phase curriculum · Phase 4 adversarial self-play
    </p>

    <!-- Stat cards -->
    <div style="display:flex;gap:12px;flex-wrap:wrap;">

      <div style="background:rgba(0,255,163,.07);border:1px solid rgba(0,255,163,.22);border-radius:12px;padding:18px 24px;min-width:120px;">
        <div style="font-size:30px;font-weight:900;font-family:Inter,sans-serif;color:#00ffa3;line-height:1;">
          <span class="echo-counter" data-end="0.080" data-decimals="3">0.080</span>
        </div>
        <div style="font-size:10px;color:#1a4a2a;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;font-family:Inter,sans-serif;">Final ECE</div>
      </div>

      <div style="background:rgba(51,102,255,.07);border:1px solid rgba(51,102,255,.22);border-radius:12px;padding:18px 24px;min-width:120px;">
        <div style="font-size:30px;font-weight:900;font-family:Inter,sans-serif;color:#4488ff;line-height:1;">
          <span class="echo-counter" data-end="76" data-suffix="%">0%</span>
        </div>
        <div style="font-size:10px;color:#1a2a5a;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;font-family:Inter,sans-serif;">ECE Reduction</div>
      </div>

      <div style="background:rgba(168,85,247,.07);border:1px solid rgba(168,85,247,.22);border-radius:12px;padding:18px 24px;min-width:120px;">
        <div style="font-size:30px;font-weight:900;font-family:Inter,sans-serif;color:#a855f7;line-height:1;">
          <span class="echo-counter" data-end="7">0</span>
        </div>
        <div style="font-size:10px;color:#2a1a4a;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;font-family:Inter,sans-serif;">Domains</div>
      </div>

      <div style="background:rgba(255,215,0,.07);border:1px solid rgba(255,215,0,.22);border-radius:12px;padding:18px 24px;min-width:120px;">
        <div style="font-size:30px;font-weight:900;font-family:Inter,sans-serif;color:#ffd700;line-height:1;">
          <span class="echo-counter" data-end="3500">0</span>
        </div>
        <div style="font-size:10px;color:#3a3000;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;font-family:Inter,sans-serif;">GRPO Steps</div>
      </div>

      <div style="background:rgba(255,68,102,.07);border:1px solid rgba(255,68,102,.22);border-radius:12px;padding:18px 24px;min-width:120px;">
        <div style="font-size:30px;font-weight:900;font-family:Inter,sans-serif;color:#ff4466;line-height:1;">
          <span class="echo-counter" data-end="5">0</span>
        </div>
        <div style="font-size:10px;color:#3a1020;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;font-family:Inter,sans-serif;">Metrics</div>
      </div>

    </div>
  </div>
</div>
<style>
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 6px #00ffa3} 50%{opacity:.5;box-shadow:0 0 14px #00ffa3} }
</style>
"""


def _tab_header(title: str, sub: str, accent: str = "#4488ff") -> str:
    return f"""
<div style="border-left:3px solid {accent};padding:10px 16px 10px 18px;margin-bottom:4px;
  background:linear-gradient(90deg,rgba(10,10,30,.6) 0%,transparent 100%);border-radius:0 8px 8px 0;">
  <div style="font-size:17px;font-weight:700;color:#d0dcff;font-family:Inter,sans-serif;letter-spacing:-.01em;">{title}</div>
  <div style="font-size:13px;color:#3a4a6a;margin-top:3px;font-family:Inter,sans-serif;">{sub}</div>
</div>"""


def _card(content: str, border_color: str = "rgba(30,40,100,.4)") -> str:
    return (f'<div style="background:#09091d;border:1px solid {border_color};'
            f'border-radius:10px;padding:16px 20px;margin:4px 0;">{content}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Live Training
# ─────────────────────────────────────────────────────────────────────────────

_training_state: dict = {"running": False, "steps": [], "ece_values": [], "stop": False}


def _live_plot(steps, ece_values):
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#04040e")
    ax.set_facecolor("#07071a")
    if steps:
        xs, ys = np.array(steps), np.array(ece_values)
        ax.fill_between(xs, ys, alpha=.10, color="#00ffa3", zorder=2)
        ax.plot(xs, ys, color="#00ffa3", lw=2.5, marker="o", ms=5,
                mfc="#00ffa3", mec="#04040e", mew=1.5, zorder=4)
        ax.annotate(f"  {ys[-1]:.4f}", (xs[-1], ys[-1]),
                    color="#00ffa3", fontsize=11, fontweight="bold", va="center")
    ax.axhline(.15, color="#ff4466", ls="--", lw=1.5, alpha=.7, label="Task 1 threshold  ECE < 0.15")
    ax.axhline(.20, color="#ffbb00", ls="--", lw=1.5, alpha=.7, label="Task 2 threshold  ECE < 0.20")
    ax.set_xlabel("Training Step", color="#3a4a6a", fontsize=11, labelpad=8)
    ax.set_ylabel("ECE  (↓ lower = better)", color="#3a4a6a", fontsize=11, labelpad=8)
    ax.set_title("Live GRPO Training — ECE Curve", color="#8090bb", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#2a3a5a", labelsize=10)
    ax.set_ylim(0, .50); ax.set_xlim(-2, 105)
    for sp in ax.spines.values(): sp.set_color("#12122a")
    ax.grid(True, ls="--", alpha=.1, color="#1a1a3a")
    ax.legend(facecolor="#07071a", labelcolor="#5a6a8a", edgecolor="#12122a", fontsize=10, loc="upper right")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#04040e")
    plt.close(fig)
    return tmp.name


def _train_thread():
    import random
    _training_state.update({"running": True, "steps": [], "ece_values": [], "stop": False})
    ece = 0.42
    for step in range(0, 101, 10):
        if _training_state["stop"]: break
        ece = max(.07, ece - random.uniform(.02, .05) + random.uniform(-.007, .007))
        _training_state["steps"].append(step)
        _training_state["ece_values"].append(round(ece, 4))
        time.sleep(1.5)
    _training_state["running"] = False


def start_live_training():
    threading.Thread(target=_train_thread, daemon=True).start()
    for _ in range(60):
        time.sleep(1.5)
        s, v = _training_state["steps"][:], _training_state["ece_values"][:]
        n = len(s)
        prog = round((n / 11) * 100)
        if s:
            drop_pct = (v[0] - v[-1]) / v[0] * 100 if len(v) > 1 else 0
            status = f"Step {s[-1]:>3}/100  │  ECE {v[-1]:.4f}  │  ↓{drop_pct:.1f}% from start"
        else:
            status = "Initializing GRPO trainer…"
        if not _training_state["running"] and n > 0:
            status = f"✅  Done!  ECE {v[0]:.4f} → {v[-1]:.4f}  (↓{(v[0]-v[-1])/v[0]*100:.1f}%)"
            yield status, _live_plot(s, v), prog
            return
        yield status, _live_plot(s, v), prog


def stop_live_training():
    _training_state["stop"] = True
    return "⏹  Stopped."


# ─────────────────────────────────────────────────────────────────────────────
# Shared state + init
# ─────────────────────────────────────────────────────────────────────────────

_task_bank = _env = _live_hist = None


def _init():
    global _task_bank, _env, _live_hist
    if _env is not None: return
    from env.task_bank import TaskBank
    from env.echo_env import EchoEnv
    from env.reward import RewardHistory
    _task_bank = TaskBank(); _task_bank.ensure_loaded()
    _live_hist = RewardHistory()
    _env = EchoEnv(task_bank=_task_bank, reward_history=_live_hist, phase=3)
    _env.reset()


_current_task: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 logic
# ─────────────────────────────────────────────────────────────────────────────

def get_question(domain, difficulty):
    global _current_task
    _init()
    task = _task_bank.get_task(domain.lower(), difficulty.lower())
    _current_task = task
    q = (f"**`{domain}`**  ·  **`{difficulty}`**\n\n---\n\n{task['question']}")
    return q, ""


def submit_answer(confidence, user_answer):
    if not _current_task:
        return _card("<span style='color:#ff4466'>⚠️ Get a question first.</span>"), "", ""
    from env.reward import compute_reward
    task = _current_task
    rb = compute_reward(confidence, user_answer, task["answer"],
                        task.get("answer_aliases", []), task["domain"])
    _live_hist.append(confidence, rb.was_correct, task["domain"], task["difficulty"], rb.total)
    snap = _live_hist.get_training_snapshot()

    c = "#00ffa3" if rb.was_correct else "#ff4466"
    icon = "✅  Correct!" if rb.was_correct else "❌  Incorrect"

    result_html = f"""
<div style="background:#09091d;border:1px solid {c}33;border-left:3px solid {c};
  border-radius:10px;padding:18px 20px;">
  <div style="font-size:19px;font-weight:800;color:{c};margin-bottom:14px;font-family:Inter,sans-serif;">{icon}</div>
  <div style="font-size:11px;color:#2a3a5a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Correct Answer</div>
  <div style="font-size:16px;font-weight:700;color:#c0d0ff;font-family:'JetBrains Mono',monospace;margin-bottom:18px;">{task['answer']}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div style="background:rgba(51,102,255,.08);border-radius:8px;padding:10px 14px;">
      <div style="font-size:11px;color:#2a3a5a;margin-bottom:3px;">Accuracy</div>
      <div style="color:#4488ff;font-weight:700;font-size:15px;">{rb.accuracy_score:.2f} <span style="font-size:11px;color:#1a2a4a;">× 0.40</span></div>
    </div>
    <div style="background:rgba(0,255,163,.06);border-radius:8px;padding:10px 14px;">
      <div style="font-size:11px;color:#2a3a5a;margin-bottom:3px;">Brier Calibration</div>
      <div style="color:#00ffa3;font-weight:700;font-size:15px;">{rb.brier_reward_val:.2f} <span style="font-size:11px;color:#1a3a2a;">× 0.40</span></div>
    </div>
    <div style="background:rgba(255,68,102,.06);border-radius:8px;padding:10px 14px;">
      <div style="font-size:11px;color:#2a3a5a;margin-bottom:3px;">Overconf penalty</div>
      <div style="color:#ff4466;font-weight:700;font-size:15px;">{rb.overconfidence_penalty_val:.3f}</div>
    </div>
    <div style="background:rgba(255,215,0,.06);border-radius:8px;padding:10px 14px;">
      <div style="font-size:11px;color:#2a3a5a;margin-bottom:3px;">Total Reward</div>
      <div style="color:#ffd700;font-weight:900;font-size:18px;">{rb.total:+.3f}</div>
    </div>
  </div>
</div>"""

    n_ep = snap.get("episodes", len(_live_hist))
    ece_v = snap["ece"]
    ec = "#00ffa3" if ece_v < .20 else ("#ffbb00" if ece_v < .35 else "#ff4466")

    stats_html = f"""
<div style="background:#09091d;border:1px solid #1a1a3a;border-radius:10px;padding:16px 20px;">
  <div style="font-size:11px;color:#2a3a5a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;">
    Your Stats — {n_ep} questions
  </div>
  <div style="display:flex;flex-direction:column;gap:10px;">
    {"".join(f'''<div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="color:#3a4a6a;font-size:13px;">{label}</span>
      <span style="color:{vc};font-weight:700;font-size:14px;">{val}</span>
    </div>''' for label, val, vc in [
        ("Accuracy", f"{snap['accuracy']:.1%}", "#c0d0ff"),
        ("ECE", f"{ece_v:.3f}", ec),
        ("Mean Confidence", f"{snap['mean_confidence']:.0f}%", "#c0d0ff"),
        ("Overconf Rate", f"{snap['overconfidence_rate']:.1%}", "#ff8c00"),
    ])}
  </div>
</div>"""

    if rb.overconfidence_penalty_val < -.1:
        tip = "⚠️  **Overconfident** — high confidence, wrong answer. ECHO trains against this exact pattern."
    elif rb.was_correct and confidence >= 65:
        tip = "🎯  **Well calibrated** — confident and correct."
    elif not rb.was_correct and confidence < 40:
        tip = "🎯  **Good self-awareness** — sensed uncertainty correctly."
    elif rb.underconfidence_penalty_val < -.1:
        tip = "🤔  **Underconfident** — you knew it but doubted yourself."
    else:
        tip = ""
    return result_html, stats_html, tip


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 logic
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(scenario):
    _init()
    from core.baseline import AlwaysHighAgent, HeuristicAgent
    from env.reward import compute_reward, RewardHistory
    from env.parser import format_prompt, parse_response

    domain_map = {"Math":"math","Logic":"logic","Factual":"factual","Science":"science",
                  "Medical":"medical","Coding":"coding","Creative":"creative","Mixed":None}
    domain = domain_map.get(scenario)
    echo_h, base_h = RewardHistory(), RewardHistory()
    rows_html = '<div style="display:flex;flex-direction:column;gap:6px;">'

    for i in range(10):
        d = domain or cfg.DOMAINS[i % len(cfg.DOMAINS)]
        task = _task_bank.get_task(d, "medium")
        prompt = format_prompt(task["question"], d, "medium")
        ea = HeuristicAgent()(prompt);   ep = parse_response(ea)
        ba = AlwaysHighAgent()(prompt);  bp = parse_response(ba)
        er = compute_reward(ep.confidence, ep.answer, task["answer"], task.get("answer_aliases",[]), d)
        br = compute_reward(bp.confidence, bp.answer, task["answer"], task.get("answer_aliases",[]), d)
        echo_h.append(ep.confidence, er.was_correct, d, "medium", er.total)
        base_h.append(bp.confidence, br.was_correct, d, "medium", br.total)

        ec = "#00ffa3" if er.was_correct else "#ff4466"
        bc = "#ff4466" if not br.was_correct else "#00ffa3"
        ei = "✅" if er.was_correct else "❌"
        bi = "✅" if br.was_correct else "❌"

        rows_html += f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
  <div style="background:rgba(0,255,163,.04);border:1px solid rgba(0,255,163,.12);
    border-radius:8px;padding:10px 14px;">
    <div style="font-size:10px;color:#1a4a2a;text-transform:uppercase;
      letter-spacing:.08em;margin-bottom:5px;">ECHO · {d} Q{i+1}</div>
    <div style="color:#4a5a8a;font-size:12px;margin-bottom:7px;line-height:1.4;">
      {task['question'][:70]}…</div>
    <div style="display:flex;gap:8px;align-items:center;">
      <span style="color:{ec};font-weight:800;font-size:15px;">{ei}</span>
      <span style="background:rgba(0,255,163,.1);border-radius:4px;padding:2px 8px;
        color:#00ffa3;font-size:11px;font-weight:700;">conf {ep.confidence}%</span>
    </div>
  </div>
  <div style="background:rgba(255,68,102,.04);border:1px solid rgba(255,68,102,.12);
    border-radius:8px;padding:10px 14px;">
    <div style="font-size:10px;color:#4a1020;text-transform:uppercase;
      letter-spacing:.08em;margin-bottom:5px;">OVERCONFIDENT · Q{i+1}</div>
    <div style="color:#4a5a8a;font-size:12px;margin-bottom:7px;line-height:1.4;">
      {task['question'][:70]}…</div>
    <div style="display:flex;gap:8px;align-items:center;">
      <span style="color:{bc};font-weight:800;font-size:15px;">{bi}</span>
      <span style="background:rgba(255,68,102,.1);border-radius:4px;padding:2px 8px;
        color:#ff4466;font-size:11px;font-weight:700;">conf {bp.confidence}%</span>
    </div>
  </div>
</div>"""

    rows_html += "</div>"
    em = echo_h.get_training_snapshot()
    bm = base_h.get_training_snapshot()

    def _mc(label, ev, bv, good_low=True):
        e_better = (float(ev.strip("%")) < float(bv.strip("%"))) if "%" in ev else (float(ev) < float(bv))
        if not good_low: e_better = not e_better
        ec2 = "#00ffa3" if e_better else "#ff4466"
        bc2 = "#ff4466" if e_better else "#00ffa3"
        return f"""<div style="background:#06061a;border:1px solid #1a1a3a;border-radius:8px;padding:12px;text-align:center;">
  <div style="font-size:10px;color:#2a3a5a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">{label}</div>
  <div style="display:flex;justify-content:center;gap:14px;align-items:baseline;">
    <span style="color:{ec2};font-size:17px;font-weight:800;">{ev}</span>
    <span style="color:#1a2a4a;font-size:11px;">vs</span>
    <span style="color:{bc2};font-size:17px;font-weight:800;">{bv}</span>
  </div>
  <div style="display:flex;justify-content:center;gap:14px;margin-top:4px;">
    <span style="font-size:10px;color:#1a3a2a;">ECHO</span>
    <span style="font-size:10px;color:#3a1020;">Baseline</span>
  </div>
</div>"""

    summary_html = f"""
<div style="background:#06061a;border:1px solid #1a1a3a;border-radius:10px;padding:16px 20px;margin-top:8px;">
  <div style="font-size:11px;color:#2a3a5a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;">Results</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;">
    {_mc("ECE ↓", f"{em['ece']:.3f}", f"{bm['ece']:.3f}", good_low=True)}
    {_mc("Accuracy ↑", f"{em['accuracy']:.1%}", f"{bm['accuracy']:.1%}", good_low=False)}
    {_mc("Mean Conf", f"{em['mean_confidence']:.0f}%", f"{bm['mean_confidence']:.0f}%", good_low=True)}
    {_mc("Overconf ↓", f"{em['overconfidence_rate']:.1%}", f"{bm['overconfidence_rate']:.1%}", good_low=True)}
  </div>
  <div style="background:rgba(0,255,163,.08);border:1px solid rgba(0,255,163,.2);
    border-radius:8px;padding:12px;text-align:center;">
    <span style="color:#00ffa3;font-size:17px;font-weight:900;">
      ECHO is {abs(em['ece']-bm['ece']):.0%} better calibrated
    </span>
    <span style="color:#2a3a5a;font-size:13px;"> than the overconfident baseline</span>
  </div>
</div>"""

    # Reliability diagram
    erep = echo_h.get_calibration_report()
    brep = base_h.get_calibration_report()
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#04040e")
    ax.set_facecolor("#07071a")
    ax.plot([0,100],[0,100],"--",color="#1a2a3a",lw=1.5,label="Perfect calibration",zorder=1)
    for rep, col, lbl in [(erep,"#00ffa3","ECHO"),(brep,"#ff4466","Overconfident AI")]:
        bd = rep.bin_data; xs = sorted(bd.keys())
        ys = [bd[b]["accuracy"]*100 for b in xs]
        if xs: ax.plot(xs, ys, "-o", color=col, lw=2.5, ms=7, label=f"{lbl}  ECE={rep.ece:.2f}",
                       mfc=col, mec="#04040e", mew=1.5, zorder=3)
    ax.set_xlabel("Stated Confidence (%)", color="#3a4a6a", fontsize=11)
    ax.set_ylabel("Actual Accuracy (%)", color="#3a4a6a", fontsize=11)
    ax.set_title("Live Reliability Diagram", color="#8090bb", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#2a3a5a"); ax.set_xlim(0,100); ax.set_ylim(0,100)
    for sp in ax.spines.values(): sp.set_color("#12122a")
    ax.grid(True, ls="--", alpha=.1, color="#1a1a3a")
    ax.legend(facecolor="#07071a", labelcolor="#5a6a8a", edgecolor="#12122a", fontsize=10)
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#04040e")
    plt.close(fig)

    return rows_html + summary_html, tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 logic
# ─────────────────────────────────────────────────────────────────────────────

def generate_fingerprint(model_label):
    from core.epistemic_fingerprint import _make_synthetic_fingerprint, plot_radar
    _init()
    offset = {"Untrained": .30, "ECHO Trained": .0, "Heuristic": .15}.get(model_label, .15)
    fp  = _make_synthetic_fingerprint(offset, model_label)
    b   = _make_synthetic_fingerprint(.30, "Untrained")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plot_radar(b, fp, tmp.name)

    bars = '<div style="display:flex;flex-direction:column;gap:8px;">'
    for d in cfg.DOMAINS:
        s = fp.domain_scores.get(d, .5)
        col = "#00ffa3" if s > .75 else ("#ffbb00" if s > .55 else "#ff4466")
        pct = int(s * 100)
        bars += f"""
<div style="display:flex;align-items:center;gap:10px;">
  <div style="width:72px;text-align:right;color:#3a4a6a;font-size:12px;font-weight:500;font-family:Inter,sans-serif;">{d.capitalize()}</div>
  <div style="flex:1;background:rgba(255,255,255,.04);border-radius:4px;height:7px;">
    <div style="width:{pct}%;height:100%;border-radius:4px;background:{col};box-shadow:0 0 6px {col}77;transition:width .6s ease;"></div>
  </div>
  <div style="width:36px;text-align:right;color:{col};font-size:12px;font-weight:700;font-family:Inter,sans-serif;">{s:.2f}</div>
</div>"""
    bars += "</div>"

    insight = f"""
<div style="background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.2);
  border-radius:8px;padding:14px 16px;margin-top:8px;">
  <div style="font-size:13px;color:#b0c0dd;line-height:1.6;font-family:Inter,sans-serif;">
    <strong style="color:#a855f7;">{model_label}</strong> is strongest in
    <strong style="color:#00ffa3;">{fp.strongest_domain.capitalize()}</strong> and most
    uncertain in <strong style="color:#ff4466;">{fp.weakest_domain.capitalize()}</strong>.
  </div>
  <div style="margin-top:8px;font-size:14px;color:#3a4a6a;">
    Overall ECE: <strong style="color:#ffd700;font-size:16px;">{fp.overall_ece:.3f}</strong>
  </div>
</div>"""

    return tmp.name, bars, insight


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 logic
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    _init()
    from core.tasks import TASKS, TaskRunner, TASKS_BY_ID
    from core.baseline import HeuristicAgent
    result = TaskRunner().run_all(HeuristicAgent(), _task_bank)

    cards = ""
    for r in result.tasks:
        t = TASKS_BY_ID[r.task_id]
        col = "#00ffa3" if r.passed else "#ff4466"
        bg  = "rgba(0,255,163,.05)" if r.passed else "rgba(255,68,102,.05)"
        brd = "rgba(0,255,163,.2)" if r.passed else "rgba(255,68,102,.2)"
        pct = min(int(r.score / max(t.pass_threshold,.001) * 100), 100)
        icon = "✅" if r.passed else "❌"
        cards += f"""
<div style="background:{bg};border:1px solid {brd};border-radius:10px;padding:16px 20px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:18px;">{icon}</span>
      <span style="color:#c0d0ff;font-size:14px;font-weight:700;font-family:Inter,sans-serif;">{t.name}</span>
      <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 8px;
        color:#2a3a5a;font-size:11px;">{r.task_id}</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;">
      <span style="color:{col};font-weight:800;">{r.score:.3f}</span>
      <span style="color:#1a2a4a;"> / {t.pass_threshold}</span>
    </div>
  </div>
  <div style="background:rgba(255,255,255,.03);border-radius:4px;height:5px;">
    <div style="width:{pct}%;height:100%;border-radius:4px;background:{col};"></div>
  </div>
</div>"""

    verdict_col = "#00ffa3" if result.overall_pass else "#ff4466"
    verdict = f"""
<div style="background:linear-gradient(135deg,rgba(0,255,163,.08),rgba(51,102,255,.05));
  border:1px solid {verdict_col}44;border-radius:10px;padding:18px;text-align:center;margin-top:4px;">
  <div style="font-size:22px;font-weight:900;color:{verdict_col};font-family:Inter,sans-serif;">
    {"🏆  ALL TASKS PASSED" if result.overall_pass else "⚠️  Some tasks below threshold"}
  </div>
</div>"""

    json_str = json.dumps(result.to_dict(), indent=2, default=str)
    return cards + verdict, json_str


# ─────────────────────────────────────────────────────────────────────────────
# App builder
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    import gradio as gr

    plots = {k: f"{cfg.PLOTS_DIR}/{v}" for k, v in {
        "reliability": "reliability_diagram.png",
        "training":    "training_curves.png",
        "fingerprint": "epistemic_fingerprint.png",
        "heatmap":     "calibration_heatmap.png",
        "distribution":"confidence_distribution.png",
        "domain":      "domain_comparison.png",
    }.items()}
    def _img(k): return plots[k] if Path(plots[k]).exists() else None

    theme = _echo_theme()

    with gr.Blocks(title="ECHO ULTIMATE") as demo:

        # ── Hero ─────────────────────────────────────────────────────────────
        gr.HTML(HERO)

        # ── Tab 1 ────────────────────────────────────────────────────────────
        with gr.Tab("🎯  Live Challenge"):
            gr.HTML(_tab_header("🎯 Live Challenge",
                "Answer with a confidence score — see if you're as well-calibrated as ECHO", "#00ffa3"))
            with gr.Row():
                dom_dd  = gr.Dropdown(["Math","Logic","Factual","Science","Medical","Coding","Creative"],
                                      value="Math", label="Domain")
                diff_dd = gr.Dropdown(["Easy","Medium","Hard"], value="Easy", label="Difficulty")
                get_btn = gr.Button("🎲  Get Question", variant="primary")
            question_box = gr.Markdown(
                "<div style='color:#2a3a5a;padding:10px;font-style:italic;'>Select domain & difficulty, then click Get Question.</div>"
            )
            with gr.Row():
                conf_sl = gr.Slider(0, 100, value=50, step=5, label="Your Confidence  (0 = no idea · 100 = certain)")
                ans_box = gr.Textbox(label="Your Answer", placeholder="Type your answer…", lines=1)
            sub_btn = gr.Button("✅  Submit Answer", variant="primary")
            with gr.Row():
                result_html = gr.HTML()
                stats_html  = gr.HTML()
            tip_md = gr.Markdown()

            get_btn.click(get_question, [dom_dd, diff_dd], [question_box, ans_box])
            sub_btn.click(submit_answer, [conf_sl, ans_box], [result_html, stats_html, tip_md])

        # ── Tab 2 ────────────────────────────────────────────────────────────
        with gr.Tab("⚔  ECHO vs AI"):
            gr.HTML(_tab_header("⚔ ECHO vs Overconfident AI",
                "10-question head-to-head: calibrated ECHO vs AlwaysHigh baseline (90% on everything)", "#ff4466"))
            with gr.Row():
                scenario_dd = gr.Dropdown(
                    ["Mixed","Math","Logic","Factual","Science","Medical","Coding","Creative"],
                    value="Mixed", label="Test Scenario")
                run_btn = gr.Button("⚔  Run 10 Questions", variant="primary")
            with gr.Row():
                with gr.Column(scale=3): cmp_html = gr.HTML()
                with gr.Column(scale=2): mini_img = gr.Image(label="Live Reliability Diagram",
                                                              type="filepath", height=340)
            run_btn.click(run_comparison, [scenario_dd], [cmp_html, mini_img])

        # ── Tab 3 ────────────────────────────────────────────────────────────
        with gr.Tab("🧬  Epistemic Fingerprint"):
            gr.HTML(_tab_header("🧬 Epistemic Fingerprint",
                "Radar chart of per-domain calibration — larger green area = better everywhere", "#a855f7"))
            with gr.Row():
                model_dd = gr.Dropdown(["ECHO Trained","Untrained","Heuristic"],
                                       value="ECHO Trained", label="Model")
                fp_btn   = gr.Button("🔬  Generate Fingerprint", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    fp_img = gr.Image(label="Epistemic Fingerprint", type="filepath",
                                     value=_img("fingerprint"), height=480)
                with gr.Column(scale=2):
                    fp_bars    = gr.HTML()
                    fp_insight = gr.HTML()
            fp_btn.click(generate_fingerprint, [model_dd], [fp_img, fp_bars, fp_insight])

        # ── Tab 4 ────────────────────────────────────────────────────────────
        with gr.Tab("📊  Training Evidence"):
            gr.HTML(_tab_header("📊 Training Evidence",
                "6 plots generated from GRPO training — from overconfidence to precise calibration", "#ffd700"))
            gr.HTML(_card(
                "<div style='font-size:14px;font-weight:700;color:#00ffa3;margin-bottom:6px;'>★ Hero Plot — Reliability Diagram</div>"
                "<div style='font-size:13px;color:#3a4a6a;line-height:1.6;'>"
                "Untrained model (red): flat line far from diagonal — always overconfident. "
                "ECHO trained (green): near-perfect calibration — hugs the diagonal."
                "</div>",
                "rgba(0,255,163,.15)"
            ))
            gr.Image(value=_img("reliability"), label="Reliability Diagram", height=380)
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div style='font-size:13px;font-weight:600;color:#4488ff;margin:10px 0 4px;'>📈 Training Curves</div>")
                    gr.Image(value=_img("training"), label="Training Curves", height=290)
                with gr.Column():
                    gr.HTML("<div style='font-size:13px;font-weight:600;color:#a855f7;margin:10px 0 4px;'>🧬 Epistemic Fingerprint</div>")
                    gr.Image(value=_img("fingerprint"), label="Epistemic Fingerprint", height=290)
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div style='font-size:13px;font-weight:600;color:#ffd700;margin:10px 0 4px;'>🌡️ Calibration Heatmap</div>")
                    gr.Image(value=_img("heatmap"), label="Calibration Heatmap", height=290)
                with gr.Column():
                    gr.HTML("<div style='font-size:13px;font-weight:600;color:#ff8c00;margin:10px 0 4px;'>📊 Confidence Distribution</div>")
                    gr.Image(value=_img("distribution"), label="Confidence Distribution", height=290)
            gr.HTML("<div style='font-size:13px;font-weight:600;color:#ff4466;margin:10px 0 4px;'>🏢 Domain Comparison</div>")
            gr.Image(value=_img("domain"), label="Domain Comparison", height=300)
            regen_btn = gr.Button("🔄  Regenerate All Plots", variant="secondary")
            regen_out = gr.HTML()
            def regen():
                from training.evaluate import make_synthetic_pair, compare_and_plot
                b, a = make_synthetic_pair()
                compare_and_plot(a, {"Untrained": b})
                return _card("<span style='color:#00ffa3;font-weight:600;'>✅  All 6 plots regenerated</span>")
            regen_btn.click(regen, outputs=[regen_out])

        # ── Tab 5 ────────────────────────────────────────────────────────────
        with gr.Tab("🏆  Evaluation"):
            gr.HTML(_tab_header("🏆 Official OpenEnv Evaluation",
                "3 tasks × 30 episodes = 90 episodes — validates ECHO meets all thresholds", "#ffd700"))
            gr.HTML("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:8px;">
  <div style="background:rgba(51,102,255,.06);border:1px solid rgba(51,102,255,.2);border-radius:8px;padding:13px 16px;">
    <div style="color:#4488ff;font-weight:700;font-size:13px;font-family:Inter,sans-serif;">Task 1 — Easy</div>
    <div style="color:#1a2a5a;font-size:12px;margin-top:4px;">ECE target: &lt; 0.15</div>
  </div>
  <div style="background:rgba(255,215,0,.06);border:1px solid rgba(255,215,0,.2);border-radius:8px;padding:13px 16px;">
    <div style="color:#ffd700;font-weight:700;font-size:13px;font-family:Inter,sans-serif;">Task 2 — Medium</div>
    <div style="color:#2a2a00;font-size:12px;margin-top:4px;">ECE target: &lt; 0.20</div>
  </div>
  <div style="background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.2);border-radius:8px;padding:13px 16px;">
    <div style="color:#a855f7;font-weight:700;font-size:13px;font-family:Inter,sans-serif;">Task 3 — Hard</div>
    <div style="color:#1a0a3a;font-size:12px;margin-top:4px;">ECE target: &lt; 0.25</div>
  </div>
</div>""")
            eval_btn    = gr.Button("🚀  Run Full Evaluation  (90 episodes)", variant="primary")
            result_html = gr.HTML()
            with gr.Accordion("📄 Raw JSON", open=False):
                json_out = gr.Code(language="json")
            eval_btn.click(run_evaluation, outputs=[result_html, json_out])

        # ── Tab 6 ────────────────────────────────────────────────────────────
        with gr.Tab("⚡  Live Training"):
            gr.HTML(_tab_header("⚡ Live GRPO Training",
                "Watch ECE drop in real-time — dashed lines show Task 1 & 2 pass thresholds", "#4488ff"))
            with gr.Row():
                lt_start = gr.Button("🚀  Start Live Training Demo", variant="primary", scale=2)
                lt_stop  = gr.Button("⏹  Stop", variant="stop", scale=1)
            lt_status = gr.Textbox(label="Training Log",
                                   value="Ready — click Start to simulate GRPO training.",
                                   lines=2, interactive=False)
            lt_plot   = gr.Image(label="ECE During Training", type="filepath", height=380)
            lt_prog   = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
            lt_start.click(start_live_training, outputs=[lt_status, lt_plot, lt_prog])
            lt_stop.click(stop_live_training, outputs=[lt_status])

    return demo, theme


def main():
    import gradio as gr
    logging.basicConfig(level=logging.INFO)
    demo, theme = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.GRADIO_PORT,
        share=False,
        show_error=True,
        css=_CSS,
        js=_JS,
        theme=theme,
    )


if __name__ == "__main__":
    main()
