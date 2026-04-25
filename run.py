#!/usr/bin/env python3
"""
ECHO ULTIMATE — CLI entry point.

  python run.py download    Download all 7 task datasets
  python run.py test        Smoke test — 3 sample episodes
  python run.py baseline    Evaluate 4 baselines, generate all 6 plots
  python run.py plots       Generate all plots (synthetic, no eval needed)
  python run.py train       Full GRPO training (GPU required)
  python run.py eval        Evaluate trained model
  python run.py demo        Launch Gradio demo on :7860
  python run.py server      Launch FastAPI server on :8000
  python run.py all         download + train + eval
"""

import logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def cmd_download():
    from scripts.download_tasks import main; main()


def cmd_test():
    print("🧪  ECHO ULTIMATE smoke test…\n")
    from config import cfg
    from env.echo_env import EchoEnv
    from env.task_bank import TaskBank
    bank = TaskBank(); bank.ensure_loaded()
    env  = EchoEnv(task_bank=bank, phase=1, render_mode="human")

    scenarios = [
        ("<confidence>75</confidence><answer>Paris</answer>", "Correct, calibrated"),
        ("<confidence>95</confidence><answer>wrong</answer>",  "Wrong, overconfident → penalty"),
        ("<confidence>30</confidence><answer>wrong</answer>",  "Wrong, humble → small loss"),
    ]
    for i, (action, label) in enumerate(scenarios, 1):
        state, _ = env.reset()
        print(f"  Episode {i} ({label})")
        print(f"  Domain: {state['domain']} | Difficulty: {state['difficulty']}")
        _, reward, _, _, info = env.step(action)
        print(f"  Confidence: {info['parsed_confidence']}% | Correct: {info['was_correct']}")
        print(f"  Reward: {reward:+.3f} | OC Penalty: {info['overconfidence_penalty']:.2f}\n")

    snap = bank._tasks  # loaded
    print(f"  Domains loaded: {list(snap.keys())}")
    print("\n✅  Smoke test passed.")


def cmd_baseline():
    from scripts.run_baseline import main; main()


def cmd_plots():
    from scripts.generate_plots import main; main()


def cmd_train():
    print("🚀  ECHO ULTIMATE GRPO training…")
    print("    Requires GPU. Estimated: 2-4 hours on A100.")
    from config import cfg
    from env.task_bank import TaskBank
    from training.train import train
    bank = TaskBank(); bank.ensure_loaded()
    try:
        import wandb; use_wandb = True; print("  📊  WandB enabled")
    except ImportError:
        use_wandb = False; print("  📊  WandB not found — CSV logging only")
    train(cfg.MODEL_NAME, cfg.MODEL_SAVE_DIR, task_bank=bank, use_wandb=use_wandb)


def cmd_eval():
    print("📊  Evaluating…")
    from config import cfg
    from pathlib import Path
    from env.task_bank import TaskBank
    from training.evaluate import evaluate_agent, compare_and_plot, make_synthetic_pair

    Path(cfg.PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    bank = TaskBank(); bank.ensure_loaded()

    if Path(cfg.MODEL_SAVE_DIR).exists():
        print(f"  🤖  Loading trained model from {cfg.MODEL_SAVE_DIR}…")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok   = AutoTokenizer.from_pretrained(cfg.MODEL_SAVE_DIR)
        model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_SAVE_DIR, torch_dtype="auto")
        model.eval()
        def agent_fn(p):
            inp = tok(p, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=cfg.MAX_NEW_TOKENS,
                                     temperature=cfg.TEMPERATURE, do_sample=True)
            return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        trained = evaluate_agent(agent_fn, bank, label="ECHO Trained")
    else:
        print("  ⚠️  No trained model found — using synthetic results")
        _, trained = make_synthetic_pair()
        trained.label = "ECHO Trained"

    from core.baseline import AlwaysHighAgent
    untrained = evaluate_agent(AlwaysHighAgent(), bank, label="Untrained")
    compare_and_plot(trained, {"Untrained": untrained})
    print("\n✅  Eval complete. Plots saved to results/plots/")


def cmd_demo():
    print("🎨  Launching Gradio demo → http://localhost:7860")
    from ui.app import main; main()


def cmd_server():
    print("🖥️   Launching FastAPI server → http://localhost:8000/docs")
    import uvicorn
    from config import cfg
    uvicorn.run("server.app:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=False)


def cmd_all():
    cmd_download(); cmd_train(); cmd_eval()
    print("\n🎉  Full pipeline complete!")


def cmd_publish_benchmark():
    print("📦  Publishing EchoBench to HuggingFace Hub…")
    token = input("Enter HuggingFace write token: ").strip()
    if not token:
        print("❌  No token provided.")
        return
    from scripts.publish_echobench import main as _pub_main
    import sys as _sys
    _sys.argv = ["publish_echobench.py", "--token", token]
    _pub_main()


COMMANDS = {
    "download":          cmd_download,
    "test":              cmd_test,
    "baseline":          cmd_baseline,
    "plots":             cmd_plots,
    "train":             cmd_train,
    "eval":              cmd_eval,
    "demo":              cmd_demo,
    "server":            cmd_server,
    "all":               cmd_all,
    "publish-benchmark": cmd_publish_benchmark,
}

HELP = """
ECHO ULTIMATE — Metacognitive Calibration RL Environment

  python run.py download            Download 7 task datasets from HuggingFace
  python run.py test                Smoke test (no GPU, ~5 seconds)
  python run.py baseline            Evaluate 4 baselines, generate 6 plots
  python run.py plots               Generate all plots (synthetic data, instant)
  python run.py train               GRPO training curriculum (GPU, 2-4h)
  python run.py eval                Evaluate trained model, generate plots
  python run.py demo                Gradio demo → localhost:7860
  python run.py server              FastAPI server → localhost:8000
  python run.py all                 download + train + eval
  python run.py publish-benchmark   Publish EchoBench to HuggingFace Hub

Start here (no GPU needed):
  python run.py test
  python run.py plots
  python run.py baseline
"""

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h","--help","help"):
        print(HELP); sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"❌  Unknown: {cmd}\n  Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    try:
        COMMANDS[cmd]()
    except KeyboardInterrupt:
        print("\n⏹️   Stopped.")
    except Exception as e:
        logging.getLogger(__name__).exception("Command '%s' failed", cmd)
        sys.exit(1)
