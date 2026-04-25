"""Generate all 6 publication-quality plots using synthetic data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("📊  Generating all 6 ECHO ULTIMATE plots…")
    from config import cfg
    from pathlib import Path
    Path(cfg.PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    from training.evaluate import (
        make_synthetic_pair, compare_and_plot, make_synthetic_training_log
    )
    make_synthetic_training_log(cfg.TRAINING_LOG)
    before, after = make_synthetic_pair(ece_before=0.34, ece_after=0.08)
    paths = compare_and_plot(after, {"Untrained": before})

    print("\n✅  All plots saved:")
    for k, p in paths.items():
        print(f"   {k:15s} → {p}")

if __name__ == "__main__":
    main()
