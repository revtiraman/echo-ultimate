"""Download all 7 ECHO task datasets."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.quiet:
        print("📥  Downloading ECHO ULTIMATE task datasets (7 domains)…")
    from env.task_bank import TaskBank
    bank = TaskBank()
    bank.download_all()
    bank.stats()
    print("✅  All datasets downloaded → data/tasks_cache.json")

if __name__ == "__main__":
    main()
