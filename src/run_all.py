"""Reproduce every result of the project with ONE command (all random seeds are fixed to 42).

Usage (from the project folder):
    python src/run_all.py
    python src/run_all.py --merge haa,hamza --merge zuad,zuey      (look-alike signs to merge)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
LOG_PATH = ROOT / "outputs" / "results" / "run_log.txt"
DEFAULT_MERGE = ["haa,hamza"]


def build_steps(merge):
    """The scripts to run, in order: (title, script name and its arguments)."""
    merge_args = []
    for group in merge:
        merge_args += ["--merge", group]
    return [
        ("Data checks and features (EDA)", ["eda.py"]),
        ("Model comparison, all signs", ["train_eval.py", "--tag", "_34signs_v2"]),
        ("Model comparison, look-alike signs merged", ["train_eval.py", *merge_args, "--tag", "_merged_v2"]),
        ("Kalman filter experiment", ["kalman_experiment.py"]),
        ("Feedback threshold trade-off", ["feedback_tradeoff.py"]),
        ("Feedback engine validation", ["feedback_experiment.py"]),
        ("Final model for the live trainer", ["train_final.py", *merge_args]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", action="append", default=None,
                        help="look-alike signs to treat as one group, e.g. haa,hamza (default: haa,hamza)")
    args = parser.parse_args()
    steps = build_steps(args.merge if args.merge is not None else DEFAULT_MERGE)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        for number, (title, command) in enumerate(steps, 1):
            print(f"\n[{number}/{len(steps)}] {title}")
            started = time.time()
            try:
                result = subprocess.run([sys.executable, str(SRC / command[0]), *command[1:]], cwd=ROOT,
                                        capture_output=True, text=True, encoding="utf-8", errors="replace")
            except OSError as error:
                raise SystemExit(f"Could not start step {number} ({command[0]}): {error}")
            log.write(f"=== {title} ({' '.join(command)}) ===\n{result.stdout}\n")
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                raise SystemExit(f"Step {number} ({command[0]}) failed. Read the message above.")
            print(f"    finished in {time.time() - started:.0f} s")

    print(f"\nAll {len(steps)} steps finished in {time.time() - began:.0f} s.")
    print("Tables: outputs/results/   Charts: outputs/figures/   Printed numbers: outputs/results/run_log.txt")


if __name__ == "__main__":
    main()