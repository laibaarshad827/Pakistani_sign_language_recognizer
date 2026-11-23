"""Trade-off: a stricter flag threshold gives fewer false alarms but misses more real mistakes."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from classifiers import SEED
from feedback import ANGLES, FINGERS, FeedbackEngine
from feedback_experiment import FEATURES_PATH, FIGURES_DIR, RESULTS_DIR, make_mistake

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
DELTAS = [30, 45]
MAX_FALSE_ALARM = 0.10      # rule fixed in advance: at most 1 in 10 correct attempts may be flagged


def main():
    np.random.seed(SEED)
    try:
        feats = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Missing {FEATURES_PATH}. Run src/eda.py first.")
    labels = feats["sign"].to_numpy()
    groups = feats["session"].to_numpy()
    angles = feats[ANGLES].to_numpy(dtype=float)

    correct_worst = []
    mistakes = {delta: {"blamed": [], "worst": [], "truth": []} for delta in DELTAS}
    for train, test in LeaveOneGroupOut().split(angles, labels, groups):
        engine = FeedbackEngine().fit(feats.iloc[train], labels[train])
        size, _ = engine.finger_deviation(labels[test], angles[test])
        correct_worst.append(size.max(axis=1))
        for delta in DELTAS:
            for finger in range(len(FINGERS)):
                broken = make_mistake(angles[test], labels[test], engine, finger, delta)
                size, _ = engine.finger_deviation(labels[test], broken)
                mistakes[delta]["blamed"].append(size.argmax(axis=1))
                mistakes[delta]["worst"].append(size.max(axis=1))
                mistakes[delta]["truth"].append(np.full(len(test), finger))

    correct_worst = np.concatenate(correct_worst)
    rows = []
    for threshold in THRESHOLDS:
        row = {"threshold": threshold, "false_alarm": float((correct_worst > threshold).mean())}
        for delta in DELTAS:
            blamed = np.concatenate(mistakes[delta]["blamed"])
            worst = np.concatenate(mistakes[delta]["worst"])
            truth = np.concatenate(mistakes[delta]["truth"])
            row[f"named_at_{delta}deg"] = float(((blamed == truth) & (worst > threshold)).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    allowed = table[table["false_alarm"] <= MAX_FALSE_ALARM]
    if len(allowed):
        print(f"\nSuggested threshold (lowest with at most {MAX_FALSE_ALARM:.0%} false alarms): "
              f"{allowed.iloc[0]['threshold']}")
    else:
        print(f"\nNo threshold reaches {MAX_FALSE_ALARM:.0%} false alarms; try larger values.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "feedback_tradeoff.csv"), index=False)
    plt.figure(figsize=(6, 4))
    for delta in DELTAS:
        plt.plot(table["false_alarm"], table[f"named_at_{delta}deg"], "o-", label=f"{delta}-degree mistake")
    for _, r in table.iterrows():
        plt.annotate(str(r["threshold"]), (r["false_alarm"], r[f"named_at_{DELTAS[-1]}deg"]),
                     textcoords="offset points", xytext=(4, -10), fontsize=8)
    plt.xlabel("false alarms on correct attempts")
    plt.ylabel("mistakes where the right finger was named")
    plt.title("Feedback trade-off (numbers = flag threshold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feedback_tradeoff.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()