"""Experiment: does the feedback engine name the finger that is wrong?

We take correct attempts from a session the engine has never seen, deliberately bend ONE
finger by a known number of degrees, and check whether the engine blames that finger.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from classifiers import SEED
from feedback import ANGLES, FINGERS, FeedbackEngine

FEATURES_PATH = "data/processed/features.csv"
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
DELTAS = [15, 30, 45, 60]        # size of the deliberate mistake, in degrees
MAIN_DELTA = 45


def make_mistake(angles, labels, engine, finger, delta):
    """Move the three joints of one finger `delta` degrees away from the sign's typical angle."""
    out = np.asarray(angles, dtype=float).reshape(-1, 5, 3).copy()
    typical = engine.mean.loc[list(labels), ANGLES].to_numpy().reshape(-1, 5, 3)[:, finger, :]
    direction = np.where(typical > 90, -1.0, 1.0)      # a straight finger gets bent, a bent one straightened
    out[:, finger, :] = np.clip(out[:, finger, :] + direction * delta, 0.0, 180.0)
    return out.reshape(-1, 15)


def main():
    np.random.seed(SEED)
    try:
        feats = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Missing {FEATURES_PATH}. Run src/eda.py first.")
    labels = feats["sign"].to_numpy()
    groups = feats["session"].to_numpy()
    angles = feats[ANGLES].to_numpy(dtype=float)

    rows, false_alarms = [], []
    for train, test in LeaveOneGroupOut().split(angles, labels, groups):
        engine = FeedbackEngine().fit(feats.iloc[train], labels[train])
        size, _ = engine.finger_deviation(labels[test], angles[test])
        false_alarms.append((size.max(axis=1) > engine.flag_threshold).mean())
        for delta in DELTAS:
            for finger, name in enumerate(FINGERS):
                broken = make_mistake(angles[test], labels[test], engine, finger, delta)
                size, _ = engine.finger_deviation(labels[test], broken)
                blamed = size.argmax(axis=1)
                flagged = size.max(axis=1) > engine.flag_threshold
                rows.append({"delta": delta, "finger": name,
                             "named_correctly": float(((blamed == finger) & flagged).mean()),
                             "mean_score": float(engine.score(size).mean())})

    results = pd.DataFrame(rows).groupby(["delta", "finger"]).mean().reset_index()
    by_delta = results.groupby("delta")[["named_correctly", "mean_score"]].mean()
    false_alarm = float(np.mean(false_alarms))
    print(f"False alarms on CORRECT attempts: {false_alarm:.1%} of attempts wrongly flagged")
    print("\nShare of deliberate mistakes where the right finger was named:")
    print(by_delta.round(3).to_string())
    print(f"\nPer finger at a {MAIN_DELTA}-degree mistake:")
    print(results[results["delta"] == MAIN_DELTA].set_index("finger")["named_correctly"].round(3).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "feedback_experiment.csv"), index=False)

    main_rows = results[results["delta"] == MAIN_DELTA].set_index("finger").loc[FINGERS]
    plt.figure(figsize=(6, 4))
    plt.bar(FINGERS, main_rows["named_correctly"])
    plt.ylim(0, 1)
    plt.ylabel("mistakes where the right finger was named")
    plt.title(f"Feedback accuracy per finger ({MAIN_DELTA}-degree mistake)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feedback_per_finger.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(by_delta.index, by_delta["named_correctly"], "o-", label="right finger named")
    plt.axhline(false_alarm, color="red", linestyle="--", label="false alarms on correct attempts")
    plt.ylim(0, 1)
    plt.xlabel("size of the deliberate mistake (degrees)")
    plt.ylabel("share of attempts")
    plt.title("Feedback accuracy vs size of the mistake")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feedback_vs_delta.png"), dpi=150)
    plt.close()

    # one example message
    engine = FeedbackEngine().fit(feats, labels)
    label = labels[0]
    broken = make_mistake(angles[:1], [label], engine, 1, MAIN_DELTA)
    print(f"\nExample, sign '{label}' with the index finger changed by {MAIN_DELTA} degrees:")
    print(engine.explain(label, broken[0]))


if __name__ == "__main__":
    main()