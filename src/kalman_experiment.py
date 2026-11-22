"""Experiment: does Kalman smoothing make the predicted sign steadier (less flickering)?"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from classifiers import SVMModel, SEED
from features import META_COLS, angle_columns, build_features, coord_columns
from smoothing import estimate_noise, smooth_dataframe
from train_eval import merge_labels

RAW_PATH = "data/raw/landmarks.csv"
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
FPS = 30                          # webcam frames per second
KEYS = ["sign", "session", "take"]
DEFAULT_RATIO = 0.01              # chosen from the noise estimate, not from test scores
RATIOS = [0.001, 0.01, 0.1, 1.0]


def feature_matrix(feats):
    """The numbers the classifier sees (same columns as in train_eval.py)."""
    return feats[coord_columns() + angle_columns()].to_numpy(dtype=float)


def take_metrics(feats, pred, y):
    """Average jitter, flicker (label changes per second) and accuracy over the takes."""
    frame = feats[META_COLS].copy()
    frame["pred"] = pred
    changes = frame.groupby(KEYS)["pred"].apply(lambda s: (s.to_numpy()[1:] != s.to_numpy()[:-1]).mean())
    jitter = feats.groupby(KEYS)[coord_columns()].diff().abs().mean().mean()
    return {"jitter": jitter, "flicker_per_second": changes.mean() * FPS,
            "accuracy": accuracy_score(y, pred)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="append", default=[], help="e.g. --merge haa,hamza")
    parser.add_argument("--trace-sign", default="alif", help="sign used for the trace picture")
    args = parser.parse_args()
    np.random.seed(SEED)

    try:
        raw = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {RAW_PATH}")
    y = merge_labels(raw["sign"].to_numpy(), args.merge)
    groups = raw["session"].to_numpy()

    noise = estimate_noise(raw)
    print("Estimated measurement noise (std) per axis:",
          {axis: round(float(np.sqrt(v)), 5) for axis, v in noise.items()})

    settings = [("Raw (no filter)", None)] + [(f"Kalman, Q/R = {r}", r) for r in RATIOS]
    raw_feats = build_features(raw)
    smoothed_feats = {r: build_features(smooth_dataframe(raw, noise, r)) for r in RATIOS}

    totals = {name: [] for name, _ in settings}
    for train, test in LeaveOneGroupOut().split(raw, y, groups):
        model = SVMModel().fit(feature_matrix(raw_feats.iloc[train]), y[train])
        for name, ratio in settings:
            feats = (raw_feats if ratio is None else smoothed_feats[ratio]).iloc[test]
            totals[name].append(take_metrics(feats, model.predict(feature_matrix(feats)), y[test]))

    rows = []
    for name, _ in settings:
        stats = pd.DataFrame(totals[name]).mean()
        rows.append({"setting": name, **stats.round(4).to_dict()})
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "kalman_experiment.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(table["setting"], table["flicker_per_second"])
    axes[0].set_ylabel("predicted-label changes per second")
    axes[0].set_title("Flicker (lower is steadier)")
    axes[1].bar(table["setting"], table["accuracy"])
    axes[1].set_ylabel("accuracy on unseen sessions")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kalman_flicker_accuracy.png"), dpi=150)
    plt.close()

    # picture: one coordinate over one take, raw vs smoothed
    sign = args.trace_sign if args.trace_sign in set(raw["sign"]) else raw["sign"].iloc[0]
    take = raw[(raw["sign"] == sign)].groupby(["session", "take"]).ngroup()
    first = raw[(raw["sign"] == sign) & (take == 0)]
    smoothed = smooth_dataframe(first, noise, DEFAULT_RATIO)
    plt.figure(figsize=(8, 4))
    plt.plot(first["frame"], first["x8"], "o-", alpha=0.5, label="raw (noisy)")
    plt.plot(smoothed["frame"], smoothed["x8"], "-", linewidth=2.5, label=f"Kalman (Q/R={DEFAULT_RATIO})")
    plt.xlabel("frame")
    plt.ylabel("x position of the index fingertip (image fraction)")
    plt.title(f"Raw vs Kalman-smoothed landmark, sign '{sign}'")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kalman_trace.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()