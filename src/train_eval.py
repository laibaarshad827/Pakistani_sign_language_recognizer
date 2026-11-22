"""Compare the baseline and the classifiers, testing each one on a recording session it never saw."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from classifiers import KNNModel, RandomForestModel, SVMModel, TemplateBaseline, SEED
from features import LANDMARK_COLS, META_COLS, angle_columns, coord_columns

RAW_PATH = "data/raw/landmarks.csv"
FEATURES_PATH = "data/processed/features.csv"
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"


def load_data(exclude):
    """Load raw landmarks and engineered features (same rows), dropping excluded signs."""
    try:
        raw = pd.read_csv(RAW_PATH)
        feats = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing data file: {error.filename}. Run src/eda.py first.")
    if not raw[META_COLS].equals(feats[META_COLS]):
        raise SystemExit("landmarks.csv and features.csv are out of sync. Re-run src/eda.py.")
    keep = ~feats["sign"].isin(exclude)
    return raw[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)


def experiments():
    """The models to compare: (name, which features it uses, a function that builds a fresh model)."""
    return [
        ("Baseline (raw template)", "raw", TemplateBaseline),
        ("SVM on raw coordinates", "raw", SVMModel),
        ("KNN", "feat", KNNModel),
        ("SVM", "feat", SVMModel),
        ("Random Forest", "feat", RandomForestModel),
    ]


def evaluate(build_model, X, y, groups):
    """Leave-one-session-out: train on the other sessions, test on the held-out one."""
    predictions = np.empty(len(y), dtype=object)
    folds = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        model = build_model().fit(X[train], y[train])
        pred = model.predict(X[test])
        predictions[test] = pred
        folds.append((groups[test][0], accuracy_score(y[test], pred), f1_score(y[test], pred, average="macro")))
    return predictions, folds


def save_confusion_matrix(y, pred, labels, title, path):
    """Plot a row-normalised confusion matrix and return the most confused sign pairs."""
    matrix = confusion_matrix(y, pred, labels=labels)
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 9))
    plt.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="share of that sign's samples")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("predicted sign")
    plt.ylabel("true sign")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    off = shares.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:8]
    return [(labels[i], labels[j], off[i, j]) for i, j in order if off[i, j] > 0]

def merge_labels(y, merge):
    """Rename look-alike signs to one shared group label, e.g. 'haa,hamza' -> 'haa+hamza'."""
    mapping = {}
    for group in merge:
        names = group.split(",")
        for name in names:
            mapping[name] = "+".join(names)
    return np.array([mapping.get(label, label) for label in y])
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="*", default=[], help="signs to leave out, e.g. pay tay")
    parser.add_argument("--merge", action="append", default=[],
                        help="treat look-alike signs as one group, e.g. --merge haa,hamza")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _34signs")
    args = parser.parse_args()
    np.random.seed(SEED)

    raw, feats = load_data(args.exclude)
    y = merge_labels(feats["sign"].to_numpy(), args.merge)
    groups = feats["session"].to_numpy()
    spaces = {"raw": raw[LANDMARK_COLS].to_numpy(dtype=float),
              "feat": feats[coord_columns() + angle_columns()].to_numpy(dtype=float)}
    labels = sorted(set(y))
    print(f"{len(labels)} classes, sessions: {sorted(set(groups))}, "
          f"excluded: {args.exclude or 'none'}, merged: {args.merge or 'none'}")

    rows, all_pred, per_sign = [], {}, {}
    for name, space, build_model in experiments():
        pred, folds = evaluate(build_model, spaces[space], y, groups)
        all_pred[name] = pred
        per_sign[name] = f1_score(y, pred, labels=labels, average=None)
        accs, f1s = [f[1] for f in folds], [f[2] for f in folds]
        rows.append({"model": name, "accuracy": np.mean(accs), "macro_f1": np.mean(f1s),
                     "macro_f1_std": np.std(f1s),
                     "per_session_f1": "; ".join(f"{f[0]}={f[2]:.3f}" for f in folds)})
        print(f"{name:26s} accuracy={np.mean(accs):.3f}  macro-F1={np.mean(f1s):.3f}")

    table = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, f"model_comparison{args.tag}.csv"), index=False)
    pd.DataFrame(per_sign, index=labels).to_csv(os.path.join(RESULTS_DIR, f"per_sign_f1{args.tag}.csv"))

    plt.figure(figsize=(8, 4))
    plt.bar(table["model"], table["macro_f1"], yerr=table["macro_f1_std"], capsize=4)
    plt.ylabel("macro-F1 on unseen sessions")
    plt.title("Model comparison (mean and std over held-out sessions)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"model_comparison{args.tag}.png"), dpi=150)
    plt.close()

    best = table.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    pairs = save_confusion_matrix(y, all_pred[best], labels, f"Confusion matrix: {best}",
                                  os.path.join(FIGURES_DIR, f"confusion_matrix{args.tag}.png"))
    print(f"\nBest model: {best}. Most confused pairs (true -> predicted, share):")
    for true, predicted, share in pairs:
        print(f"  {true} -> {predicted}: {share:.2f}")
    worst = pd.Series(per_sign[best], index=labels).sort_values().head(5)
    print("\nLowest per-sign F1 for the best model:")
    print(worst.round(3))

if __name__ == "__main__":
    main()