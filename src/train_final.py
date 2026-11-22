"""Train the final SVM on ALL sessions and save it for the live trainer."""
import argparse
import os

import joblib
import numpy as np

from classifiers import SVMModel, SEED
from features import angle_columns, coord_columns
from train_eval import load_data, merge_labels

MODEL_PATH = "models/psl_svm.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="*", default=[], help="signs to leave out")
    parser.add_argument("--merge", action="append", default=[], help="e.g. --merge haa,hamza")
    args = parser.parse_args()
    np.random.seed(SEED)

    _, feats = load_data(args.exclude)
    X = feats[coord_columns() + angle_columns()].to_numpy(dtype=float)
    y = merge_labels(feats["sign"].to_numpy(), args.merge)

    model = SVMModel().fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "classes": sorted(set(map(str, y)))}, MODEL_PATH)
    print(f"Trained on {len(y)} rows, {len(set(y))} classes. Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()