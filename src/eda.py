"""Look at the raw data, check its quality and save the feature table."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import LANDMARK_COLS, META_COLS, build_features, coord_columns

RAW_PATH = "data/raw/landmarks.csv"
OUT_PATH = "data/processed/features.csv"
FIG_PATH = "outputs/figures/eda_within_take_motion.png"


def main():
    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {RAW_PATH}. Record some sessions first.")

    print("== Size ==")
    print("rows:", len(df), "| signs:", df["sign"].nunique(), "| sessions:", sorted(df["session"].unique()))

    print("\n== Balance: rows per sign (min / max) ==")
    per_sign = df.groupby("sign").size()
    print(per_sign.min(), "/", per_sign.max())
    print("\nrows per session:")
    print(df.groupby("session").size())

    print("\n== Missing values and empty frames ==")
    print("missing cells:", int(df[LANDMARK_COLS].isna().sum().sum()))
    print("all-zero rows:", int((df[LANDMARK_COLS].abs().sum(axis=1) == 0).sum()))

    xy = [f"{a}{i}" for i in range(21) for a in "xy"]
    outside = ((df[xy] < 0) | (df[xy] > 1)).any(axis=1)
    print("rows where part of the hand is outside the image:", int(outside.sum()))

    feats = build_features(df)
    feature_cols = [c for c in feats.columns if c not in META_COLS]

    print("\n== Within-take motion (a high value = the hand shape changes during the take) ==")
    motion = (feats.groupby(["sign", "session", "take"])[coord_columns()].std().mean(axis=1)
              .groupby("sign").mean().sort_values(ascending=False))
    print(motion.head(8).round(4))

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.figure(figsize=(10, 4))
    motion.plot(kind="bar")
    plt.ylabel("average within-take std of normalized coordinates")
    plt.title("How much each sign changes during a take")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(feats)} rows x {len(feature_cols)} features to {OUT_PATH}")


if __name__ == "__main__":
    main()