
"""Run Phase 4 algorithms on the real PSL dataset."""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold

from clustering import SignClusterAnalyzer
from genetic_algorithm import GeneticSVMOptimizer


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
OUTPUT_DIR = ROOT / "reports" / "phase4"
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42


def prepare_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_columns = {"sign", "session", "take", "frame"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset is missing required columns.")

    metadata_columns = ["sign", "session", "take", "frame"]
    feature_columns = [
        column for column in df.columns
        if column not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    if df[feature_columns].isnull().any().any():
        raise ValueError("Dataset contains missing feature values.")

    if not np.isfinite(df[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite feature values.")

    train_df = df[df["session"].isin(["s1", "s2"])].copy()
    test_df = df[df["session"] == "s3"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test session is empty.")

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["sign"].to_numpy()

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df["sign"].to_numpy()

    groups = train_df["session"].to_numpy()

    if len(np.unique(y_train)) != len(np.unique(y_test)):
        raise ValueError("Training and test classes do not match.")

    if set(y_train) != set(y_test):
        raise ValueError("Some signs are missing from a dataset split.")

    return (
        df,
        train_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
        groups,
        feature_columns
    )


def evaluate_model(name, model, X_test, y_test, classes):
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted")
        )
    }

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
    report_rows = {
    key: value
    for key, value in report.items()
    if isinstance(value, dict)
    }

    report_df = pd.DataFrame.from_dict(
        report_rows,
        orient="index"
    )

    report_df.loc["accuracy"] = {
        "precision": np.nan,
        "recall": np.nan,
        "f1-score": report["accuracy"],
        "support": len(y_test)
    }

    report_df.to_csv(
        OUTPUT_DIR / f"{name}_classification_report.csv"
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    )

    matrix_df.to_csv(OUTPUT_DIR / f"{name}_confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes
    )

    fig, ax = plt.subplots(figsize=(18, 18))
    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False
    )
    plt.title(f"{name} - Test Session s3")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_confusion_matrix.png",
        dpi=200
    )
    plt.close(fig)

    return metrics, predictions


def run_clustering(X_train, y_train):
    print("\n" + "=" * 60)
    print("K-MEANS SIGN CLUSTERING")
    print("=" * 60)

    analyzer = SignClusterAnalyzer(
        n_clusters=5,
        random_state=RANDOM_STATE
    )

    results = analyzer.analyze(
        X_train,
        y_train,
        top_n=30
    )

    clusters = results["clusters"]
    similar_signs = results["similar_signs"]

    with open(OUTPUT_DIR / "sign_clusters.json", "w") as file:
        json.dump(clusters, file, indent=4)

    pd.DataFrame(similar_signs).to_csv(
        OUTPUT_DIR / "similar_signs.csv",
        index=False
    )

    for cluster_number, signs in clusters.items():
        print(f"\nCluster {cluster_number}: {', '.join(signs)}")

    print("\nMost similar sign pairs:")

    for pair in similar_signs[:10]:
        print(
            f"{pair['sign_1']} - {pair['sign_2']}: "
            f"distance={pair['distance']:.4f}, "
            f"same cluster={pair['same_cluster']}"
        )

    return results


def run_baseline(X_train, y_train, X_test, y_test, classes):
    print("\n" + "=" * 60)
    print("BASELINE SVM")
    print("=" * 60)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    metrics, predictions = evaluate_model(
        "baseline_svm",
        model,
        X_test,
        y_test,
        classes
    )

    metrics["training_time_seconds"] = round(training_time, 2)

    joblib.dump(
        {
            "model": model,
            "feature_names": None
        },
        MODEL_DIR / "baseline_svm.joblib"
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return metrics, predictions


def run_genetic_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    groups,
    feature_columns,
    classes
):
    print("\n" + "=" * 60)
    print("GENETIC ALGORITHM OPTIMIZED SVM")
    print("=" * 60)

    optimizer = GeneticSVMOptimizer(
        population_size=8,
        generations=4,
        mutation_rate=0.08,
        cv_folds=2,
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()

    optimizer.fit(
        X_train,
        y_train,
        groups=groups
    )

    optimization_time = time.perf_counter() - start

    parameters = optimizer.get_best_parameters()

    selected_indices = parameters["selected_features"]
    selected_names = [
        feature_columns[index]
        for index in selected_indices
    ]

    print("\nBest Genetic Algorithm parameters:")
    print(f"Selected features: {parameters['number_of_features']}")
    print(f"Total features: {parameters['total_features']}")
    print(f"C: {parameters['C']}")
    print(f"Gamma: {parameters['gamma']}")
    print(f"Cross-validation fitness: {parameters['fitness']:.4f}")

    print("\nSelected feature names:")
    print(", ".join(selected_names))

    pd.DataFrame({
        "feature": selected_names
    }).to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False
    )

    pd.DataFrame(parameters["history"]).to_csv(
        OUTPUT_DIR / "genetic_algorithm_history.csv",
        index=False
    )

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model = optimizer.build_model()

    model.fit(X_train_selected, y_train)

    metrics, predictions = evaluate_model(
        "genetic_svm",
        model,
        X_test_selected,
        y_test,
        classes
    )

    metrics["optimization_time_seconds"] = round(
        optimization_time, 2
    )

    metrics["selected_features"] = len(selected_indices)
    metrics["total_features"] = len(feature_columns)
    metrics["best_C"] = parameters["C"]
    metrics["best_gamma"] = parameters["gamma"]
    metrics["cross_validation_fitness"] = parameters["fitness"]

    joblib.dump(
        {
            "model": model,
            "feature_names": selected_names,
            "selected_indices": selected_indices
        },
        MODEL_DIR / "genetic_svm.joblib"
    )

    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Optimization time: {optimization_time:.2f} seconds")

    return metrics, predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PAKISTANI SIGN LANGUAGE - PHASE 4 EXPERIMENT")
    print("=" * 60)

    (
        df,
        train_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
        groups,
        feature_columns
    ) = prepare_data()

    classes = sorted(np.unique(y_train))

    print(f"\nTotal samples: {len(df)}")
    print(f"Training samples (s1 + s2): {len(X_train)}")
    print(f"Final test samples (s3): {len(X_test)}")
    print(f"Number of signs: {len(classes)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Recording groups for GA validation: {len(np.unique(groups))}")

    print("\nTraining session distribution:")
    print(train_df.groupby("session").size())

    print("\nTest session distribution:")
    print(test_df.groupby("session").size())

    print("\nChecking grouped validation splits:")

    group_cv = GroupKFold(n_splits=2)

    for fold, (train_idx, validation_idx) in enumerate(
        group_cv.split(X_train, y_train, groups),
        start=1
    ):
        train_groups = set(groups[train_idx])
        validation_groups = set(groups[validation_idx])

        print(
            f"Fold {fold}: "
            f"training groups={sorted(train_groups)}, "
            f"validation groups={sorted(validation_groups)}"
        )

        if train_groups.intersection(validation_groups):
            raise RuntimeError("Recording leakage detected!")

    clustering_results = run_clustering(X_train, y_train)

    baseline_metrics, baseline_predictions = run_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        classes
    )

    genetic_metrics, genetic_predictions = run_genetic_svm(
        X_train,
        y_train,
        X_test,
        y_test,
        groups,
        feature_columns,
        classes
    )

    comparison = pd.DataFrame([
        baseline_metrics,
        genetic_metrics
    ])

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    comparison_records = comparison.to_dict(orient="records")

    with open(OUTPUT_DIR / "experiment_summary.json", "w") as file:
        json.dump(
            {
                "dataset": {
                    "total_samples": len(df),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "number_of_signs": len(classes),
                    "number_of_features": len(feature_columns),
                    "training_sessions": ["s1", "s2"],
                    "test_session": "s3"
                },
                "models": comparison_records,
                "clustering": clustering_results["clusters"]
            },
            file,
            indent=4
        )

    predictions_df = test_df[
        ["sign", "session", "take", "frame"]
    ].copy()

    predictions_df["baseline_prediction"] = baseline_predictions
    predictions_df["genetic_svm_prediction"] = genetic_predictions

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(comparison.to_string(index=False))

    print("\nAll experiment outputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
