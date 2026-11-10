
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

OUTPUT_PATH = REPORTS_DIR / "calibrated_confidence_analysis.csv"


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def calculate_confidence_stats(name, y_true, predictions, probabilities):
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(y_true)

    print(f"\n=== {name} ===")
    print(f"Mean confidence: {confidence.mean():.4f}")

    print(
        f"Mean confidence (correct): "
        f"{confidence[correct].mean():.4f}"
        if correct.any()
        else "Mean confidence (correct): N/A"
    )

    print(
        f"Mean confidence (incorrect): "
        f"{confidence[~correct].mean():.4f}"
        if (~correct).any()
        else "Mean confidence (incorrect): N/A"
    )

    print(f"Correct predictions: {correct.sum()}")
    print(f"Incorrect predictions: {(~correct).sum()}")

    return confidence, correct


def print_confidence_bands(confidence, correct):
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]

    df = pd.DataFrame({
        "Confidence": confidence,
        "Correct": correct,
    })

    df["Confidence_Band"] = pd.cut(
        df["Confidence"],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    summary = df.groupby(
        "Confidence_Band",
        observed=False,
    ).agg(
        Tickets=("Correct", "size"),
        Correct=("Correct", "sum"),
        Accuracy=("Correct", "mean"),
        Mean_Confidence=("Confidence", "mean"),
    )

    print(
        summary.to_string(
            formatters={
                "Accuracy": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
                "Mean_Confidence": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
            }
        )
    )


def build_model():
    return Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])


def main():
    for path in (TRAIN_PATH, VALIDATION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    required_columns = {"Document", "Topic_group"}

    for name, df in (
        ("training", train_df),
        ("validation", validation_df),
    ):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing)}"
            )

    X_train = train_df["Document"].map(normalize_text)
    y_train = train_df["Topic_group"]

    X_validation = validation_df["Document"].map(normalize_text)
    y_validation = validation_df["Topic_group"].to_numpy()

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    # Original, uncalibrated model.
    original_model = build_model()
    print("\nTraining original model...")
    original_model.fit(X_train, y_train)

    original_probabilities = original_model.predict_proba(
        X_validation
    )

    # Calibrate the complete pipeline so each CV fold fits its own
    # text preprocessing and classifier.
    calibrated_model = CalibratedClassifierCV(
        estimator=build_model(),
        method="sigmoid",
        cv=3,
    )

    print("\nTraining calibrated model...")
    calibrated_model.fit(X_train, y_train)

    calibrated_probabilities = calibrated_model.predict_proba(
        X_validation
    )

    original_predictions = original_model.predict(X_validation)
    calibrated_predictions = calibrated_model.predict(X_validation)

    original_confidence, original_correct = calculate_confidence_stats(
        "ORIGINAL MODEL",
        y_validation,
        original_predictions,
        original_probabilities,
    )

    calibrated_confidence, calibrated_correct = calculate_confidence_stats(
        "CALIBRATED MODEL",
        y_validation,
        calibrated_predictions,
        calibrated_probabilities,
    )

    print("\n=== ORIGINAL MODEL CONFIDENCE BANDS ===")
    print_confidence_bands(original_confidence, original_correct)

    print("\n=== CALIBRATED MODEL CONFIDENCE BANDS ===")
    print_confidence_bands(calibrated_confidence, calibrated_correct)

    output = pd.DataFrame({
        "Actual": y_validation,
        "Original_Predicted": original_predictions,
        "Original_Confidence": original_confidence,
        "Original_Correct": original_correct,
        "Calibrated_Predicted": calibrated_predictions,
        "Calibrated_Confidence": calibrated_confidence,
        "Calibrated_Correct": calibrated_correct,
    })

    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
