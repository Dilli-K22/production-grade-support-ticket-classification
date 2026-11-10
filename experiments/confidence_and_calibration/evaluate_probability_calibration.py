
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

OUTPUT_PATH = REPORTS_DIR / "probability_calibration.csv"


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


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


def calculate_ece(y_true, probabilities, classes, n_bins=10):
    """
    Calculate top-label Expected Calibration Error.
    """
    predicted_indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    predictions = classes[predicted_indices]
    correct = (predictions == np.asarray(y_true)).astype(int)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence < bin_edges[i + 1])
            )

        if mask.any():
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(bin_accuracy - bin_confidence)
            )

    return ece


def evaluate_model(name, y_true, probabilities, classes):
    predictions = classes[probabilities.argmax(axis=1)]

    accuracy = np.mean(predictions == np.asarray(y_true))
    loss = log_loss(y_true, probabilities, labels=classes)
    ece = calculate_ece(y_true, probabilities, classes)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Log_Loss": loss,
        "ECE": ece,
    }


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

    # Original model.
    original_model = build_model()

    print("\nTraining original model...")
    original_model.fit(X_train, y_train)

    original_probabilities = original_model.predict_proba(
        X_validation
    )
    original_classes = original_model.classes_

    # Calibrate the entire pipeline so TF-IDF is fitted separately
    # within each cross-validation fold.
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
    calibrated_classes = calibrated_model.classes_

    if not np.array_equal(original_classes, calibrated_classes):
        raise ValueError("Original and calibrated class orders differ.")

    results = [
        evaluate_model(
            "Original",
            y_validation,
            original_probabilities,
            original_classes,
        ),
        evaluate_model(
            "Calibrated",
            y_validation,
            calibrated_probabilities,
            calibrated_classes,
        ),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== PROBABILITY CALIBRATION COMPARISON ===")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Log_Loss": "{:.4f}".format,
                "ECE": "{:.4f}".format,
            },
        )
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")

    print("\n=== CALIBRATION CURVE DATA ===")

    for name, probabilities, classes in [
        ("Original", original_probabilities, original_classes),
        ("Calibrated", calibrated_probabilities, calibrated_classes),
    ]:
        confidence = probabilities.max(axis=1)
        predictions = classes[probabilities.argmax(axis=1)]
        correct = (
            predictions == y_validation
        ).astype(int)

        observed_accuracy, mean_confidence = calibration_curve(
            correct,
            confidence,
            n_bins=10,
            strategy="uniform",
        )

        print(f"\n{name}")

        for conf, acc in zip(mean_confidence, observed_accuracy):
            print(
                f"Mean confidence: {conf:.4f} | "
                f"Observed accuracy: {acc:.4f}"
            )


if __name__ == "__main__":
    main()
