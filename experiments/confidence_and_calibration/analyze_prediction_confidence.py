
import re
from pathlib import Path

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline


# Resolve project paths independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
OUTPUT_PATH = REPORTS_DIR / "validation_prediction_confidence.csv"


def normalize_text(text):
    """Normalize whitespace without removing ticket content."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def main():
    # Validate input files.
    for path in (TRAIN_PATH, VALIDATION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {path}")

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

    X_train = train_df["Document"].fillna("").map(normalize_text)
    y_train = train_df["Topic_group"]

    X_val = validation_df["Document"].fillna("").map(normalize_text)
    y_val = validation_df["Topic_group"]

    model = Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        lowercase=True,
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

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    print("\nTraining model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)
    predicted_indices = probabilities.argmax(axis=1)
    predictions = model.classes_[predicted_indices]
    confidence = probabilities.max(axis=1)
    correct = predictions == y_val.to_numpy()

    results = validation_df.copy()
    results["Actual"] = y_val.to_numpy()
    results["Predicted"] = predictions
    results["Confidence"] = confidence
    results["Correct"] = correct

    print("\n--- Overall results ---")
    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Mean maximum confidence: {confidence.mean():.4f}")
    print(f"Correct predictions: {correct.sum()}")
    print(f"Incorrect predictions: {(~correct).sum()}")

    # Group predictions by maximum confidence.
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = [
        "0-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    results["Confidence_Band"] = pd.cut(
        results["Confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    summary = (
        results.groupby("Confidence_Band", observed=False)
        .agg(
            tickets=("Correct", "size"),
            accuracy=("Correct", "mean"),
            mean_confidence=("Confidence", "mean"),
        )
        .reset_index()
    )

    print("\n--- Accuracy by confidence band ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save ticket-level results for further analysis.
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    # A descriptive calibration diagnostic, not a guarantee of reliability.
    fraction_correct, mean_predicted = calibration_curve(
        correct.astype(int),
        confidence,
        n_bins=10,
        strategy="uniform",
    )

    print("\n--- Confidence calibration bins ---")
    for index, (predicted, observed) in enumerate(
        zip(mean_predicted, fraction_correct),
        start=1,
    ):
        print(
            f"Bin {index}: mean confidence={predicted:.4f}, "
            f"observed accuracy={observed:.4f}"
        )


if __name__ == "__main__":
    main()
