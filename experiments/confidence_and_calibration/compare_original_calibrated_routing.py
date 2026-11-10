
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

OUTPUT_PATH = REPORTS_DIR / "original_vs_calibrated_routing.csv"


THRESHOLDS = {
    "Access": 0.80,
    "Administrative rights": 0.95,
    "HR Support": 0.90,
    "Hardware": 0.90,
    "Internal Project": 0.90,
    "Miscellaneous": 0.90,
    "Purchase": 0.80,
    "Storage": 0.90,
}


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


def evaluate_routing(name, y_true, probabilities, classes):
    predicted_indices = probabilities.argmax(axis=1)
    predictions = classes[predicted_indices]
    confidence = probabilities.max(axis=1)

    thresholds = pd.Series(predictions).map(THRESHOLDS).to_numpy()

    if pd.isna(thresholds).any():
        raise ValueError(
            "A predicted category has no routing threshold."
        )

    auto_routed = confidence >= thresholds
    correct = predictions == np.asarray(y_true)

    total = len(y_true)
    auto_count = int(auto_routed.sum())
    review_count = total - auto_count
    correct_auto = int((auto_routed & correct).sum())
    wrong_auto = auto_count - correct_auto

    result = {
        "Model": name,
        "Total_Tickets": total,
        "Auto_Routed": auto_count,
        "Auto_Coverage": auto_count / total if total else np.nan,
        "Human_Review": review_count,
        "Review_Rate": review_count / total if total else np.nan,
        "Correct_Auto": correct_auto,
        "Wrong_Auto": wrong_auto,
        "Auto_Precision": (
            correct_auto / auto_count if auto_count else np.nan
        ),
        "Overall_Accuracy": correct.mean() if total else np.nan,
    }

    return result, predictions, confidence, auto_routed, correct


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

    # Calibrate the complete pipeline to prevent feature-extraction
    # information from being shared across calibration folds.
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

    original_classes = original_model.classes_
    calibrated_classes = calibrated_model.classes_

    if not np.array_equal(original_classes, calibrated_classes):
        raise ValueError(
            "Original and calibrated class orders differ."
        )

    original_result, original_predictions, original_confidence, original_auto, original_correct = (
        evaluate_routing(
            "Original",
            y_validation,
            original_probabilities,
            original_classes,
        )
    )

    calibrated_result, calibrated_predictions, calibrated_confidence, calibrated_auto, calibrated_correct = (
        evaluate_routing(
            "Calibrated",
            y_validation,
            calibrated_probabilities,
            calibrated_classes,
        )
    )

    results_df = pd.DataFrame([
        original_result,
        calibrated_result,
    ])

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== ORIGINAL VS CALIBRATED ROUTING ===")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Auto_Coverage": "{:.2%}".format,
                "Review_Rate": "{:.2%}".format,
                "Auto_Precision": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
                "Overall_Accuracy": "{:.2%}".format,
            },
        )
    )

    print("\n=== CATEGORY-WISE ROUTING COMPARISON ===")

    for category in THRESHOLDS:
        print(f"\n{category}")

        for name, predictions, confidence, auto, correct in [
            (
                "Original",
                original_predictions,
                original_confidence,
                original_auto,
                original_correct,
            ),
            (
                "Calibrated",
                calibrated_predictions,
                calibrated_confidence,
                calibrated_auto,
                calibrated_correct,
            ),
        ]:
            category_mask = predictions == category
            accepted = category_mask & auto

            accepted_count = int(accepted.sum())
            category_count = int(category_mask.sum())
            correct_count = int((accepted & correct).sum())

            precision = (
                correct_count / accepted_count
                if accepted_count else np.nan
            )

            coverage = (
                accepted_count / category_count
                if category_count else np.nan
            )

            print(
                f"  {name}: accepted={accepted_count}, "
                f"coverage={coverage:.2%}, "
                f"precision={precision:.2%}"
                if accepted_count
                else f"  {name}: accepted=0"
            )

    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
