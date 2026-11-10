
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

OUTPUT_PATH = REPORTS_DIR / "misclassified_validation_tickets.csv"


def validate_data(df, file_path):
    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    if df["Topic_group"].isna().any():
        raise ValueError(f"Missing category labels found in {file_path}")


def build_model():
    return Pipeline([
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


def main():
    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    validate_data(train_df, TRAIN_PATH)
    validate_data(validation_df, VALIDATION_PATH)

    X_train = train_df["Document"].fillna("").astype(str)
    y_train = train_df["Topic_group"]

    X_val = validation_df["Document"].fillna("").astype(str)
    y_val = validation_df["Topic_group"]

    model = build_model()

    print("Training candidate model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    classes = model.classes_

    print("\n--- Validation classification report ---")
    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0,
            digits=4,
        )
    )

    # Export incorrect predictions for manual review.
    errors = validation_df.copy()

    errors["Actual"] = y_val.to_numpy()
    errors["Predicted"] = predictions
    errors["Prediction_Confidence"] = probabilities.max(axis=1)
    errors["Correct"] = errors["Actual"] == errors["Predicted"]

    # Find the top two predictions.
    ranked_indices = probabilities.argsort(axis=1)[:, ::-1]

    errors["Top_1_Category"] = classes[ranked_indices[:, 0]]
    errors["Top_1_Confidence"] = probabilities[
        range(len(probabilities)), ranked_indices[:, 0]
    ]

    errors["Top_2_Category"] = classes[ranked_indices[:, 1]]
    errors["Top_2_Confidence"] = probabilities[
        range(len(probabilities)), ranked_indices[:, 1]
    ]

    errors["Top_2_Confidence_Gap"] = (
        errors["Top_1_Confidence"] - errors["Top_2_Confidence"]
    )

    misclassified = errors.loc[~errors["Correct"]].copy()

    # Put the most confident incorrect predictions first.
    misclassified = misclassified.sort_values(
        "Prediction_Confidence",
        ascending=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    misclassified.to_csv(OUTPUT_PATH, index=False)

    print(f"\nValidation tickets: {len(errors)}")
    print(f"Misclassified tickets: {len(misclassified)}")
    print(f"Validation errors: {len(misclassified) / len(errors):.2%}")
    print(f"Saved review file: {OUTPUT_PATH}")

    print("\n--- Most frequent confusion pairs ---")

    confusion_pairs = (
        misclassified.groupby(["Actual", "Predicted"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    if confusion_pairs.empty:
        print("No misclassified tickets found.")
    else:
        for (actual, predicted), count in confusion_pairs.items():
            print(f"{actual} -> {predicted}: {count}")

    print("\n--- Hardware predicted as Miscellaneous ---")

    hardware_to_misc = misclassified[
        (misclassified["Actual"] == "Hardware")
        & (misclassified["Predicted"] == "Miscellaneous")
    ]

    if hardware_to_misc.empty:
        print("No matching errors found.")
    else:
        for _, row in hardware_to_misc.head(10).iterrows():
            print(f"\nConfidence: {row['Prediction_Confidence']:.2%}")
            print(f"Top 2: {row['Top_2_Category']}")
            print(f"Top 2 confidence: {row['Top_2_Confidence']:.2%}")
            print(f"Ticket: {row['Document']}")


if __name__ == "__main__":
    main()
