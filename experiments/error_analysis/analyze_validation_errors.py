
from pathlib import Path

import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

CONFUSION_OUTPUT_PATH = REPORTS_DIR / "validation_confusion_matrix.csv"
CLASSIFICATION_OUTPUT_PATH = REPORTS_DIR / "validation_classification_report.csv"


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

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print("\n--- Classification Report ---")

    report = classification_report(
        y_val,
        predictions,
        zero_division=0,
        digits=4,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())

    labels = model.classes_
    matrix = confusion_matrix(y_val, predictions, labels=labels)

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print("\n--- Confusion Matrix ---")
    print("Rows = actual categories; columns = predicted categories")
    print(confusion_df.to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    confusion_df.to_csv(CONFUSION_OUTPUT_PATH)
    report_df.to_csv(CLASSIFICATION_OUTPUT_PATH)

    print(f"\nSaved confusion matrix: {CONFUSION_OUTPUT_PATH}")
    print(f"Saved classification report: {CLASSIFICATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
