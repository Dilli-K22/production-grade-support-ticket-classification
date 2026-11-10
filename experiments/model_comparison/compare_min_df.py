
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

MIN_DF_VALUES = [2, 3]

TARGET_CLASSES = [
    "Administrative rights",
    "Hardware",
    "Access",
]


def load_dataset(path, dataset_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")

    if df["Topic_group"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing target labels."
        )

    df["Document"] = df["Document"].fillna("").astype(str)

    return df


def build_model(min_df):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=min_df,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def main():
    print("Loading datasets...")

    train_df = load_dataset(TRAIN_PATH, "Training")
    validation_df = load_dataset(
        VALIDATION_PATH,
        "Validation",
    )

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_val = validation_df["Document"]
    y_val = validation_df["Topic_group"]

    if set(y_train.unique()) != set(y_val.unique()):
        raise ValueError(
            "Training and validation category sets do not match."
        )

    missing_targets = set(TARGET_CLASSES) - set(y_val.unique())

    if missing_targets:
        raise ValueError(
            f"Target categories missing from validation data: "
            f"{sorted(missing_targets)}"
        )

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    results = []

    for min_df in MIN_DF_VALUES:
        print(f"\nTraining configuration: min_df={min_df}")

        model = build_model(min_df)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        report = classification_report(
            y_val,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        result = {
            "min_df": min_df,
            "accuracy": accuracy_score(y_val, predictions),
            "macro_f1": f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

        for category in TARGET_CLASSES:
            result[f"{category}_f1"] = (
                report[category]["f1-score"]
            )

        results.append(result)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Weighted F1: {result['weighted_f1']:.4f}")

        for category in TARGET_CLASSES:
            print(
                f"{category} F1: "
                f"{result[f'{category}_f1']:.4f}"
            )

    results_df = pd.DataFrame(results)

    print("\n--- min_df comparison summary ---")
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "min_df_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved comparison report: {output_path}")
    print("\nMinimum document frequency comparison completed.")


if __name__ == "__main__":
    main()
