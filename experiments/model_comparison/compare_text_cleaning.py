
import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def normalize_text(text):
    """Normalize whitespace without removing ticket content."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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

    X_train_original = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_val_original = validation_df["Document"]
    y_val = validation_df["Topic_group"]

    if set(y_train.unique()) != set(y_val.unique()):
        raise ValueError(
            "Training and validation category sets do not match."
        )

    datasets = {
        "original_text": (
            X_train_original,
            X_val_original,
        ),
        "normalized_text": (
            X_train_original.map(normalize_text),
            X_val_original.map(normalize_text),
        ),
    }

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    results = []

    for name, (X_train, X_val) in datasets.items():
        print(f"\nTraining configuration: {name}")

        model = build_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0,
        )

        results.append({
            "Configuration": name,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n--- Comparison summary ---")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORTS_DIR / "text_cleaning_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved comparison report: {output_path}")

    print(
        "\nNote: whitespace normalization is intentionally "
        "conservative. A small or zero difference is a valid result."
    )

    print("\nText cleaning comparison completed successfully.")


if __name__ == "__main__":
    main()
