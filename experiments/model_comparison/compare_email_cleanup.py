
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

MAX_AUDIT_EXAMPLES = 100


def normalize_text(text):
    """Normalize whitespace without removing ticket content."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_email_phrases(text):
    """
    Experimentally remove common greetings and standalone sign-offs.

    This is deliberately conservative. It does not attempt to remove
    arbitrary sentences or entire email signatures.
    """
    text = normalize_text(text)

    # Remove common greetings only at the beginning.
    text = re.sub(
        r"^(?:dear\b[^,!.]{0,80}[,!.]?|"
        r"hello\b[^,!.]{0,50}[,!.]?|"
        r"hi\b[^,!.]{0,50}[,!.]?)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone closing phrases, optionally followed by
    # a short name. Avoid removing arbitrary text after "thanks".
    text = re.sub(
        r"\s*(?:kind regards|best regards|regards|thank you|thanks)"
        r"[,.!]*"
        r"(?:\s+[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?)?"
        r"\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip()


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

    # Apply the two preprocessing configurations.
    X_train_normalized = X_train_original.map(normalize_text)
    X_val_normalized = X_val_original.map(normalize_text)

    X_train_cleaned = X_train_original.map(clean_email_phrases)
    X_val_cleaned = X_val_original.map(clean_email_phrases)

    datasets = {
        "normalized_only": (
            X_train_normalized,
            X_val_normalized,
        ),
        "normalized_email_cleanup": (
            X_train_cleaned,
            X_val_cleaned,
        ),
    }

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    # Measure how many validation tickets were modified.
    changed_mask = (
        X_val_normalized != X_val_cleaned
    )

    changed_count = int(changed_mask.sum())

    print("\n--- Email cleanup diagnostics ---")
    print(f"Validation tickets: {len(validation_df)}")
    print(f"Tickets modified: {changed_count}")
    print(
        f"Modification rate: "
        f"{changed_count / len(validation_df):.2%}"
    )

    audit_df = pd.DataFrame({
        "Original_Text": X_val_original[changed_mask],
        "Normalized_Text": X_val_normalized[changed_mask],
        "Cleaned_Text": X_val_cleaned[changed_mask],
        "Actual_Category": y_val[changed_mask],
    }).head(MAX_AUDIT_EXAMPLES)

    results = []

    for name, (X_train, X_val) in datasets.items():
        print(f"\nTraining configuration: {name}")

        model = build_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        result = {
            "Configuration": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro_F1": f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "Weighted_F1": f1_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

        results.append(result)

        print(f"Accuracy: {result['Accuracy']:.4f}")
        print(f"Macro F1: {result['Macro_F1']:.4f}")
        print(f"Weighted F1: {result['Weighted_F1']:.4f}")

    results_df = pd.DataFrame(results)

    print("\n--- Comparison summary ---")
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        REPORTS_DIR / "email_cleanup_comparison.csv"
    )
    audit_path = (
        REPORTS_DIR / "email_cleanup_audit.csv"
    )

    results_df.to_csv(metrics_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    print(f"\nSaved comparison report: {metrics_path}")
    print(f"Saved cleanup audit: {audit_path}")

    print(
        "\nNote: email cleanup is experimental. "
        "Review the audit examples before adopting it."
    )

    print("\nEmail cleanup comparison completed successfully.")


if __name__ == "__main__":
    main()
