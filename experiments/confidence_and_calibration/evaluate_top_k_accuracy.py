
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
OUTPUT_PATH = REPORTS_DIR / "top_k_accuracy_by_category.csv"


def normalize_text(text):
    return " ".join(str(text).split())


def validate_data(df, file_path):
    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    if df["Document"].isna().any():
        raise ValueError(f"Missing Document values found in {file_path}")

    if df["Topic_group"].isna().any():
        raise ValueError(f"Missing Topic_group values found in {file_path}")


def main():
    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    validate_data(train_df, TRAIN_PATH)
    validate_data(validation_df, VALIDATION_PATH)

    X_train = train_df["Document"].map(normalize_text)
    y_train = train_df["Topic_group"]

    X_validation = validation_df["Document"].map(normalize_text)
    y_validation = validation_df["Topic_group"]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=100_000,
        sublinear_tf=True,
    )

    features = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ])

    X_train_features = features.fit_transform(X_train)
    X_validation_features = features.transform(X_validation)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(X_train_features, y_train)

    probabilities = model.predict_proba(X_validation_features)
    classes = model.classes_

    if len(classes) < 3:
        raise ValueError("Top-3 accuracy requires at least three classes.")

    ranked_indices = probabilities.argsort(axis=1)[:, ::-1]

    top1_predictions = classes[ranked_indices[:, 0]]
    top2_predictions = classes[ranked_indices[:, :2]]
    top3_predictions = classes[ranked_indices[:, :3]]

    top1_accuracy = accuracy_score(y_validation, top1_predictions)

    top2_accuracy = sum(
        actual in predictions
        for actual, predictions in zip(y_validation, top2_predictions)
    ) / len(y_validation)

    top3_accuracy = sum(
        actual in predictions
        for actual, predictions in zip(y_validation, top3_predictions)
    ) / len(y_validation)

    print("\n=== TOP-K ACCURACY EVALUATION ===")
    print(f"Validation tickets: {len(y_validation)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f} ({top1_accuracy:.2%})")
    print(f"Top-2 accuracy: {top2_accuracy:.4f} ({top2_accuracy:.2%})")
    print(f"Top-3 accuracy: {top3_accuracy:.4f} ({top3_accuracy:.2%})")

    print("\n=== TOP-K ACCURACY BY CATEGORY ===")

    results = []

    for category in classes:
        mask = y_validation.to_numpy() == category
        actual_count = int(mask.sum())

        if actual_count == 0:
            continue

        top1_correct = int(
            (top1_predictions[mask] == category).sum()
        )

        top2_correct = sum(
            category in predictions
            for predictions in top2_predictions[mask]
        )

        top3_correct = sum(
            category in predictions
            for predictions in top3_predictions[mask]
        )

        results.append({
            "Category": category,
            "Support": actual_count,
            "Top1_Accuracy": top1_correct / actual_count,
            "Top2_Accuracy": top2_correct / actual_count,
            "Top3_Accuracy": top3_correct / actual_count,
        })

    results_df = pd.DataFrame(results)

    print(results_df.to_string(
        index=False,
        formatters={
            "Top1_Accuracy": "{:.2%}".format,
            "Top2_Accuracy": "{:.2%}".format,
            "Top3_Accuracy": "{:.2%}".format,
        }
    ))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCategory results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
