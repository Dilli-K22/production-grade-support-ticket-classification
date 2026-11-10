
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

# Frozen thresholds selected before final test evaluation.
# Do not optimize or modify these using test-set results.
THRESHOLDS = {
    "Access": 0.9343,
    "Administrative rights": 0.9913,
    "HR Support": 0.8168,
    "Hardware": 0.8625,
    "Internal Project": 0.9568,
    "Miscellaneous": 0.9084,
    "Purchase": 0.8097,
    "Storage": 0.8139,
}


def normalize_text(text):
    """Normalize whitespace without removing ticket content."""
    return " ".join(str(text).split())


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

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing required values."
        )

    df["Document"] = df["Document"].map(normalize_text)
    df["Topic_group"] = (
        df["Topic_group"].astype(str).str.strip()
    )

    if (df["Topic_group"] == "").any():
        raise ValueError(
            f"{dataset_name} contains empty category labels."
        )

    return df


def build_model():
    """Build the frozen word + character TF-IDF classifier."""

    features = FeatureUnion([
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
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def evaluate_routing(test_results):
    total = len(test_results)

    auto_mask = test_results["Auto_Routed"]
    correct_mask = test_results["Correct"]

    auto_count = int(auto_mask.sum())
    review_count = total - auto_count

    correct_auto = int(
        (auto_mask & correct_mask).sum()
    )

    wrong_auto = auto_count - correct_auto

    overall_accuracy = float(correct_mask.mean())

    auto_precision = (
        correct_auto / auto_count
        if auto_count
        else np.nan
    )

    overall_summary = pd.DataFrame([{
        "Total_Test_Tickets": total,
        "Overall_Accuracy": overall_accuracy,
        "Auto_Routed": auto_count,
        "Human_Review": review_count,
        "Automation_Coverage": auto_count / total,
        "Correct_Auto": correct_auto,
        "Wrong_Auto": wrong_auto,
        "Auto_Precision": auto_precision,
    }])

    category_results = []

    for category, threshold in THRESHOLDS.items():
        category_mask = (
            test_results["Predicted"] == category
        )

        accepted_mask = category_mask & auto_mask

        predicted_count = int(category_mask.sum())
        accepted_count = int(accepted_mask.sum())

        correct_count = int(
            (accepted_mask & correct_mask).sum()
        )

        wrong_count = accepted_count - correct_count

        precision = (
            correct_count / accepted_count
            if accepted_count
            else np.nan
        )

        coverage = (
            accepted_count / predicted_count
            if predicted_count
            else np.nan
        )

        category_results.append({
            "Category": category,
            "Threshold": threshold,
            "Predicted_Tickets": predicted_count,
            "Auto_Routed": accepted_count,
            "Human_Review": predicted_count - accepted_count,
            "Correct_Auto": correct_count,
            "Wrong_Auto": wrong_count,
            "Precision": precision,
            "Coverage": coverage,
        })

    return overall_summary, pd.DataFrame(category_results)


def main():
    print("Loading training and untouched test datasets...")

    train_df = load_dataset(TRAIN_PATH, "Training")
    test_df = load_dataset(TEST_PATH, "Test")

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_test = test_df["Document"]
    y_test = test_df["Topic_group"].to_numpy()

    training_categories = set(y_train.unique())
    test_categories = set(y_test)

    if training_categories != test_categories:
        raise ValueError(
            "Training and test category sets do not match."
        )

    if training_categories != set(THRESHOLDS):
        raise ValueError(
            "Frozen threshold categories do not match "
            "the dataset categories."
        )

    print(f"Training tickets: {len(train_df)}")
    print(f"Test tickets: {len(test_df)}")
    print(f"Categories: {len(training_categories)}")

    print("\nFrozen routing thresholds:")

    for category, threshold in THRESHOLDS.items():
        print(f"  {category}: {threshold:.2%}")

    print("\nBuilding and training final model...")

    model = build_model()
    model.fit(X_train, y_train)

    print("Generating test predictions...")

    probabilities = model.predict_proba(X_test)

    classes = model.classes_
    predicted_indices = probabilities.argmax(axis=1)

    predictions = classes[predicted_indices]
    confidence = probabilities.max(axis=1)

    thresholds = pd.Series(predictions).map(
        THRESHOLDS
    ).to_numpy(dtype=float)

    if not np.isfinite(thresholds).all():
        raise ValueError(
            "A predicted category has no valid frozen threshold."
        )

    auto_routed = confidence >= thresholds
    correct = predictions == y_test

    # Preserve ticket-level results for downstream statistical analysis.
    test_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions,
        "Confidence": confidence,
        "Threshold": thresholds,
        "Auto_Routed": auto_routed,
        "Correct": correct,
    })

    overall_summary, category_results = evaluate_routing(
        test_results
    )

    total = len(test_results)
    auto_count = int(auto_routed.sum())
    review_count = total - auto_count

    correct_auto = int(
        (auto_routed & correct).sum()
    )

    wrong_auto = auto_count - correct_auto

    print("\n=== FINAL UNTOUCHED TEST SET EVALUATION ===")
    print(f"Total test tickets: {total}")
    print(
        f"Overall classification accuracy: "
        f"{correct.mean():.2%}"
    )
    print(f"Automatically routed: {auto_count}")
    print(f"Human review: {review_count}")
    print(f"Automation coverage: {auto_count / total:.2%}")
    print(f"Correct automatic predictions: {correct_auto}")
    print(f"Incorrect automatic predictions: {wrong_auto}")

    if auto_count:
        print(
            f"Automatic routing precision: "
            f"{correct_auto / auto_count:.2%}"
        )
    else:
        print("Automatic routing precision: N/A")

    print("\n=== CATEGORY-WISE TEST ROUTING ===")

    print(
        category_results.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2%}".format,
                "Precision": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
                "Coverage": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ticket_path = (
        REPORTS_DIR / "final_test_routing_results.csv"
    )

    overall_path = (
        REPORTS_DIR / "final_test_routing_summary.csv"
    )

    category_path = (
        REPORTS_DIR / "final_test_routing_by_category.csv"
    )

    test_results.to_csv(ticket_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    category_results.to_csv(category_path, index=False)

    print("\n=== SAVED TEST REPORTS ===")
    print(f"Ticket-level results: {ticket_path}")
    print(f"Overall summary: {overall_path}")
    print(f"Category results: {category_path}")

    print(
        "\nFinal routing policy evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
