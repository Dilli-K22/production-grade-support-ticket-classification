
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_prediction_confidence.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

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


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "Actual",
        "Predicted",
        "Confidence",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Validation confidence report is empty.")

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            "Input report contains missing required values."
        )

    df["Actual"] = df["Actual"].astype(str).str.strip()
    df["Predicted"] = df["Predicted"].astype(str).str.strip()

    df["Confidence"] = pd.to_numeric(
        df["Confidence"],
        errors="raise",
    )

    if not np.isfinite(df["Confidence"]).all():
        raise ValueError("Confidence contains non-finite values.")

    if not df["Confidence"].between(0, 1).all():
        raise ValueError(
            "Confidence values must be between 0 and 1."
        )

    unknown_categories = (
        set(df["Predicted"].unique()) - set(THRESHOLDS)
    )

    if unknown_categories:
        raise ValueError(
            "Predicted categories have no configured thresholds: "
            f"{sorted(unknown_categories)}"
        )

    return df


def evaluate_policy(df):
    df = df.copy()

    # Select threshold using the model's predicted category.
    df["Threshold"] = df["Predicted"].map(THRESHOLDS)

    df["Auto_Routed"] = (
        df["Confidence"] >= df["Threshold"]
    )

    df["Correct"] = (
        df["Actual"] == df["Predicted"]
    )

    total = len(df)
    auto = int(df["Auto_Routed"].sum())
    review = total - auto

    correct_auto = int(
        (df["Auto_Routed"] & df["Correct"]).sum()
    )

    wrong_auto = auto - correct_auto

    coverage = auto / total
    review_rate = review / total

    precision = (
        correct_auto / auto
        if auto > 0
        else float("nan")
    )

    print("\n=== OVERALL ROUTING POLICY ===")
    print(f"Total tickets: {total}")
    print(f"Automatically routed: {auto} ({coverage:.2%})")
    print(f"Sent for human review: {review} ({review_rate:.2%})")

    if auto:
        print(f"Automatic routing precision: {precision:.2%}")
    else:
        print("Automatic routing precision: N/A")

    print(f"Incorrect automatic predictions: {wrong_auto}")

    overall_result = {
        "Category": "OVERALL",
        "Threshold": np.nan,
        "Predicted_Tickets": total,
        "Automatically_Routed": auto,
        "Human_Review": review,
        "Coverage": coverage,
        "Precision": precision,
        "Incorrect_Auto_Routed": wrong_auto,
    }

    category_results = []

    print("\n=== CATEGORY-WISE ROUTING ===")

    for category, threshold in THRESHOLDS.items():
        category_df = df[
            df["Predicted"] == category
        ]

        accepted = category_df[
            category_df["Auto_Routed"]
        ]

        rejected = category_df[
            ~category_df["Auto_Routed"]
        ]

        predicted_count = len(category_df)
        accepted_count = len(accepted)
        rejected_count = len(rejected)

        accepted_correct = int(
            accepted["Correct"].sum()
        )

        incorrect_count = (
            accepted_count - accepted_correct
        )

        category_precision = (
            accepted_correct / accepted_count
            if accepted_count > 0
            else float("nan")
        )

        category_coverage = (
            accepted_count / predicted_count
            if predicted_count > 0
            else float("nan")
        )

        result = {
            "Category": category,
            "Threshold": threshold,
            "Predicted_Tickets": predicted_count,
            "Automatically_Routed": accepted_count,
            "Human_Review": rejected_count,
            "Coverage": category_coverage,
            "Precision": category_precision,
            "Incorrect_Auto_Routed": incorrect_count,
        }

        category_results.append(result)

        print(f"\n{category}")
        print(f"  Threshold: {threshold:.0%}")
        print(f"  Predicted tickets: {predicted_count}")
        print(f"  Automatically routed: {accepted_count}")
        print(f"  Sent for review: {rejected_count}")

        print(
            f"  Auto-routing coverage: "
            f"{category_coverage:.2%}"
            if predicted_count
            else "  Auto-routing coverage: N/A"
        )

        print(
            f"  Auto-routing precision: "
            f"{category_precision:.2%}"
            if accepted_count
            else "  Auto-routing precision: N/A"
        )

        print(
            f"  Incorrect auto-routed: {incorrect_count}"
        )

    return (
        pd.DataFrame([overall_result]),
        pd.DataFrame(category_results),
    )


def main():
    print("Loading validation confidence report...")

    df = load_data()

    print(f"Validation tickets: {len(df)}")

    overall_df, category_df = evaluate_policy(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_path = (
        REPORTS_DIR / "routing_policy_overall.csv"
    )

    category_path = (
        REPORTS_DIR / "routing_policy_by_category.csv"
    )

    overall_df.to_csv(overall_path, index=False)
    category_df.to_csv(category_path, index=False)

    print(f"\nSaved overall results: {overall_path}")
    print(f"Saved category results: {category_path}")

    print("\nRouting policy simulation completed successfully.")


if __name__ == "__main__":
    main()
