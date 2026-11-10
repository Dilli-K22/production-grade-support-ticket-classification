
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

THRESHOLDS = [0.70, 0.80, 0.90]


def parse_boolean_column(series):
    """Safely convert boolean-like values into actual booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series

    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    invalid_values = set(normalized.unique()) - set(mapping)

    if invalid_values:
        raise ValueError(
            f"Invalid Correct values found: {sorted(invalid_values)}"
        )

    return normalized.map(mapping).astype(bool)


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"Confidence", "Correct"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Input confidence report is empty.")

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

    df["Correct"] = parse_boolean_column(df["Correct"])

    return df


def evaluate_thresholds(df):
    results = []

    print("\n--- Automatic routing threshold comparison ---")

    for threshold in THRESHOLDS:
        auto_route = df["Confidence"] >= threshold
        review = ~auto_route

        routed_count = int(auto_route.sum())
        review_count = int(review.sum())

        coverage = float(auto_route.mean())
        review_rate = float(review.mean())

        routed_accuracy = (
            float(df.loc[auto_route, "Correct"].mean())
            if routed_count > 0
            else float("nan")
        )

        routed_errors = int(
            (~df.loc[auto_route, "Correct"]).sum()
        )

        result = {
            "Threshold": threshold,
            "Total_Tickets": len(df),
            "Automatically_Routed": routed_count,
            "Human_Review": review_count,
            "Coverage": coverage,
            "Review_Rate": review_rate,
            "Routed_Accuracy": routed_accuracy,
            "Routed_Errors": routed_errors,
        }

        results.append(result)

        print(f"\nThreshold: {threshold:.0%}")
        print(f"Automatically routed: {routed_count}")
        print(f"Human review queue: {review_count}")
        print(f"Automatic routing coverage: {coverage:.2%}")
        print(f"Human review rate: {review_rate:.2%}")

        if routed_count:
            print(
                "Accuracy of automatically routed tickets: "
                f"{routed_accuracy:.2%}"
            )
        else:
            print("Accuracy of automatically routed tickets: N/A")

        print(f"Incorrect automatic predictions: {routed_errors}")

    return pd.DataFrame(results)


def main():
    print("Loading validation confidence report...")

    df = load_data()

    print(f"Validation tickets: {len(df)}")
    print(f"Overall accuracy: {df['Correct'].mean():.2%}")

    results_df = evaluate_thresholds(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORTS_DIR / "review_threshold_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved threshold comparison: {output_path}")
    print("\nReview threshold evaluation completed successfully.")


if __name__ == "__main__":
    main()
