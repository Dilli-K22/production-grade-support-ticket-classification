
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "validation_prediction_confidence.csv"

PREDICTED_OUTPUT_PATH = REPORTS_DIR / "category_routing_precision.csv"
ACTUAL_OUTPUT_PATH = REPORTS_DIR / "category_routing_coverage.csv"

THRESHOLD = 0.80


def validate_data(df):
    required_columns = {"Correct", "Confidence", "Predicted", "Actual"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if not pd.api.types.is_numeric_dtype(df["Confidence"]):
        raise ValueError("Confidence column must contain numeric values.")

    if not df["Confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1.")

    if not df["Correct"].isin([True, False, 1, 0, "True", "False"]).all():
        raise ValueError("Correct column contains invalid boolean values.")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    validate_data(df)

    # Safely convert boolean values, including CSV text values.
    df["Correct"] = (
        df["Correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if df["Correct"].isna().any():
        raise ValueError("Could not convert all Correct values to booleans.")

    df["Auto_Routed"] = df["Confidence"] >= THRESHOLD

    auto_df = df[df["Auto_Routed"]].copy()

    print(f"Confidence threshold: {THRESHOLD:.0%}")
    print(f"Total validation tickets: {len(df)}")
    print(f"Automatically routed: {len(auto_df)}")
    print(f"Sent for human review: {len(df) - len(auto_df)}")

    # Quality of automatically routed predictions by predicted category.
    if auto_df.empty:
        predicted_summary = pd.DataFrame(
            columns=[
                "auto_routed",
                "correct",
                "incorrect",
                "routing_precision",
            ]
        )
        predicted_summary.index.name = "Predicted"
    else:
        predicted_summary = (
            auto_df.groupby("Predicted")
            .agg(
                auto_routed=("Correct", "size"),
                correct=("Correct", "sum"),
            )
        )

        predicted_summary["incorrect"] = (
            predicted_summary["auto_routed"]
            - predicted_summary["correct"]
        )

        predicted_summary["routing_precision"] = (
            predicted_summary["correct"]
            / predicted_summary["auto_routed"]
        )

    print("\n--- Automatic routing quality by predicted category ---")
    print(
        predicted_summary.sort_values("routing_precision")
        .to_string(float_format=lambda x: f"{x:.4f}")
    )

    # Percentage of each actual category that gets automatically routed.
    actual_summary = (
        df.groupby("Actual")
        .agg(
            total_tickets=("Correct", "size"),
            auto_routed=("Auto_Routed", "sum"),
        )
    )

    actual_summary["reviewed"] = (
        actual_summary["total_tickets"]
        - actual_summary["auto_routed"]
    )

    actual_summary["routing_coverage"] = (
        actual_summary["auto_routed"]
        / actual_summary["total_tickets"]
    )

    print("\n--- Automatic routing coverage by actual category ---")
    print(
        actual_summary.sort_values("routing_coverage")
        .to_string(float_format=lambda x: f"{x:.4f}")
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predicted_summary.to_csv(PREDICTED_OUTPUT_PATH)
    actual_summary.to_csv(ACTUAL_OUTPUT_PATH)

    print(f"\nPrecision report saved: {PREDICTED_OUTPUT_PATH}")
    print(f"Coverage report saved: {ACTUAL_OUTPUT_PATH}")

    print("\nNote: routing precision and coverage are different measures.")
    print("These are validation results, not production guarantees.")


if __name__ == "__main__":
    main()
