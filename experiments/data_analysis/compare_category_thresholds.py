
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "validation_prediction_confidence.csv"
OUTPUT_PATH = REPORTS_DIR / "category_threshold_comparison.csv"

THRESHOLDS = [0.70, 0.80, 0.90, 0.95]


def validate_data(df):
    required_columns = {"Correct", "Confidence", "Predicted"}
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
        raise ValueError("Confidence column must be numeric.")

    if not df["Confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1.")

    valid_boolean_values = {True, False, 1, 0, "True", "False", "1", "0"}

    if not df["Correct"].isin(valid_boolean_values).all():
        raise ValueError("Correct column contains invalid boolean values.")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    validate_data(df)

    # Safely convert CSV boolean values.
    df["Correct"] = (
        df["Correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if df["Correct"].isna().any():
        raise ValueError("Could not convert all Correct values to booleans.")

    results = []

    print(f"Validation tickets: {len(df)}")
    print("\n--- Category-specific threshold analysis ---")

    for category in sorted(df["Predicted"].unique()):
        category_df = df[df["Predicted"] == category]

        print(f"\nCategory: {category}")
        print(f"Total predicted tickets: {len(category_df)}")

        for threshold in THRESHOLDS:
            accepted = category_df[
                category_df["Confidence"] >= threshold
            ]

            accepted_count = len(accepted)

            if accepted_count == 0:
                precision = float("nan")
                coverage = 0.0
                incorrect = 0
            else:
                precision = accepted["Correct"].mean()
                coverage = accepted_count / len(category_df)
                incorrect = int((~accepted["Correct"]).sum())

            results.append({
                "category": category,
                "threshold": threshold,
                "total_predicted": len(category_df),
                "auto_routed": accepted_count,
                "coverage": coverage,
                "precision": precision,
                "incorrect": incorrect,
            })

            precision_display = (
                f"{precision:.2%}" if pd.notna(precision) else "N/A"
            )

            print(
                f"  Threshold: {threshold:.0%} | "
                f"Auto-routed: {accepted_count:4d} | "
                f"Coverage: {coverage:6.2%} | "
                f"Precision: {precision_display:>6} | "
                f"Incorrect: {incorrect}"
            )

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved detailed results: {OUTPUT_PATH}")

    print("\n--- Summary: 80% threshold ---")

    summary = results_df[
        results_df["threshold"] == 0.80
    ][[
        "category",
        "auto_routed",
        "coverage",
        "precision",
        "incorrect",
    ]]

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()
