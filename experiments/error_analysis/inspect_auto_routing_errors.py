
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "validation_prediction_confidence.csv"
OUTPUT_PATH = REPORTS_DIR / "auto_routing_errors.csv"

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


def validate_data(df):
    required_columns = {
        "Document",
        "Actual",
        "Predicted",
        "Confidence",
    }

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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    validate_data(df)

    df["Threshold"] = df["Predicted"].map(THRESHOLDS)

    if df["Threshold"].isna().any():
        unknown_categories = sorted(
            df.loc[df["Threshold"].isna(), "Predicted"].unique()
        )
        raise ValueError(
            f"Unknown predicted categories found: {unknown_categories}"
        )

    df["Auto_Routed"] = df["Confidence"] >= df["Threshold"]
    df["Correct"] = df["Actual"] == df["Predicted"]

    errors = df[
        df["Auto_Routed"] & (~df["Correct"])
    ].copy()

    errors["Confidence"] = errors["Confidence"].round(4)
    errors["Threshold"] = errors["Threshold"].round(2)

    errors = errors.sort_values(
        by="Confidence",
        ascending=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_PATH, index=False)

    print("\n=== INCORRECT AUTOMATIC ROUTING ERRORS ===")
    print(f"Validation tickets: {len(df)}")
    print(f"Incorrect auto-routed tickets: {len(errors)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\n=== ERROR COUNTS BY PREDICTED CATEGORY ===")

    if errors.empty:
        print("No incorrect auto-routing errors found.")
    else:
        print(errors["Predicted"].value_counts().to_string())

    print("\n=== ERROR COUNTS BY ACTUAL CATEGORY ===")

    if errors.empty:
        print("No incorrect auto-routing errors found.")
    else:
        print(errors["Actual"].value_counts().to_string())

    print("\n=== FIRST 10 INCORRECT AUTO-ROUTED TICKETS ===")

    columns = [
        "Document",
        "Actual",
        "Predicted",
        "Confidence",
        "Threshold",
    ]

    if errors.empty:
        print("No examples available.")
    else:
        print(
            errors[columns]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
