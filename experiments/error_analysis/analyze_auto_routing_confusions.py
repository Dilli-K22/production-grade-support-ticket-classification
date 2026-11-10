
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "auto_routing_errors.csv"

CONFUSION_OUTPUT_PATH = REPORTS_DIR / "auto_routing_confusion_pairs.csv"
CONFIDENCE_OUTPUT_PATH = REPORTS_DIR / "auto_routing_confidence_by_actual.csv"


def validate_data(df):
    required_columns = {"Actual", "Predicted", "Confidence"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["Actual"].isna().any() or df["Predicted"].isna().any():
        raise ValueError("Missing category values found.")

    if not pd.api.types.is_numeric_dtype(df["Confidence"]):
        raise ValueError("Confidence column must be numeric.")

    if not df["Confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1.")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    validate_data(df)

    print("\n=== ACTUAL vs PREDICTED CATEGORY CONFUSIONS ===")

    confusion_pairs = (
        df.groupby(["Actual", "Predicted"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    if confusion_pairs.empty:
        print("No category confusion pairs found.")
    else:
        print(confusion_pairs.to_string(index=False))

    print("\n=== TOTAL ERRORS BY CATEGORY PAIR ===")

    for _, row in confusion_pairs.iterrows():
        print(
            f"{row['Actual']} -> {row['Predicted']}: "
            f"{row['Count']} ticket(s)"
        )

    print("\n=== MOST CONFUSING CATEGORY PAIRS ===")

    top_pairs = confusion_pairs.head(10)

    for _, row in top_pairs.iterrows():
        percentage = row["Count"] / len(df) * 100

        print(
            f"{row['Actual']} -> {row['Predicted']}: "
            f"{row['Count']} ({percentage:.2f}% of all errors)"
        )

    print("\n=== CONFIDENCE OF INCORRECT AUTO-ROUTED TICKETS ===")
    print(df["Confidence"].describe().to_string())

    print("\n=== AVERAGE CONFIDENCE BY ACTUAL CATEGORY ===")

    confidence_by_actual = (
        df.groupby("Actual")["Confidence"]
        .agg(["count", "mean", "min", "max"])
        .sort_values("count", ascending=False)
    )

    print(confidence_by_actual.to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    confusion_pairs.to_csv(CONFUSION_OUTPUT_PATH, index=False)
    confidence_by_actual.to_csv(CONFIDENCE_OUTPUT_PATH)

    print(f"\nSaved confusion pairs: {CONFUSION_OUTPUT_PATH}")
    print(f"Saved confidence analysis: {CONFIDENCE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
