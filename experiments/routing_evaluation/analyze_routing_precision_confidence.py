
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportion_confint


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test_routing_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test_routing_precision_confidence.csv"
)

CONFIDENCE_LEVEL = 0.95
CATEGORIES = [
    "Access",
    "Administrative rights",
    "HR Support",
    "Hardware",
    "Internal Project",
    "Miscellaneous",
    "Purchase",
    "Storage",
]


def parse_boolean_column(series, column_name):
    """Safely convert CSV boolean values into actual booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = series.astype(str).str.strip().str.lower().map(
        mapping
    )

    if converted.isna().any():
        raise ValueError(
            f"Invalid boolean values found in {column_name}."
        )

    return converted.astype(bool)


def calculate_interval(correct, total):
    """Calculate a Wilson confidence interval for a proportion."""

    if total == 0:
        return None, None

    lower, upper = proportion_confint(
        count=correct,
        nobs=total,
        alpha=1 - CONFIDENCE_LEVEL,
        method="wilson",
    )

    return float(lower), float(upper)


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Test routing results not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "Actual",
        "Predicted",
        "Auto_Routed",
        "Correct",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Test routing results are empty.")

    df["Auto_Routed"] = parse_boolean_column(
        df["Auto_Routed"],
        "Auto_Routed",
    )

    df["Correct"] = parse_boolean_column(
        df["Correct"],
        "Correct",
    )

    # Verify that the correctness field matches actual predictions.
    expected_correct = df["Actual"] == df["Predicted"]

    if not df["Correct"].equals(expected_correct):
        raise ValueError(
            "Correct column does not match Actual and Predicted."
        )

    return df


def analyze_overall_precision(df):
    total_tickets = len(df)

    auto_mask = df["Auto_Routed"]

    total_auto = int(auto_mask.sum())

    correct_auto = int(
        (auto_mask & df["Correct"]).sum()
    )

    wrong_auto = total_auto - correct_auto

    precision = (
        correct_auto / total_auto
        if total_auto
        else None
    )

    lower, upper = calculate_interval(
        correct_auto,
        total_auto,
    )

    review_count = total_tickets - total_auto

    print("\n=== OVERALL ROUTING PRECISION ===")
    print(f"Total test tickets: {total_tickets}")
    print(f"Automatically routed: {total_auto}")
    print(f"Human review: {review_count}")
    print(f"Correct automatic predictions: {correct_auto}")
    print(f"Incorrect automatic predictions: {wrong_auto}")

    if precision is not None:
        print(f"Observed precision: {precision:.2%}")
        print(
            f"Wilson {CONFIDENCE_LEVEL:.0%} confidence interval: "
            f"{lower:.2%} - {upper:.2%}"
        )
    else:
        print("Observed precision: N/A")
        print("Confidence interval: N/A")

    return {
        "Total_Test_Tickets": total_tickets,
        "Auto_Routed": total_auto,
        "Human_Review": review_count,
        "Correct_Auto": correct_auto,
        "Wrong_Auto": wrong_auto,
        "Precision": precision,
        "CI_Lower": lower,
        "CI_Upper": upper,
    }


def analyze_category_precision(df):
    results = []

    for category in CATEGORIES:
        category_mask = df["Predicted"] == category

        auto_mask = (
            category_mask & df["Auto_Routed"]
        )

        predicted_count = int(category_mask.sum())
        auto_count = int(auto_mask.sum())

        correct_count = int(
            (auto_mask & df["Correct"]).sum()
        )

        wrong_count = auto_count - correct_count

        precision = (
            correct_count / auto_count
            if auto_count
            else None
        )

        lower, upper = calculate_interval(
            correct_count,
            auto_count,
        )

        results.append({
            "Category": category,
            "Predicted_Tickets": predicted_count,
            "Auto_Routed": auto_count,
            "Human_Review": predicted_count - auto_count,
            "Correct_Auto": correct_count,
            "Wrong_Auto": wrong_count,
            "Precision": precision,
            "CI_Lower": lower,
            "CI_Upper": upper,
        })

    results_df = pd.DataFrame(results)

    print("\n=== CATEGORY-WISE ROUTING CONFIDENCE ===")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Precision": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
                "CI_Lower": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
                "CI_Upper": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    return results_df


def main():
    print("Loading final test routing results...")

    df = load_results()

    print(f"Loaded test predictions: {len(df)}")

    overall_results = analyze_overall_precision(df)

    category_results = analyze_category_precision(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    overall_output_path = (
        OUTPUT_PATH.parent
        / "final_test_overall_precision_confidence.csv"
    )

    pd.DataFrame([overall_results]).to_csv(
        overall_output_path,
        index=False,
    )

    print("\n=== REPORTS SAVED ===")
    print(f"Category confidence: {OUTPUT_PATH}")
    print(f"Overall confidence: {overall_output_path}")

    print("\nStatistical confidence analysis completed.")


if __name__ == "__main__":
    main()
