
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

VALIDATION_PATH = DATA_DIR / "validation.csv"
OUTPUT_PATH = REPORTS_DIR / "validation_text_quality.csv"


PATTERNS = {
    "email_greeting": r"\b(?:dear|hello|hi|hey)\b",
    "email_closing": r"\b(?:regards|thank you|thanks|kind regards)\b",
    "forwarded_or_reply": r"\b(?:forwarded message|original message|re:)\b",
    "approval_terms": r"\b(?:approve|approved|approval|approvals)\b",
    "workflow_terms": r"\b(?:assigned|assignee|ticket|request|status)\b",
}


def validate_data(df):
    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Validation dataset is empty.")


def calculate_text_metrics(df):
    text = df["Document"].fillna("").astype(str)

    df["text_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len()

    def word_statistics(value):
        words = value.lower().split()

        if not words:
            return pd.Series({
                "unique_word_ratio": 0.0,
                "repeated_word_count": 0,
            })

        counts = Counter(words)

        return pd.Series({
            "unique_word_ratio": len(counts) / len(words),
            "repeated_word_count": sum(
                count - 1 for count in counts.values() if count > 1
            ),
        })

    word_metrics = text.apply(word_statistics)

    df["unique_word_ratio"] = word_metrics["unique_word_ratio"]
    df["repeated_word_count"] = word_metrics["repeated_word_count"]

    df["empty_text"] = text.str.strip().eq("")
    df["very_short_text"] = df["word_count"] < 5
    df["long_text"] = df["word_count"] > 500

    for name, pattern in PATTERNS.items():
        df[f"count_{name}"] = text.apply(
            lambda value: len(
                re.findall(pattern, value, flags=re.IGNORECASE)
            )
        )

    return df


def main():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    df = pd.read_csv(VALIDATION_PATH)
    validate_data(df)

    df = calculate_text_metrics(df)

    print(f"Validation tickets: {len(df)}")
    print(f"Empty text: {df['empty_text'].sum()}")
    print(f"Very short text (<5 words): {df['very_short_text'].sum()}")
    print(f"Long text (>500 words): {df['long_text'].sum()}")

    print("\n--- Text length summary ---")
    print(
        df[
            [
                "text_length",
                "word_count",
                "unique_word_ratio",
                "repeated_word_count",
            ]
        ]
        .describe()
        .round(2)
    )

    print("\n--- Text patterns ---")

    for name in PATTERNS:
        column = f"count_{name}"
        tickets_with_pattern = int((df[column] > 0).sum())
        total_occurrences = int(df[column].sum())

        print(
            f"{name}: {tickets_with_pattern} tickets, "
            f"{total_occurrences} total occurrences"
        )

    print("\n--- Average text length by category ---")

    category_summary = (
        df.groupby("Topic_group")["word_count"]
        .agg(["count", "mean", "median", "max"])
        .round(2)
        .sort_values("mean", ascending=False)
    )

    print(category_summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved text-quality report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
