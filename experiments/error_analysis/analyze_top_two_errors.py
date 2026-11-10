
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

OUTPUT_PATH = REPORTS_DIR / "top_two_auto_routing_errors.csv"

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


def normalize_text(text):
    return " ".join(str(text).split())


def validate_data(df, file_path):
    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    if df["Topic_group"].isna().any():
        raise ValueError(f"Missing category labels found in {file_path}")


def main():
    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    validate_data(train_df, TRAIN_PATH)
    validate_data(validation_df, VALIDATION_PATH)

    X_train = train_df["Document"].fillna("").astype(str).map(normalize_text)
    y_train = train_df["Topic_group"]

    X_validation = (
        validation_df["Document"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )
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

    missing_thresholds = set(classes) - set(THRESHOLDS)

    if missing_thresholds:
        raise ValueError(
            f"Missing thresholds for categories: {sorted(missing_thresholds)}"
        )

    results = []

    for index, row in validation_df.iterrows():
        ranked_indices = probabilities[index].argsort()[::-1]

        first_index = ranked_indices[0]
        second_index = ranked_indices[1]

        predicted = classes[first_index]
        confidence = probabilities[index][first_index]

        second_predicted = classes[second_index]
        second_confidence = probabilities[index][second_index]

        threshold = THRESHOLDS[predicted]

        auto_routed = confidence >= threshold
        actual = y_validation.iloc[index]
        correct = predicted == actual

        if auto_routed and not correct:
            results.append({
                "Document": row["Document"],
                "Actual": actual,
                "Predicted": predicted,
                "Confidence": confidence,
                "Threshold": threshold,
                "Second_Predicted": second_predicted,
                "Second_Confidence": second_confidence,
                "Confidence_Gap": confidence - second_confidence,
                "Actual_In_Top_Two": actual in [
                    predicted,
                    second_predicted,
                ],
            })

    columns = [
        "Document",
        "Actual",
        "Predicted",
        "Confidence",
        "Threshold",
        "Second_Predicted",
        "Second_Confidence",
        "Confidence_Gap",
        "Actual_In_Top_Two",
    ]

    errors = pd.DataFrame(results, columns=columns)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_PATH, index=False)

    print("\n=== TOP TWO ANALYSIS ===")
    print(f"Validation tickets: {len(validation_df)}")
    print(f"Incorrect auto-routed tickets: {len(errors)}")
    print(f"Saved to: {OUTPUT_PATH}")

    if errors.empty:
        print("\nNo incorrect auto-routed tickets found.")
        return

    print("\n=== SECOND-BEST PREDICTION COUNTS ===")
    print(errors["Second_Predicted"].value_counts().to_string())

    print("\n=== ACTUAL CATEGORY IN TOP TWO ===")
    print(errors["Actual_In_Top_Two"].value_counts().to_string())

    top_two_count = int(errors["Actual_In_Top_Two"].sum())

    print(
        f"\nCorrect category appeared in top two: "
        f"{top_two_count}/{len(errors)} "
        f"({top_two_count / len(errors):.2%})"
    )

    print("\n=== CONFIDENCE GAP STATISTICS ===")
    print(errors["Confidence_Gap"].describe().to_string())

    print("\n=== FIRST 10 EXAMPLES ===")

    display_columns = [
        "Actual",
        "Predicted",
        "Confidence",
        "Second_Predicted",
        "Second_Confidence",
        "Confidence_Gap",
        "Actual_In_Top_Two",
    ]

    print(errors[display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
