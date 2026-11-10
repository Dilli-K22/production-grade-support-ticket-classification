
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_CLASSES = ["Hardware", "Access", "HR Support"]
TOP_N_FEATURES = 20


def load_training_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(train_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    train_df = train_df.dropna(
        subset=["Document", "Topic_group"]
    ).copy()

    train_df["Document"] = train_df["Document"].astype(str).str.strip()
    train_df["Topic_group"] = train_df["Topic_group"].astype(str).str.strip()

    train_df = train_df[
        (train_df["Document"] != "")
        & (train_df["Topic_group"] != "")
    ]

    if train_df.empty:
        raise ValueError("Training dataset is empty after cleaning.")

    missing_targets = set(TARGET_CLASSES) - set(train_df["Topic_group"])

    if missing_targets:
        raise ValueError(
            f"Target categories missing from training data: "
            f"{sorted(missing_targets)}"
        )

    return train_df


def build_model():
    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=300_000,
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


def analyze_features(model):
    classifier = model.named_steps["classifier"]
    feature_union = model.named_steps["features"]

    feature_names = feature_union.get_feature_names_out()
    classes = list(classifier.classes_)

    results = []

    for target in TARGET_CLASSES:
        class_index = classes.index(target)
        coefficients = classifier.coef_[class_index]

        top_indices = coefficients.argsort()[-TOP_N_FEATURES:][::-1]

        print(f"\n{'=' * 65}")
        print(f"Top features associated with {target}")
        print(f"{'=' * 65}")

        for rank, index in enumerate(top_indices, start=1):
            feature = feature_names[index]
            coefficient = float(coefficients[index])

            print(
                f"{rank:>2}. {feature:<40} {coefficient:.4f}"
            )

            results.append({
                "Category": target,
                "Rank": rank,
                "Feature": feature,
                "Coefficient": coefficient,
            })

    return pd.DataFrame(results)


def main():
    print("Loading training dataset...")
    train_df = load_training_data()

    print(f"Training records: {len(train_df)}")
    print(f"Training categories: {train_df['Topic_group'].nunique()}")

    print("\nBuilding word + character TF-IDF model...")
    model = build_model()

    print("Training model...")
    model.fit(
        train_df["Document"],
        train_df["Topic_group"],
    )

    print("\nAnalyzing important features...")
    results_df = analyze_features(model)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "important_model_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved feature analysis: {output_path}")
    print("\nInterpretability analysis completed successfully.")


if __name__ == "__main__":
    main()
