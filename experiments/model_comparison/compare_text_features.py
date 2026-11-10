
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
BASELINE_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_CLASSES = [
    "Administrative rights",
    "Hardware",
    "Access",
]

EXAMPLE_TICKETS = [
    "I cannot connect to the company VPN from my laptop",
    "I need access to the company VPN",
    "I cannot log in to my account",
    "My laptop will not connect to the monitor",
    "I cannot access the company network remotely",
]


def load_dataset(path, dataset_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {"Document", "Topic_group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")

    if df["Topic_group"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing target labels."
        )

    df["Document"] = df["Document"].fillna("").astype(str)

    if df["Topic_group"].nunique() < 2:
        raise ValueError(
            f"{dataset_name} must contain at least two categories."
        )

    return df


def build_experiments():
    return {
        "word_only": TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
        ),

        "word_sublinear": TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),

        "word_and_character": FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
        ]),
    }


def build_model(features):
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


def evaluate_model(name, model, X_val, y_val):
    predictions = model.predict(X_val)

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro_F1": f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "Weighted_F1": f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    for category in TARGET_CLASSES:
        if category not in report:
            raise ValueError(
                f"Category '{category}' is missing from validation results."
            )

        result[f"{category}_F1"] = report[category]["f1-score"]

    print(f"\n--- {name} ---")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Macro F1: {result['Macro_F1']:.4f}")
    print(f"Weighted F1: {result['Weighted_F1']:.4f}")

    for category in TARGET_CLASSES:
        print(
            f"{category} F1: "
            f"{result[f'{category}_F1']:.4f}"
        )

    return result


def compare_example_predictions(baseline_model, combined_model):
    if not hasattr(baseline_model, "predict_proba"):
        raise ValueError(
            "Saved baseline model does not support predict_proba."
        )

    if set(baseline_model.classes_) != set(combined_model.classes_):
        raise ValueError(
            "Baseline and combined model have different category labels."
        )

    results = []

    print("\n" + "=" * 65)
    print("SAVED BASELINE VS WORD + CHARACTER MODEL")
    print("=" * 65)

    for ticket in EXAMPLE_TICKETS:
        print(f"\nTicket: {ticket}")

        for model_name, model in [
            ("Saved baseline", baseline_model),
            ("Word + character", combined_model),
        ]:
            probabilities = model.predict_proba([ticket])[0]
            classes = model.classes_

            ranking = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )

            predicted_category, confidence = ranking[0]

            print(
                f"{model_name}: {predicted_category} "
                f"({confidence:.2%})"
            )

            results.append({
                "Ticket": ticket,
                "Model": model_name,
                "Predicted_Category": predicted_category,
                "Confidence": float(confidence),
                "Second_Category": ranking[1][0],
                "Second_Confidence": float(ranking[1][1]),
            })

    return pd.DataFrame(results)


def main():
    print("Loading datasets...")

    train_df = load_dataset(TRAIN_PATH, "Training")
    validation_df = load_dataset(VALIDATION_PATH, "Validation")

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_val = validation_df["Document"]
    y_val = validation_df["Topic_group"]

    if set(y_train.unique()) != set(y_val.unique()):
        raise ValueError(
            "Training and validation category sets do not match."
        )

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    print("\nBuilding feature experiments...")

    experiments = build_experiments()
    trained_models = {}
    metric_results = []

    for name, features in experiments.items():
        print(f"\nTraining experiment: {name}")

        model = build_model(features)
        model.fit(X_train, y_train)

        trained_models[name] = model

        result = evaluate_model(
            name,
            model,
            X_val,
            y_val,
        )

        metric_results.append(result)

    metrics_df = pd.DataFrame(metric_results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORTS_DIR / "text_feature_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved experiment metrics: {metrics_path}")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Saved baseline model not found: {BASELINE_PATH}"
        )

    print("\nLoading saved baseline model...")
    baseline_model = joblib.load(BASELINE_PATH)

    combined_model = trained_models["word_and_character"]

    example_results = compare_example_predictions(
        baseline_model,
        combined_model,
    )

    examples_path = (
        REPORTS_DIR / "text_feature_example_predictions.csv"
    )

    example_results.to_csv(examples_path, index=False)

    print(f"\nSaved example predictions: {examples_path}")

    print("\nText feature comparison completed successfully.")


if __name__ == "__main__":
    main()
