
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = REPORTS_DIR / "optimized_routing_thresholds.csv"
SUMMARY_PATH = REPORTS_DIR / "optimized_routing_summary.csv"
CATEGORY_PATH = REPORTS_DIR / "optimized_routing_validation_by_category.csv"

TARGET_PRECISION = 0.99
MIN_ACCEPTED_PER_CATEGORY = 30

N_SPLITS = 3
RANDOM_STATE = 42


def normalize_text(text):
    """Normalize whitespace without removing ticket content."""
    return " ".join(str(text).split())


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

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing required values."
        )

    df["Document"] = df["Document"].map(normalize_text)
    df["Topic_group"] = (
        df["Topic_group"].astype(str).str.strip()
    )

    if (df["Topic_group"] == "").any():
        raise ValueError(
            f"{dataset_name} contains empty category labels."
        )

    return df


def build_base_model():
    """Build the complete TF-IDF and Logistic Regression pipeline."""

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                max_features=100_000,
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
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_calibrated_model():
    """Calibrate the complete pipeline within CV folds."""

    return CalibratedClassifierCV(
        estimator=build_base_model(),
        method="sigmoid",
        cv=N_SPLITS,
    )


def find_category_thresholds(
    predictions,
    confidence,
    actual,
    categories,
):
    """
    Select thresholds using out-of-fold training predictions.

    For each predicted category, maximize accepted ticket count
    subject to the empirical precision target and minimum sample size.
    """

    df = pd.DataFrame({
        "Predicted": predictions,
        "Confidence": confidence,
        "Actual": actual,
    })

    df["Correct"] = df["Predicted"] == df["Actual"]

    results = []

    for category in categories:
        category_df = df[
            df["Predicted"] == category
        ]

        best_threshold = None
        best_count = 0
        best_correct = 0

        for threshold in sorted(
            category_df["Confidence"].unique()
        ):
            accepted = category_df[
                category_df["Confidence"] >= threshold
            ]

            accepted_count = len(accepted)

            if accepted_count < MIN_ACCEPTED_PER_CATEGORY:
                continue

            correct_count = int(
                accepted["Correct"].sum()
            )

            precision = correct_count / accepted_count

            if precision >= TARGET_PRECISION:
                if accepted_count > best_count:
                    best_threshold = float(threshold)
                    best_count = accepted_count
                    best_correct = correct_count

        if best_threshold is None:
            results.append({
                "Category": category,
                "Threshold": np.nan,
                "OOF_Accepted": 0,
                "OOF_Correct": 0,
                "OOF_Incorrect": 0,
                "OOF_Precision": np.nan,
                "Status": "Human review only",
            })

        else:
            results.append({
                "Category": category,
                "Threshold": best_threshold,
                "OOF_Accepted": best_count,
                "OOF_Correct": best_correct,
                "OOF_Incorrect": best_count - best_correct,
                "OOF_Precision": (
                    best_correct / best_count
                ),
                "Status": "Threshold selected",
            })

    return pd.DataFrame(results)


def evaluate_fixed_policy(
    model_name,
    y_true,
    probabilities,
    classes,
    policy,
):
    """
    Evaluate previously selected thresholds on validation data.
    This function does not optimize or modify thresholds.
    """

    predicted_indices = probabilities.argmax(axis=1)

    predictions = np.asarray(classes)[predicted_indices]
    confidence = probabilities.max(axis=1)

    threshold_map = dict(
        zip(policy["Category"], policy["Threshold"])
    )

    thresholds = pd.Series(predictions).map(
        threshold_map
    ).to_numpy(dtype=float)

    auto_routed = (
        np.isfinite(thresholds)
        & (confidence >= np.nan_to_num(
            thresholds,
            nan=np.inf,
        ))
    )

    correct = predictions == np.asarray(y_true)

    total = len(y_true)
    auto_count = int(auto_routed.sum())
    review_count = total - auto_count

    correct_auto = int(
        (auto_routed & correct).sum()
    )

    wrong_auto = auto_count - correct_auto

    overall_precision = (
        correct_auto / auto_count
        if auto_count
        else np.nan
    )

    summary = {
        "Model": model_name,
        "Total_Validation_Tickets": total,
        "Auto_Routed": auto_count,
        "Human_Review": review_count,
        "Automation_Coverage": auto_count / total,
        "Correct_Auto": correct_auto,
        "Wrong_Auto": wrong_auto,
        "Auto_Precision": overall_precision,
    }

    print(f"\n=== {model_name.upper()} VALIDATION RESULTS ===")
    print(f"Total validation tickets: {total}")
    print(f"Automatically routed: {auto_count}")
    print(f"Human review: {review_count}")
    print(f"Automation coverage: {auto_count / total:.2%}")
    print(f"Correct automatic predictions: {correct_auto}")
    print(f"Incorrect automatic predictions: {wrong_auto}")

    if auto_count:
        print(f"Overall routing precision: {overall_precision:.2%}")
    else:
        print("Overall routing precision: N/A")

    category_results = []

    for category in policy["Category"]:
        category_mask = predictions == category
        accepted_mask = auto_routed & category_mask

        predicted_count = int(category_mask.sum())
        accepted_count = int(accepted_mask.sum())

        correct_count = int(
            (accepted_mask & correct).sum()
        )

        wrong_count = accepted_count - correct_count

        category_precision = (
            correct_count / accepted_count
            if accepted_count
            else np.nan
        )

        category_coverage = (
            accepted_count / predicted_count
            if predicted_count
            else np.nan
        )

        selected_threshold = threshold_map[category]

        category_results.append({
            "Model": model_name,
            "Category": category,
            "Threshold": selected_threshold,
            "Validation_Predicted": predicted_count,
            "Validation_Accepted": accepted_count,
            "Validation_Correct": correct_count,
            "Validation_Incorrect": wrong_count,
            "Validation_Coverage": category_coverage,
            "Validation_Precision": category_precision,
        })

    return summary, pd.DataFrame(category_results)


def run_experiment(
    name,
    estimator,
    X_train,
    y_train,
    X_validation,
    y_validation,
    categories,
    cv,
):
    print(f"\n{'=' * 65}")
    print(f"STARTING {name.upper()} EXPERIMENT")
    print(f"{'=' * 65}")

    print("\nGenerating out-of-fold training probabilities...")

    oof_probabilities = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )

    classes = np.asarray(categories)

    if oof_probabilities.shape[1] != len(classes):
        raise ValueError(
            f"{name}: unexpected number of probability columns."
        )

    oof_predictions = classes[
        oof_probabilities.argmax(axis=1)
    ]

    oof_confidence = oof_probabilities.max(axis=1)

    print("Selecting category-specific thresholds...")

    policy = find_category_thresholds(
        oof_predictions,
        oof_confidence,
        y_train.to_numpy(),
        categories,
    )

    print("\nSelected thresholds from training OOF predictions:")

    print(
        policy.to_string(
            index=False,
            formatters={
                "Threshold": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "Disabled"
                ),
                "OOF_Precision": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print("\nFitting final experiment model on full training data...")

    estimator.fit(X_train, y_train)

    print("Predicting validation probabilities...")

    validation_probabilities = estimator.predict_proba(
        X_validation
    )

    fitted_classes = estimator.classes_

    if not np.array_equal(
        fitted_classes,
        classes,
    ):
        raise ValueError(
            f"{name}: fitted model category order does not match."
        )

    summary, category_results = evaluate_fixed_policy(
        name,
        y_validation.to_numpy(),
        validation_probabilities,
        fitted_classes,
        policy,
    )

    policy.insert(0, "Model", name)

    return (
        policy,
        summary,
        category_results,
    )


def main():
    print("Loading training and validation datasets...")

    train_df = load_dataset(TRAIN_PATH, "Training")
    validation_df = load_dataset(
        VALIDATION_PATH,
        "Validation",
    )

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_validation = validation_df["Document"]
    y_validation = validation_df["Topic_group"]

    categories = sorted(y_train.unique())

    if set(categories) != set(y_validation.unique()):
        raise ValueError(
            "Training and validation category sets do not match."
        )

    if y_train.value_counts().min() < N_SPLITS:
        raise ValueError(
            "At least one training category has too few records "
            "for the configured cross-validation folds."
        )

    print(f"Training tickets: {len(train_df)}")
    print(f"Validation tickets: {len(validation_df)}")
    print(f"Categories: {len(categories)}")
    print(f"Target precision: {TARGET_PRECISION:.0%}")
    print(
        f"Minimum accepted per category: "
        f"{MIN_ACCEPTED_PER_CATEGORY}"
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    experiments = [
        ("Original", build_base_model()),
        ("Calibrated", build_calibrated_model()),
    ]

    all_policies = []
    all_summaries = []
    all_category_results = []

    for (
        name,
        estimator,
    ) in experiments:

        policy, summary, category_results = run_experiment(
            name=name,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            categories=categories,
            cv=cv,
        )

        all_policies.append(policy)
        all_summaries.append(summary)
        all_category_results.append(category_results)

    combined_policy = pd.concat(
        all_policies,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(all_summaries)

    category_df = pd.concat(
        all_category_results,
        ignore_index=True,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_policy.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
    )

    print("\n=== FINAL VALIDATION COMPARISON ===")

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "Automation_Coverage": "{:.2%}".format,
                "Auto_Precision": lambda x: (
                    f"{x:.2%}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print("\n=== SAVED REPORTS ===")
    print(f"Threshold policies: {OUTPUT_PATH}")
    print(f"Overall validation results: {SUMMARY_PATH}")
    print(f"Category validation results: {CATEGORY_PATH}")

    print(
        "\nThreshold optimization completed successfully."
    )


if __name__ == "__main__":
    main()
