import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from src.config import MODEL_PATH, PROCESSED_DATA_DIR, REPORTS_DIR


TEST_PATH = PROCESSED_DATA_DIR / "test.csv"


def main() -> None:
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load untouched test set
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df["Document"]
    y_test = test_df["Topic_group"]

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    # Prepare structured metrics
    metrics = {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
    }

    # Save evaluation report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Evaluation metrics saved to: {metrics_path}")

    # Detailed metrics
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save confusion matrix
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    confusion_path = REPORTS_DIR / "confusion_matrix.csv"

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(confusion_path, index_label="Actual")

    print(f"Confusion matrix saved to: {confusion_path}")


if __name__ == "__main__":
    main()