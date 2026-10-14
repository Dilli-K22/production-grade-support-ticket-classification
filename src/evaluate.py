import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from src.config import MODEL_PATH, PROCESSED_DATA_DIR


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

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")

    # Detailed metrics
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
        )
    )

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()