import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.config import MODEL_PATH, PROCESSED_DATA_DIR

DATA_DIR = PROCESSED_DATA_DIR

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"


def main() -> None:
    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    # Separate text and labels
    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_validation = validation_df["Document"]
    y_validation = validation_df["Topic_group"]

    # Build ML pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")

    # Generate predictions
    y_pred = pipeline.predict(X_validation)

    # Create error analysis dataframe
    error_df = pd.DataFrame(
        {
            "Document": X_validation,
            "Actual": y_validation,
            "Predicted": y_pred,
        }
    )

    # Inspect HR Support incorrectly predicted as Hardware
    hr_to_hardware = error_df[
        (error_df["Actual"] == "HR Support")
        & (error_df["Predicted"] == "Hardware")
    ]

    print("\nHR Support → Hardware errors:")
    print(hr_to_hardware.head(10).to_string(index=False))

    hardware_to_hr = error_df[
        (error_df["Actual"] == "Hardware")
        & (error_df["Predicted"] == "HR Support")
    ]

    print("\nHardware → HR Support errors:")
    print(hardware_to_hr.head(10).to_string(index=False))

    admin_to_hardware = error_df[
        (error_df["Actual"] == "Administrative rights")
        & (error_df["Predicted"] == "Hardware")
    ]

    print("\nAdministrative rights → Hardware errors:")
    print(admin_to_hardware.head(10).to_string(index=False))

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            y_pred,
            digits=4,
        )
    )
    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_validation,
        y_pred,
        xticks_rotation=45,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()