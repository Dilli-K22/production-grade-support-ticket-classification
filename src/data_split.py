import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


RANDOM_STATE = 42

DATA_PATH = RAW_DATA_DIR / "all_tickets_processed_improved_v3.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR


def main() -> None:
    # Load raw dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df["Document"]
    y = df["Topic_group"]

    # First split: 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits
    pd.DataFrame({
        "Document": X_train,
        "Topic_group": y_train,
    }).to_csv(OUTPUT_DIR / "train.csv", index=False)

    pd.DataFrame({
        "Document": X_val,
        "Topic_group": y_val,
    }).to_csv(OUTPUT_DIR / "validation.csv", index=False)

    pd.DataFrame({
        "Document": X_test,
        "Topic_group": y_test,
    }).to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Train size:      {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size:       {len(X_test)}")


if __name__ == "__main__":
    main()