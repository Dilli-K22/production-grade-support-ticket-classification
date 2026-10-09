from pathlib import Path

import joblib


MODEL_PATH = Path("models/ticket_classifier.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict_ticket(text: str) -> tuple[str, list[tuple[str, float]]]:
    if not isinstance(text, str):
        raise TypeError("Ticket text must be a string.")

    if not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    model = load_model()
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    ranked_predictions = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return prediction, ranked_predictions



if __name__ == "__main__":
    ticket = input("Enter support ticket: ")

    predicted_class, probabilities = predict_ticket(ticket)

    print(f"\nPredicted category: {predicted_class}")
    print("\nClass probabilities:")

    for category, probability in probabilities:
        print(f"{category:<25} {probability:.4f}")