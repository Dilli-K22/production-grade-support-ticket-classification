from src.predict import load_model, predict_ticket


VALID_CLASSES = {
    "Access",
    "Administrative rights",
    "HR Support",
    "Hardware",
    "Internal Project",
    "Miscellaneous",
    "Purchase",
    "Storage",
}


def test_model_can_be_loaded():
    model = load_model()

    assert model is not None


def test_prediction_returns_valid_class():
    ticket = "My laptop screen is not working"

    predicted_class, probabilities = predict_ticket(ticket)

    assert predicted_class in VALID_CLASSES


def test_prediction_probabilities_are_valid():
    ticket = "I cannot access the shared drive"

    predicted_class, probabilities = predict_ticket(ticket)

    probability_values = [probability for _, probability in probabilities]

    assert len(probability_values) == len(VALID_CLASSES)
    assert all(0.0 <= probability <= 1.0 for probability in probability_values)
    assert abs(sum(probability_values) - 1.0) < 1e-6


def test_prediction_returns_ranked_probabilities():
    ticket = "My laptop screen is not working"

    _, probabilities = predict_ticket(ticket)

    probability_values = [probability for _, probability in probabilities]

    assert probability_values == sorted(
        probability_values,
        reverse=True,
    )