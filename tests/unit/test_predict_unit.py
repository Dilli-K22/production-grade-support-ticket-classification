import pytest
from unittest.mock import patch

from src.predict import predict_ticket


def test_empty_ticket_raises_value_error():
    with pytest.raises(ValueError, match="Ticket text cannot be empty"):
        predict_ticket("")


def test_whitespace_ticket_raises_value_error():
    with pytest.raises(ValueError, match="Ticket text cannot be empty"):
        predict_ticket("   \n\t  ")


def test_non_string_ticket_raises_type_error():
    with pytest.raises(TypeError, match="Ticket text must be a string"):
        predict_ticket(123)


def test_invalid_input_does_not_load_model():
    with patch("src.predict.load_model") as mock_load_model:
        with pytest.raises(ValueError):
            predict_ticket("")

        mock_load_model.assert_not_called()


def test_valid_input_loads_model_and_predicts():
    with patch("src.predict.load_model") as mock_load_model:
        mock_model = mock_load_model.return_value
        mock_model.predict.return_value = ["Hardware"]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_model.classes_ = ["Hardware", "Storage"]

        predicted_class, probabilities = predict_ticket(
            "My laptop screen is broken"
        )

        assert predicted_class == "Hardware"
        assert probabilities == [
            ("Hardware", 0.8),
            ("Storage", 0.2),
        ]
        mock_load_model.assert_called_once()
        mock_model.predict.assert_called_once_with(
            ["My laptop screen is broken"]
        )