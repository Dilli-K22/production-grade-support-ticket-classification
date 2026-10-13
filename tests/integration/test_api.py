from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app
import pytest
from src.predict import predict_with_model


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict_endpoint_returns_valid_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "ticket_text": "My laptop screen is not working",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_category"] in {
        "Access",
        "Administrative rights",
        "HR Support",
        "Hardware",
        "Internal Project",
        "Miscellaneous",
        "Purchase",
        "Storage",
    }

    probabilities = data["probabilities"]

    assert len(probabilities) == 8
    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities.values()
    )
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_predict_endpoint_rejects_empty_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"ticket_text": ""},
        )

    assert response.status_code == 422


def test_predict_endpoint_rejects_whitespace_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"ticket_text": "   "},
        )

    assert response.status_code == 422

def test_liveness_endpoint():
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint_when_model_is_loaded():
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_loaded": True,
    }

def test_readiness_endpoint_when_model_is_not_loaded():
    with TestClient(app) as client:
        client.app.state.model = None

        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Model is not ready."
    }

def test_application_startup_fails_when_model_cannot_load():
    with patch(
        "src.api.load_model",
        side_effect=RuntimeError("Model loading failed"),
    ):
        with pytest.raises(RuntimeError, match="Model loading failed"):
            with TestClient(app):
                pass

def test_predict_returns_503_when_model_is_not_loaded():
    with TestClient(app) as client:
        client.app.state.model = None

        response = client.post(
            "/predict",
            json={"ticket_text": "My laptop screen is broken"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction service is not ready."
    }


def test_predict_returns_500_when_prediction_fails():
    with patch(
        "src.api.predict_with_model",
        side_effect=RuntimeError("Sensitive internal detail"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={"ticket_text": "My laptop screen is broken"},
            )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred while processing the ticket."
    }
    assert "Sensitive internal detail" not in response.text