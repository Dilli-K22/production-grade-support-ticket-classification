from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app, settings
import pytest
from src.predict import predict_with_model
from pydantic import SecretStr

TEST_MONITORING_API_KEY = "test-monitoring-api-key"

@pytest.fixture
def test_database(tmp_path, monkeypatch):
    database_path = tmp_path / "test_monitoring.db"

    monkeypatch.setenv(
        "MONITORING_DATABASE_PATH",
        str(database_path),
    )

    monkeypatch.setattr(
        settings,
        "monitoring_api_key",
        SecretStr(TEST_MONITORING_API_KEY),
    )

    return database_path


def test_health_endpoint(test_database):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict_endpoint_returns_valid_prediction(test_database):
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


def test_predict_endpoint_rejects_empty_ticket(test_database):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"ticket_text": ""},
        )

    assert response.status_code == 422


def test_predict_endpoint_rejects_whitespace_ticket(test_database):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"ticket_text": "   "},
        )

    assert response.status_code == 422

def test_predict_endpoint_rejects_oversized_ticket(test_database):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"ticket_text": "A" * 5001},
        )

    assert response.status_code == 422

def test_liveness_endpoint(test_database):
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint_when_model_is_loaded(test_database):
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_loaded": True,
        "monitoring_available": True,
    }

def test_readiness_endpoint_when_model_is_not_loaded(test_database):
    with TestClient(app) as client:
        client.app.state.model = None

        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Application dependencies are not ready."
    }

def test_application_startup_fails_when_model_cannot_load(test_database):
    with patch(
        "src.api.load_model",
        side_effect=RuntimeError("Model loading failed"),
    ):
        with pytest.raises(RuntimeError, match="Model loading failed"):
            with TestClient(app):
                pass

def test_predict_returns_503_when_model_is_not_loaded(test_database):
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

def test_predict_returns_500_when_prediction_fails(test_database):
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

def test_monitoring_rejects_missing_api_key(test_database):
    with TestClient(app) as client:
        response = client.get("/monitoring")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key."
    }

def test_monitoring_rejects_invalid_api_key(test_database):
    with TestClient(app) as client:
        response = client.get(
            "/monitoring",
            headers={"X-API-Key": "incorrect-api-key"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key."
    }

def test_monitoring_returns_503_when_api_key_is_not_configured(
    test_database,
    monkeypatch,
):
    monkeypatch.setattr(settings, "monitoring_api_key", None)

    with TestClient(app) as client:
        response = client.get(
            "/monitoring",
            headers={"X-API-Key": TEST_MONITORING_API_KEY},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Monitoring authentication is not configured."
    }

def test_monitoring_endpoint_returns_valid_report(test_database):
    with TestClient(app) as client:
        response = client.get("/monitoring", headers={"X-API-Key": TEST_MONITORING_API_KEY},)

    assert response.status_code == 200

    data = response.json()

    assert "generated_at" in data
    assert "total_predictions" in data
    assert "category_distribution" in data
    assert "routing_distribution" in data
    assert "average_confidence" in data
    assert "category_metrics" in data
    assert "daily_prediction_counts" in data
    assert "daily_routing_distribution" in data
    assert "time_window_metrics" in data

    assert "last_24_hours" in data["time_window_metrics"]
    assert "last_7_days" in data["time_window_metrics"]

    for window in ("last_24_hours", "last_7_days"):
        metrics = data["time_window_metrics"][window]

        assert "total_predictions" in metrics
        assert "category_distribution" in metrics
        assert "routing_distribution" in metrics
        assert "average_confidence" in metrics

def test_monitoring_records_prediction(test_database):
    headers = {"X-API-Key": TEST_MONITORING_API_KEY}

    with TestClient(app) as client:
        before = client.get(
            "/monitoring",
            headers=headers,
        ).json()["total_predictions"]

        prediction_response = client.post(
            "/predict",
            json={
                "ticket_text": "My laptop screen is not working",
            },
        )

        after = client.get(
            "/monitoring",
            headers=headers,
        ).json()["total_predictions"]

    assert prediction_response.status_code == 200
    assert after == before + 1

def test_monitoring_returns_503_when_monitor_is_unavailable(test_database):
    with TestClient(app) as client:
        client.app.state.prediction_monitor = None

        response = client.get("/monitoring", headers={"X-API-Key": TEST_MONITORING_API_KEY},)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Monitoring is not available."
    }

def test_prediction_succeeds_when_monitoring_recording_fails(test_database):
    with patch(
        "src.api.PredictionMonitor.record_prediction",
        side_effect=RuntimeError("Database unavailable"),
    ):
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

    assert len(data["probabilities"]) == 8

def test_readiness_reports_monitoring_status(test_database):
    with TestClient(app) as client:
        client.app.state.prediction_monitor = None

        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Application dependencies are not ready."
    }

def test_application_startup_fails_when_monitoring_cannot_initialize(
    test_database,
):
    with patch(
        "src.api.PredictionMonitor",
        side_effect=RuntimeError("Monitoring initialization failed"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Monitoring initialization failed",
        ):
            with TestClient(app):
                pass

    assert app.state.model is None
    assert app.state.prediction_monitor is None
