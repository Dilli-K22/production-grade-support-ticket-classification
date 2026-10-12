
from fastapi.testclient import TestClient

from src.api import app


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