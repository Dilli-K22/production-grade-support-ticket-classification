import pytest

from src.monitoring.database import MonitoringDatabase
from src.monitoring.prediction_monitor import PredictionMonitor
from pathlib import Path

from src.settings import Settings
import src.monitoring.prediction_monitor as prediction_monitor_module

@pytest.fixture
def monitor(tmp_path):
    database_path = tmp_path / "test_monitoring.db"
    database = MonitoringDatabase(database_path)

    return PredictionMonitor(database=database)


def test_record_prediction_updates_counts(monitor):
    monitor.record_prediction("Hardware", 0.95, True)
    monitor.record_prediction("HR Support", 0.65, False)

    report = monitor.generate_report()

    assert report["total_predictions"] == 2
    assert report["category_distribution"] == {
        "Hardware": 1,
        "HR Support": 1,
    }
    assert report["routing_distribution"] == {
        "automatically_routed": 1,
        "human_review": 1,
    }
    assert report["average_confidence"] == pytest.approx(0.80)


def test_empty_monitor_returns_empty_report(monitor):
    report = monitor.generate_report()

    assert report["total_predictions"] == 0
    assert report["category_distribution"] == {}
    assert report["routing_distribution"] == {}
    assert report["average_confidence"] is None


def test_empty_category_raises_value_error(monitor):
    with pytest.raises(ValueError, match="Category must be a non-empty string"):
        monitor.record_prediction("", 0.90, True)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_out_of_range_confidence_raises_value_error(monitor, confidence):
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        monitor.record_prediction("Hardware", confidence, True)


def test_non_numeric_confidence_raises_type_error(monitor):
    with pytest.raises(TypeError, match="Confidence must be numeric"):
        monitor.record_prediction("Hardware", "high", True)


def test_non_boolean_routing_status_raises_type_error(monitor):
    with pytest.raises(TypeError, match="auto_routed must be a boolean"):
        monitor.record_prediction("Hardware", 0.90, "yes")


def test_non_string_category_raises_value_error(monitor):
    with pytest.raises(ValueError, match="Category must be a non-empty string"):
        monitor.record_prediction(123, 0.90, True)

@pytest.mark.parametrize("confidence", [True, False])
def test_boolean_confidence_raises_type_error(monitor, confidence):
    with pytest.raises(TypeError, match="Confidence must be numeric"):
        monitor.record_prediction(
            category="Hardware",
            confidence=confidence,
            auto_routed=False,
        )


def test_postgres_database_is_selected_when_url_is_configured(monkeypatch):
    created = {}

    def fake_database(**kwargs):
        created.update(kwargs)
        return object()

    monkeypatch.setattr(
        prediction_monitor_module,
        "get_settings",
        lambda: Settings(
            database_url="postgresql://test:test@localhost:5432/test_db"
        ),
    )
    monkeypatch.setattr(
        prediction_monitor_module,
        "MonitoringDatabase",
        fake_database,
    )

    PredictionMonitor()

    assert created == {
        "database_url": "postgresql://test:test@localhost:5432/test_db"
    }


def test_sqlite_uses_legacy_environment_path(monkeypatch):
    created = {}

    def fake_database(**kwargs):
        created.update(kwargs)
        return object()

    monkeypatch.setenv(
        "MONITORING_DATABASE_PATH",
        "legacy/monitoring.db",
    )

    monkeypatch.setattr(
        prediction_monitor_module,
        "get_settings",
        lambda: Settings(
            database_url=None,
            monitoring_database_path="configured/monitoring.db",
        ),
    )
    monkeypatch.setattr(
        prediction_monitor_module,
        "MonitoringDatabase",
        fake_database,
    )

    PredictionMonitor()

    assert created["database_path"] == Path("legacy/monitoring.db")


def test_injected_database_is_preserved(monkeypatch):
    injected_database = object()

    def unexpected_database_creation(**kwargs):
        raise AssertionError("Database should not be created")

    monkeypatch.setattr(
        prediction_monitor_module,
        "MonitoringDatabase",
        unexpected_database_creation,
    )

    monitor = PredictionMonitor(database=injected_database)

    assert monitor.database is injected_database
