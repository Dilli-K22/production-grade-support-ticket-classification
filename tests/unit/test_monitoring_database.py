from datetime import datetime, timezone, timedelta

import pytest

from src.monitoring.database import MonitoringDatabase
import sqlite3


@pytest.fixture
def database(tmp_path):
    database_path = tmp_path / "test_monitoring.db"
    return MonitoringDatabase(database_path)


def test_database_initialization(database):
    with database.get_connection() as connection:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'prediction_events'
            """
        ).fetchone()

    assert result is not None


def test_record_prediction_saves_event(database):
    timestamp = datetime.now(timezone.utc).isoformat()

    database.record_prediction(
        timestamp,
        "Hardware",
        0.95,
        "human_review",
    )

    with database.get_connection() as connection:
        result = connection.execute(
            """
            SELECT timestamp, predicted_category, confidence, routing_status
            FROM prediction_events
            """
        ).fetchone()

    assert result == (
        timestamp,
        "Hardware",
        0.95,
        "human_review",
    )


def test_multiple_predictions_are_saved(database):
    timestamp = datetime.now(timezone.utc).isoformat()

    database.record_prediction(
        timestamp, "Hardware", 0.95, "human_review"
    )
    database.record_prediction(
        timestamp, "HR Support", 0.80, "automatically_routed"
    )

    report = database.generate_report()

    assert report["total_predictions"] == 2


def test_database_report_calculates_correct_statistics(database):
    timestamp = datetime.now(timezone.utc).isoformat()

    database.record_prediction(
        timestamp, "Hardware", 0.90, "human_review"
    )
    database.record_prediction(
        timestamp, "Hardware", 0.80, "human_review"
    )
    database.record_prediction(
        timestamp, "HR Support", 0.70, "automatically_routed"
    )

    report = database.generate_report()

    assert report["total_predictions"] == 3
    assert report["category_distribution"] == {
        "Hardware": 2,
        "HR Support": 1,
    }
    assert report["routing_distribution"] == {
        "human_review": 2,
        "automatically_routed": 1,
    }
    assert report["average_confidence"] == pytest.approx(0.80)


def test_records_persist_across_new_connections(database):
    timestamp = datetime.now(timezone.utc).isoformat()

    database.record_prediction(
        timestamp,
        "Storage",
        0.88,
        "human_review",
    )

    with database.get_connection() as connection:
        result = connection.execute(
            "SELECT COUNT(*) FROM prediction_events"
        ).fetchone()[0]

    assert result == 1

def test_report_includes_category_metrics(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    database.record_prediction(
        "2026-11-06T09:00:00+00:00",
        "Hardware",
        0.8,
        "human_review",
    )
    database.record_prediction(
        "2026-11-06T10:00:00+00:00",
        "Hardware",
        0.6,
        "human_review",
    )
    database.record_prediction(
        "2026-11-06T11:00:00+00:00",
        "HR Support",
        0.9,
        "automatically_routed",
    )

    report = database.generate_report()

    assert report["category_metrics"]["Hardware"]["prediction_count"] == 2
    assert report["category_metrics"]["Hardware"]["average_confidence"] == 0.7
    assert report["category_metrics"]["HR Support"]["prediction_count"] == 1


def test_report_includes_daily_prediction_counts(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    database.record_prediction(
        "2026-11-06T09:00:00+00:00",
        "Hardware",
        0.8,
        "human_review",
    )
    database.record_prediction(
        "2026-11-06T10:00:00+00:00",
        "HR Support",
        0.9,
        "human_review",
    )
    database.record_prediction(
        "2026-11-07T09:00:00+00:00",
        "Hardware",
        0.7,
        "human_review",
    )

    report = database.generate_report()

    assert report["daily_prediction_counts"] == {
        "2026-11-06": 2,
        "2026-11-07": 1,
    }


def test_report_includes_daily_routing_distribution(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    database.record_prediction(
        "2026-11-06T09:00:00+00:00",
        "Hardware",
        0.8,
        "human_review",
    )
    database.record_prediction(
        "2026-11-06T10:00:00+00:00",
        "HR Support",
        0.9,
        "automatically_routed",
    )

    report = database.generate_report()

    assert report["daily_routing_distribution"]["2026-11-06"] == {
        "human_review": 1,
        "automatically_routed": 1,
    }

@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_database_rejects_invalid_confidence(tmp_path, confidence):
    database = MonitoringDatabase(tmp_path / "test.db")

    with pytest.raises(ValueError):
        database.record_prediction(
            "2026-11-06T09:00:00+00:00",
            "Hardware",
            confidence,
            "human_review",
        )


def test_database_rejects_non_numeric_confidence(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with pytest.raises(TypeError):
        database.record_prediction(
            "2026-11-06T09:00:00+00:00",
            "Hardware",
            "high",
            "human_review",
        )


def test_database_rejects_invalid_routing_status(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with pytest.raises(ValueError):
        database.record_prediction(
            "2026-11-06T09:00:00+00:00",
            "Hardware",
            0.8,
            "unknown_status",
        )


def test_database_rejects_empty_category(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with pytest.raises(ValueError):
        database.record_prediction(
            "2026-11-06T09:00:00+00:00",
            " ",
            0.8,
            "human_review",
        )


def test_database_indexes_are_created(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with database.get_connection() as connection:
        indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(prediction_events)"
            ).fetchall()
        }

    assert "idx_prediction_category" in indexes
    assert "idx_prediction_timestamp" in indexes
    assert "idx_prediction_routing" in indexes

def test_database_uses_wal_journal_mode(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with database.get_connection() as connection:
        journal_mode = connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]

    assert journal_mode.lower() == "wal"


def test_database_enables_foreign_keys(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with database.get_connection() as connection:
        foreign_keys = connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]

    assert foreign_keys == 1


def test_database_rejects_non_positive_timeout(tmp_path):
    with pytest.raises(ValueError, match="Database timeout"):
        MonitoringDatabase(
            database_path=tmp_path / "test.db",
            timeout=0,
        )


def test_database_accepts_custom_timeout(tmp_path):
    database = MonitoringDatabase(
        database_path=tmp_path / "test.db",
        timeout=5.0,
    )

    assert database.timeout == 5.0

def test_connection_is_closed_after_successful_operation(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with database.connection() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_connection_is_closed_after_failed_operation(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    connection = None

    with pytest.raises(sqlite3.OperationalError):
        with database.connection() as connection:
            connection.execute(
                "SELECT * FROM nonexistent_table"
            )

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")

def test_failed_transaction_is_rolled_back(tmp_path):
    database = MonitoringDatabase(tmp_path / "test.db")

    with pytest.raises(sqlite3.OperationalError):
        with database.connection() as connection:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    timestamp,
                    predicted_category,
                    confidence,
                    routing_status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "2026-11-10T09:00:00+00:00",
                    "Hardware",
                    0.8,
                    "human_review",
                ),
            )

            connection.execute(
                "INSERT INTO nonexistent_table VALUES (1)"
            )

    report = database.generate_report()

    assert report["total_predictions"] == 0


def test_time_window_report_includes_recent_predictions(database):
    now = datetime.now(timezone.utc)

    database.record_prediction(
        (now - timedelta(hours=2)).isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )
    database.record_prediction(
        (now - timedelta(hours=12)).isoformat(),
        "HR Support",
        0.80,
        "automatically_routed",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            now - timedelta(hours=24),
        )

    assert report["total_predictions"] == 2
    assert report["category_distribution"] == {
        "Hardware": 1,
        "HR Support": 1,
    }
    assert report["routing_distribution"] == {
        "human_review": 1,
        "automatically_routed": 1,
    }
    assert report["average_confidence"] == pytest.approx(0.85)


def test_time_window_report_excludes_old_predictions(database):
    now = datetime.now(timezone.utc)

    database.record_prediction(
        (now - timedelta(hours=25)).isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            now - timedelta(hours=24),
        )

    assert report["total_predictions"] == 0
    assert report["category_distribution"] == {}
    assert report["routing_distribution"] == {}
    assert report["average_confidence"] is None


def test_seven_day_window_excludes_predictions_older_than_seven_days(
    database,
):
    now = datetime.now(timezone.utc)

    database.record_prediction(
        (now - timedelta(days=3)).isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )
    database.record_prediction(
        (now - timedelta(days=8)).isoformat(),
        "HR Support",
        0.80,
        "automatically_routed",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            now - timedelta(days=7),
        )

    assert report["total_predictions"] == 1
    assert report["category_distribution"] == {
        "Hardware": 1,
    }


def test_time_window_report_handles_empty_database(database):
    now = datetime.now(timezone.utc)

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            now - timedelta(hours=24),
        )

    assert report == {
        "total_predictions": 0,
        "category_distribution": {},
        "routing_distribution": {},
        "average_confidence": None,
    }


def test_generate_report_includes_time_window_metrics(database):
    now = datetime.now(timezone.utc)

    database.record_prediction(
        now.isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )

    report = database.generate_report()

    assert "time_window_metrics" in report
    assert "last_24_hours" in report["time_window_metrics"]
    assert "last_7_days" in report["time_window_metrics"]

    assert (
        report["time_window_metrics"]["last_24_hours"]["total_predictions"]
        == 1
    )
    assert (
        report["time_window_metrics"]["last_7_days"]["total_predictions"]
        == 1
    )


def test_24_hour_window_includes_exact_boundary(database):
    now = datetime.now(timezone.utc)
    boundary = now - timedelta(hours=24)

    database.record_prediction(
        boundary.isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            boundary,
        )

    assert report["total_predictions"] == 1


def test_24_hour_window_excludes_prediction_before_boundary(database):
    now = datetime.now(timezone.utc)
    boundary = now - timedelta(hours=24)
    older_timestamp = boundary - timedelta(microseconds=1)

    database.record_prediction(
        older_timestamp.isoformat(),
        "Hardware",
        0.90,
        "human_review",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            boundary,
        )

    assert report["total_predictions"] == 0


def test_7_day_window_includes_exact_boundary(database):
    now = datetime.now(timezone.utc)
    boundary = now - timedelta(days=7)

    database.record_prediction(
        boundary.isoformat(),
        "HR Support",
        0.85,
        "human_review",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            boundary,
        )

    assert report["total_predictions"] == 1


def test_7_day_window_excludes_prediction_before_boundary(database):
    now = datetime.now(timezone.utc)
    boundary = now - timedelta(days=7)
    older_timestamp = boundary - timedelta(microseconds=1)

    database.record_prediction(
        older_timestamp.isoformat(),
        "HR Support",
        0.85,
        "human_review",
    )

    with database.connection() as connection:
        report = database._generate_time_window_report(
            connection,
            boundary,
        )

    assert report["total_predictions"] == 0


def test_database_rejects_invalid_timestamp(database):
    with pytest.raises(
        ValueError,
        match="valid ISO 8601 datetime",
    ):
        database.record_prediction(
            "not-a-date",
            "Hardware",
            0.90,
            "human_review",
        )


def test_database_rejects_empty_timestamp(database):
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        database.record_prediction(
            " ",
            "Hardware",
            0.90,
            "human_review",
        )


def test_database_rejects_timezone_naive_timestamp(database):
    with pytest.raises(
        ValueError,
        match="timezone information",
    ):
        database.record_prediction(
            "2026-11-10T10:00:00",
            "Hardware",
            0.90,
            "human_review",
        )


def test_database_normalizes_timestamp_to_utc(database):
    database.record_prediction(
        "2026-11-10T10:00:00+05:30",
        "Hardware",
        0.90,
        "human_review",
    )

    with database.connection() as connection:
        stored_timestamp = connection.execute(
            "SELECT timestamp FROM prediction_events"
        ).fetchone()[0]

    assert stored_timestamp == "2026-11-10T04:30:00+00:00"


def test_database_accepts_utc_timestamp(database):
    timestamp = "2026-11-10T04:30:00+00:00"

    database.record_prediction(
        timestamp,
        "Hardware",
        0.90,
        "human_review",
    )

    with database.connection() as connection:
        stored_timestamp = connection.execute(
            "SELECT timestamp FROM prediction_events"
        ).fetchone()[0]

    assert stored_timestamp == timestamp
