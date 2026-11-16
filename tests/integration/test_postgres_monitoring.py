
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy.engine import make_url

from src.monitoring.database import MonitoringDatabase


POSTGRES_URL = os.getenv("TEST_POSTGRES_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="TEST_POSTGRES_DATABASE_URL is not configured",
)


@pytest.fixture
def postgres_database():
    url = make_url(POSTGRES_URL)

    if url.database != "ticket_monitoring_test":
        pytest.fail(
            "PostgreSQL integration tests require "
            "the ticket_monitoring_test database."
        )
    if url.host not in {"localhost", "127.0.0.1"}:
        pytest.fail("PostgreSQL integration tests are restricted to localhost.")

    database = MonitoringDatabase(database_url=POSTGRES_URL)

    with database.connection() as connection:
        connection.execute("DELETE FROM prediction_events")

    yield database

    with database.connection() as connection:
        connection.execute("DELETE FROM prediction_events")

    database.engine.dispose()


def test_postgres_schema_is_initialized(postgres_database):
    with postgres_database.connection() as connection:
        result = connection.execute(
            "SELECT COUNT(*) FROM prediction_events"
        ).fetchone()

    assert result[0] == 0


def test_postgres_records_predictions_and_generates_report(
    postgres_database,
):
    timestamp = datetime.now(timezone.utc).isoformat()

    postgres_database.record_prediction(
        timestamp,
        "Hardware",
        0.95,
        "human_review",
    )

    postgres_database.record_prediction(
        timestamp,
        "HR Support",
        0.85,
        "human_review",
    )

    report = postgres_database.generate_report()

    assert report["total_predictions"] == 2
    assert report["category_distribution"] == {
        "Hardware": 1,
        "HR Support": 1,
    }
    assert report["routing_distribution"] == {
        "human_review": 2,
    }
    assert report["average_confidence"] == pytest.approx(0.90)


def test_postgres_transaction_rolls_back_on_error(
    postgres_database,
):
    before = postgres_database.generate_report()["total_predictions"]

    with pytest.raises(RuntimeError, match="Intentional rollback"):
        with postgres_database.connection() as connection:
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
                    datetime.now(timezone.utc).isoformat(),
                    "Rollback Test",
                    0.99,
                    "human_review",
                ),
            )

            raise RuntimeError("Intentional rollback")

    after = postgres_database.generate_report()["total_predictions"]

    assert before == after


def test_postgres_predictions_persist_across_connections(
    postgres_database,
):
    timestamp = datetime.now(timezone.utc).isoformat()

    postgres_database.record_prediction(
        timestamp,
        "Storage",
        0.88,
        "human_review",
    )

    url = POSTGRES_URL
    postgres_database.engine.dispose()

    fresh_database = MonitoringDatabase(database_url=url)

    try:
        report = fresh_database.generate_report()

        assert report["total_predictions"] == 1
        assert report["category_distribution"] == {
            "Storage": 1,
        }
    finally:
        fresh_database.engine.dispose()
