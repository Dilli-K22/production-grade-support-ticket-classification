
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


DATABASE_PATH = Path("data/monitoring.db")

VALID_ROUTING_STATUSES = {
    "automatically_routed",
    "human_review",
}


class _PostgresConnection:
    """Adapts SQLAlchemy connections to the existing database interface."""

    def __init__(self, connection):
        self._connection = connection
        self._transaction = connection.begin()

    def execute(self, statement, parameters=()):
        # Existing database queries use SQLite-style positional placeholders.
        # Psycopg uses %s for positional parameters.
        postgres_statement = statement.replace("?", "%s")
        return self._connection.exec_driver_sql(
            postgres_statement,
            parameters,
        )

    def commit(self):
        if self._transaction.is_active:
            self._transaction.commit()

    def rollback(self):
        if self._transaction.is_active:
            self._transaction.rollback()

    def close(self):
        self._connection.close()


class MonitoringDatabase:
    """Persistent monitoring storage supporting SQLite and PostgreSQL."""

    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
        timeout: float = 10.0,
        database_url: str | None = None,
    ):
        if timeout <= 0:
            raise ValueError("Database timeout must be greater than zero.")

        self.database_path = Path(database_path)
        self.timeout = timeout
        self.database_url = database_url
        self.engine = None

        if database_url is not None:
            if not isinstance(database_url, str):
                raise TypeError("Database URL must be a string.")

            if not database_url.strip():
                raise ValueError("Database URL must not be empty.")

            url = make_url(database_url)

            if url.get_backend_name() != "postgresql":
                raise ValueError("Only PostgreSQL URLs are supported.")

            # Explicitly select Psycopg 3 as the PostgreSQL driver.
            url = url.set(drivername="postgresql+psycopg")

            self.engine = create_engine(
                url,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": max(1, int(timeout)),
                },
            )
        else:
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.initialize_database()

    @property
    def is_postgres(self) -> bool:
        return self.engine is not None

    def get_connection(self):
        if self.is_postgres:
            connection = self.engine.connect()
            return _PostgresConnection(connection)

        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
        )

        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    def initialize_database(self):
        with self.connection() as connection:
            if self.is_postgres:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        predicted_category TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL
                            CHECK(confidence >= 0 AND confidence <= 1),
                        routing_status TEXT NOT NULL
                            CHECK(routing_status IN (
                                'automatically_routed',
                                'human_review'
                            ))
                    )
                    """
                )
            else:
                connection.execute("PRAGMA journal_mode = WAL")

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        predicted_category TEXT NOT NULL,
                        confidence REAL NOT NULL
                            CHECK(confidence >= 0 AND confidence <= 1),
                        routing_status TEXT NOT NULL
                            CHECK(routing_status IN (
                                'automatically_routed',
                                'human_review'
                            ))
                    )
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_category
                ON prediction_events(predicted_category)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_timestamp
                ON prediction_events(timestamp)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_routing
                ON prediction_events(routing_status)
                """
            )

    def record_prediction(
        self,
        timestamp: str,
        predicted_category: str,
        confidence: float,
        routing_status: str,
    ) -> None:

        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError("Timestamp must be a non-empty string.")

        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            raise ValueError(
                "Timestamp must be a valid ISO 8601 datetime."
            ) from None

        if (
            parsed_timestamp.tzinfo is None
            or parsed_timestamp.utcoffset() is None
        ):
            raise ValueError(
                "Timestamp must include timezone information."
            )

        timestamp = parsed_timestamp.astimezone(
            timezone.utc
        ).isoformat()

        if (
            not isinstance(predicted_category, str)
            or not predicted_category.strip()
        ):
            raise ValueError(
                "Predicted category must be a non-empty string."
            )

        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise TypeError("Confidence must be numeric.")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        if routing_status not in VALID_ROUTING_STATUSES:
            raise ValueError("Invalid routing status.")

        with self.connection() as connection:
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
                    timestamp,
                    predicted_category,
                    float(confidence),
                    routing_status,
                ),
            )

    def _generate_time_window_report(
        self,
        connection,
        start_time: datetime,
    ) -> dict:

        start_timestamp = start_time.astimezone(
            timezone.utc
        ).isoformat()

        total_predictions = connection.execute(
            """
            SELECT COUNT(*)
            FROM prediction_events
            WHERE timestamp >= ?
            """,
            (start_timestamp,),
        ).fetchone()[0]

        category_rows = connection.execute(
            """
            SELECT predicted_category, COUNT(*)
            FROM prediction_events
            WHERE timestamp >= ?
            GROUP BY predicted_category
            """,
            (start_timestamp,),
        ).fetchall()

        routing_rows = connection.execute(
            """
            SELECT routing_status, COUNT(*)
            FROM prediction_events
            WHERE timestamp >= ?
            GROUP BY routing_status
            """,
            (start_timestamp,),
        ).fetchall()

        average_confidence = connection.execute(
            """
            SELECT AVG(confidence)
            FROM prediction_events
            WHERE timestamp >= ?
            """,
            (start_timestamp,),
        ).fetchone()[0]

        return {
            "total_predictions": total_predictions,
            "category_distribution": dict(category_rows),
            "routing_distribution": dict(routing_rows),
            "average_confidence": average_confidence,
        }

    def generate_report(self) -> dict:
        with self.connection() as connection:
            now = datetime.now(timezone.utc)

            last_24_hours = self._generate_time_window_report(
                connection,
                now - timedelta(hours=24),
            )

            last_7_days = self._generate_time_window_report(
                connection,
                now - timedelta(days=7),
            )

            total_predictions = connection.execute(
                "SELECT COUNT(*) FROM prediction_events"
            ).fetchone()[0]

            category_rows = connection.execute(
                """
                SELECT predicted_category, COUNT(*)
                FROM prediction_events
                GROUP BY predicted_category
                """
            ).fetchall()

            routing_rows = connection.execute(
                """
                SELECT routing_status, COUNT(*)
                FROM prediction_events
                GROUP BY routing_status
                """
            ).fetchall()

            average_confidence = connection.execute(
                "SELECT AVG(confidence) FROM prediction_events"
            ).fetchone()[0]

            category_metric_rows = connection.execute(
                """
                SELECT
                    predicted_category,
                    COUNT(*),
                    AVG(confidence)
                FROM prediction_events
                GROUP BY predicted_category
                ORDER BY predicted_category
                """
            ).fetchall()

            daily_prediction_rows = connection.execute(
                """
                SELECT
                    SUBSTR(timestamp, 1, 10),
                    COUNT(*)
                FROM prediction_events
                GROUP BY SUBSTR(timestamp, 1, 10)
                ORDER BY SUBSTR(timestamp, 1, 10)
                """
            ).fetchall()

            daily_routing_rows = connection.execute(
                """
                SELECT
                    SUBSTR(timestamp, 1, 10),
                    routing_status,
                    COUNT(*)
                FROM prediction_events
                GROUP BY SUBSTR(timestamp, 1, 10), routing_status
                ORDER BY SUBSTR(timestamp, 1, 10)
                """
            ).fetchall()

        category_metrics = {
            category: {
                "prediction_count": count,
                "average_confidence": avg_confidence,
            }
            for category, count, avg_confidence in category_metric_rows
        }

        daily_routing_distribution = {}

        for date, routing_status, count in daily_routing_rows:
            daily_routing_distribution.setdefault(date, {})[
                routing_status
            ] = count

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_predictions": total_predictions,
            "category_distribution": dict(category_rows),
            "routing_distribution": dict(routing_rows),
            "average_confidence": average_confidence,
            "category_metrics": category_metrics,
            "daily_prediction_counts": dict(daily_prediction_rows),
            "daily_routing_distribution": daily_routing_distribution,
            "time_window_metrics": {
                "last_24_hours": last_24_hours,
                "last_7_days": last_7_days,
            },
        }

    @contextmanager
    def connection(self):
        connection = self.get_connection()

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
