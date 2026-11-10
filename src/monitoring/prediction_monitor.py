
import os
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.database import MonitoringDatabase


class PredictionMonitor:
    """
    Tracks prediction activity, category distribution,
    confidence levels, and routing decisions.
    """

    def __init__(self, database=None):
        if database is not None:
            self.database = database
        else:
            database_path = Path(
                os.getenv(
                    "MONITORING_DATABASE_PATH",
                    "data/monitoring.db",
                )
            )
            self.database = MonitoringDatabase(
                database_path=database_path
            )

    def record_prediction(
        self,
        category: str,
        confidence: float,
        auto_routed: bool,
    ) -> None:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Category must be a non-empty string.")

        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise TypeError("Confidence must be numeric.")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        if not isinstance(auto_routed, bool):
            raise TypeError("auto_routed must be a boolean.")

        routing_status = (
            "automatically_routed" if auto_routed else "human_review"
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        self.database.record_prediction(
            timestamp=timestamp,
            predicted_category=category,
            confidence=float(confidence),
            routing_status=routing_status,
        )

    def generate_report(self) -> dict:
        return self.database.generate_report()
