import sqlite3
from pathlib import Path

from .models import Incident


class IncidentStore:
    """SQLite persistence for the incident history."""

    def __init__(self, database_path: str | Path = "data/incidents.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    captured_path TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )

    def add(self, incident: Incident) -> int:
        x1, y1, x2, y2 = incident.box
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents
                    (activity, confidence, x1, y1, x2, y2, captured_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident.activity,
                    incident.confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    incident.captured_path,
                    incident.timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
