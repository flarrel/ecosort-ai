"""Small SQLite persistence layer for prediction history."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ecosort.db"


def get_connection() -> sqlite3.Connection:
    """Open the database and return rows that behave like dictionaries."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Create the database and prediction table when they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )


def save_prediction(filename: str, predicted_class: str, confidence: float) -> int:
    """Save one successful prediction and return its database id."""
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (filename, predicted_class, confidence)
            VALUES (?, ?, ?)
            """,
            (filename, predicted_class, confidence),
        )
        return int(cursor.lastrowid)


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Return the newest predictions first."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp, filename, predicted_class, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_counts() -> dict[str, int]:
    """Return prediction counts grouped by class."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT predicted_class, COUNT(*) AS count
            FROM predictions
            GROUP BY predicted_class
            """
        ).fetchall()
    return {row["predicted_class"]: row["count"] for row in rows}
