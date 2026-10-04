import sqlite3
from pathlib import Path

from .config import SOURCE_BACKOFF_RUNS, SOURCE_FAILURE_THRESHOLD, SOURCE_HEALTH_DB


class SourceHealthStore:
    """
    Tracks consecutive failures per source (network error, robots.txt
    disallow, empty/bozo feed) so that at 500-source scale a handful of
    permanently-broken or now-defensive feeds don't get hammered every
    run. After SOURCE_FAILURE_THRESHOLD consecutive failures, a source is
    skipped for SOURCE_BACKOFF_RUNS runs, then retried.

    This is a politeness/efficiency mechanism, not a retry-around-defenses
    mechanism -- a source that goes into backoff is simply left alone.
    """

    def __init__(self, db_path: str = SOURCE_HEALTH_DB):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS source_health ("
            "  name TEXT PRIMARY KEY,"
            "  consecutive_failures INTEGER NOT NULL DEFAULT 0,"
            "  skip_remaining_runs INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.commit()

    def should_skip(self, name: str) -> bool:
        cur = self._conn.execute(
            "SELECT skip_remaining_runs FROM source_health WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        return bool(row and row[0] > 0)

    def record_success(self, name: str) -> None:
        self._conn.execute(
            "INSERT INTO source_health (name, consecutive_failures, skip_remaining_runs) "
            "VALUES (?, 0, 0) "
            "ON CONFLICT(name) DO UPDATE SET consecutive_failures = 0, skip_remaining_runs = 0",
            (name,),
        )
        self._conn.commit()

    def record_failure(self, name: str) -> None:
        cur = self._conn.execute(
            "SELECT consecutive_failures FROM source_health WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        failures = (row[0] if row else 0) + 1
        skip_runs = SOURCE_BACKOFF_RUNS if failures >= SOURCE_FAILURE_THRESHOLD else 0
        self._conn.execute(
            "INSERT INTO source_health (name, consecutive_failures, skip_remaining_runs) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET consecutive_failures = ?, skip_remaining_runs = ?",
            (name, failures, skip_runs, failures, skip_runs),
        )
        self._conn.commit()

    def decrement_skip_counters(self) -> None:
        """Call once per run (for sources that were skipped) so backoff
        eventually expires and the source gets retried."""
        self._conn.execute(
            "UPDATE source_health SET skip_remaining_runs = skip_remaining_runs - 1 "
            "WHERE skip_remaining_runs > 0"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()