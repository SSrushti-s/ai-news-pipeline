import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

class DedupeStore:
    def __init__(self, db_path: str = "./output/.dedupe.sqlite3", db_instance=None):
        self.db = db_instance
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_urls ("
            "  url TEXT PRIMARY KEY,"
            "  first_seen TEXT"
            ")"
        )
        self._conn.commit()

    def is_new(self, url: str) -> bool:
        # Check local sqlite
        cur = self._conn.execute("SELECT 1 FROM seen_urls WHERE url = ?", (url,))
        if cur.fetchone() is not None:
            return False
            
        # Check Neon DB if connected
        if self.db and hasattr(self.db, "is_url_seen"):
            if self.db.is_url_seen(url):
                return False

        return True

    def mark_seen(self, url: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, first_seen) VALUES (?, ?)",
            (url, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def reset(self) -> None:
        self._conn.execute("DELETE FROM seen_urls")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()