from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def lock(self) -> Lock:
        return self._lock

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  url TEXT NOT NULL,
                  normalized_url TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'queued',
                  created_at TEXT NOT NULL DEFAULT (datetime('now')),
                  started_at TEXT,
                  completed_at TEXT,
                  error TEXT,
                  video_id TEXT,
                  result_path TEXT,
                  poll_count INTEGER NOT NULL DEFAULT 0,
                  attempt INTEGER NOT NULL DEFAULT 0,
                  retry_after TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at
                ON jobs(status, created_at);

                CREATE INDEX IF NOT EXISTS idx_jobs_normalized_url_status
                ON jobs(normalized_url, status);

                CREATE TABLE IF NOT EXISTS transcripts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  video_id TEXT UNIQUE NOT NULL,
                  normalized_url TEXT UNIQUE,
                  url TEXT,
                  title TEXT,
                  channel TEXT,
                  platform TEXT,
                  duration REAL,
                  upload_date TEXT,
                  description TEXT,
                  thumbnail TEXT,
                  view_count INTEGER,
                  speaker_count INTEGER,
                  word_count INTEGER,
                  confidence REAL,
                  transcribed_at TEXT NOT NULL DEFAULT (datetime('now')),
                  path TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_transcripts_platform_channel
                ON transcripts(platform, channel, transcribed_at DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                  video_id UNINDEXED,
                  title,
                  channel,
                  description,
                  transcript_text
                );

                CREATE TABLE IF NOT EXISTS watch_later_users (
                  user_id TEXT PRIMARY KEY,
                  cookies_b64 TEXT NOT NULL,
                  enabled INTEGER NOT NULL DEFAULT 1,
                  interval_minutes INTEGER NOT NULL DEFAULT 240,
                  last_sync_at TEXT,
                  last_error TEXT,
                  registered_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS watch_later_seen (
                  user_id TEXT NOT NULL,
                  video_id TEXT NOT NULL,
                  title TEXT,
                  first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
                  PRIMARY KEY (user_id, video_id)
                );
                """
            )
            self._conn.commit()

            # Migrations for existing databases
            self._migrate()

    def _migrate(self) -> None:
        """Run incremental schema migrations for existing databases."""
        cursor = self._conn.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "audio_path" not in columns:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN audio_path TEXT")
            self._conn.commit()
        if "cookies_b64" not in columns:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN cookies_b64 TEXT")
            self._conn.commit()
        if "download_progress" not in columns:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN download_progress TEXT")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
