from __future__ import annotations

import logging
import uuid
from typing import Any

from cassandra_yt_mcp.db.database import Database
from cassandra_yt_mcp.metrics import retries_total

ACTIVE_STATUSES = ("queued", "downloading", "downloaded", "transcribing")
MAX_ATTEMPTS = 3
_BASE_RETRY_DELAY_SECONDS = 30
_MAX_RETRY_DELAY_SECONDS = 600


logger = logging.getLogger(__name__)


class JobsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def recover_stale(self) -> int:
        """Reset zombie downloading/transcribing jobs back to queued on startup."""
        with self.db.lock:
            cur = self.db.conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', started_at = NULL
                WHERE status IN ('downloading', 'transcribing')
                """
            )
            self.db.conn.commit()
            count = cur.rowcount
        if count:
            logger.warning("Recovered %d stale jobs back to queued", count)
        return count

    def count_queued(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()
        return int(row[0]) if row else 0

    def enqueue(self, url: str, normalized_url: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        with self.db.lock:
            self.db.conn.execute(
                """
                INSERT INTO jobs(id, url, normalized_url, status)
                VALUES (?, ?, ?, 'queued')
                """,
                (job_id, url, normalized_url),
            )
            self.db.conn.commit()
        job = self.get(job_id)
        if job is None:
            raise RuntimeError("Failed to create job")
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row is not None else None

    def find_active_by_normalized_url(self, normalized_url: str) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        row = self.db.conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE normalized_url = ? AND status IN ({placeholders})
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (normalized_url, *ACTIVE_STATUSES),
        ).fetchone()
        return dict(row) if row is not None else None

    def claim_next(self) -> dict[str, Any] | None:
        with self.db.lock:
            self.db.conn.execute("BEGIN IMMEDIATE")
            row = self.db.conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                  AND (retry_after IS NULL OR retry_after <= datetime('now'))
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self.db.conn.commit()
                return None

            self.db.conn.execute(
                """
                UPDATE jobs
                SET status = 'downloading', started_at = datetime('now')
                WHERE id = ?
                """,
                (row["id"],),
            )
            self.db.conn.commit()

        return self.get(str(row["id"]))

    def increment_poll_count(self, job_id: str) -> None:
        with self.db.lock:
            self.db.conn.execute(
                "UPDATE jobs SET poll_count = poll_count + 1 WHERE id = ?",
                (job_id,),
            )
            self.db.conn.commit()

    def mark_downloaded(self, job_id: str, audio_path: str) -> None:
        with self.db.lock:
            self.db.conn.execute(
                "UPDATE jobs SET status = 'downloaded', audio_path = ? WHERE id = ?",
                (audio_path, job_id),
            )
            self.db.conn.commit()

    def claim_next_downloaded(self) -> dict[str, Any] | None:
        with self.db.lock:
            self.db.conn.execute("BEGIN IMMEDIATE")
            row = self.db.conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'downloaded'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self.db.conn.commit()
                return None

            self.db.conn.execute(
                "UPDATE jobs SET status = 'transcribing' WHERE id = ?",
                (row["id"],),
            )
            self.db.conn.commit()

        return self.get(str(row["id"]))

    def set_status(self, job_id: str, status: str) -> None:
        with self.db.lock:
            self.db.conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
            self.db.conn.commit()

    def mark_completed(self, job_id: str, video_id: str, result_path: str) -> None:
        with self.db.lock:
            self.db.conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', completed_at = datetime('now'), video_id = ?, result_path = ?, error = NULL
                WHERE id = ?
                """,
                (video_id, result_path, job_id),
            )
            self.db.conn.commit()

    def mark_failed(self, job_id: str, error: str, attempt: int = 0, *, transient: bool = False) -> None:
        next_attempt = attempt if transient else attempt + 1
        with self.db.lock:
            if next_attempt < MAX_ATTEMPTS:
                retries_total.inc()
                delay = min(_BASE_RETRY_DELAY_SECONDS * (2**attempt), _MAX_RETRY_DELAY_SECONDS)
                self.db.conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', started_at = NULL, attempt = ?,
                        retry_after = datetime('now', ? || ' seconds'), error = ?
                    WHERE id = ?
                    """,
                    (next_attempt, str(delay), error[:2000], job_id),
                )
            else:
                self.db.conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', completed_at = datetime('now'), error = ?
                    WHERE id = ?
                    """,
                    (error[:2000], job_id),
                )
            self.db.conn.commit()
