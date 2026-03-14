from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Callable

from cassandra_yt_mcp.db.transcripts import TranscriptsRepository
from cassandra_yt_mcp.db.watch_later import WatchLaterRepository
from cassandra_yt_mcp.services.downloader import Downloader

logger = logging.getLogger(__name__)

WL_PLAYLIST_URL = "https://www.youtube.com/playlist?list=WL"


class WatchLaterService:
    def __init__(
        self,
        *,
        watch_later_repo: WatchLaterRepository,
        transcripts_repo: TranscriptsRepository,
        work_root: Path,
        enqueue_fn: Callable[[str, str | None], dict[str, object]],
    ) -> None:
        self.watch_later = watch_later_repo
        self.transcripts = transcripts_repo
        self.work_root = work_root
        self.enqueue_fn = enqueue_fn

    def sync(self, user_id: str, cookies_b64: str) -> dict[str, Any]:
        cookies_dir = self.work_root / "_watch_later"
        cookies_dir.mkdir(parents=True, exist_ok=True)
        cookies_path = cookies_dir / f"{user_id}_cookies.txt"

        try:
            cookies_path.write_bytes(base64.b64decode(cookies_b64))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode cookies: {exc}") from exc

        try:
            downloader = Downloader(self.work_root, cookies_file=cookies_path)
            entries = downloader.expand_playlist(WL_PLAYLIST_URL)
        except RuntimeError as exc:
            msg = str(exc)
            if "No videos found" in msg:
                # Empty watch later — mark sync complete, return empty result
                self.watch_later.update_last_sync(user_id)
                return {"total": 0, "new_count": 0, "already_seen": 0, "already_transcribed": 0, "jobs": []}
            raise
        finally:
            if cookies_path.exists():
                cookies_path.unlink()

        already_seen = 0
        already_transcribed = 0
        new_entries: list[dict[str, str | None]] = []
        jobs: list[dict[str, object]] = []

        for entry in entries:
            video_id = str(entry.get("id", ""))
            title = str(entry.get("title", ""))
            url = str(entry.get("url", ""))

            if not video_id or not url:
                continue

            if self.watch_later.is_seen(user_id, video_id):
                already_seen += 1
                continue

            if self.transcripts.get_by_video_id(video_id) is not None:
                already_transcribed += 1
                new_entries.append({"video_id": video_id, "title": title})
                continue

            try:
                result = self.enqueue_fn(url, cookies_b64)
                jobs.append(result)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to enqueue %s for watch later sync", url)

            new_entries.append({"video_id": video_id, "title": title})

        if new_entries:
            self.watch_later.mark_seen_batch(user_id, new_entries)

        self.watch_later.update_last_sync(user_id)

        return {
            "total": len(entries),
            "new_count": len(jobs),
            "already_seen": already_seen,
            "already_transcribed": already_transcribed,
            "jobs": jobs,
        }
