from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from cassandra_yt_mcp.utils.url import is_youtube_url


class YouTubeInfoService:
    def __init__(self, *, cookies_file: Path | None = None) -> None:
        self.cookies_file = cookies_file

    _METADATA_KEYS = [
        "id",
        "title",
        "description",
        "channel",
        "channel_id",
        "channel_url",
        "uploader",
        "upload_date",
        "duration",
        "view_count",
        "like_count",
        "comment_count",
        "age_limit",
        "categories",
        "tags",
        "thumbnail",
        "webpage_url",
        "original_url",
        "language",
        "subtitles",
    ]

    def _run_ytdlp(self, cmd: list[str], *, timeout: int = 30, cookies_file: Path | None = None) -> subprocess.CompletedProcess[str]:
        effective_cookies = cookies_file or self.cookies_file
        if effective_cookies:
            cmd = [*cmd, "--cookies", str(effective_cookies)]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"yt-dlp timed out after {timeout} seconds") from exc
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "yt-dlp failed")
        return completed

    def search(self, query: str, limit: int = 10, cookies_file: Path | None = None) -> list[dict[str, Any]]:
        completed = self._run_ytdlp(
            [
                "yt-dlp",
                f"ytsearch{limit}:{query}",
                "--print",
                "%(id)s\t%(title)s\t%(uploader)s\t%(duration)s\t%(view_count)s",
                "--no-download",
                "--no-warnings",
                "--quiet",
            ],
            cookies_file=cookies_file,
        )
        results: list[dict[str, Any]] = []
        for line in completed.stdout.strip().splitlines():
            parts = line.split("\t", 4)
            if len(parts) < 5:
                continue
            video_id, title, channel, duration, view_count = parts
            results.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel": channel,
                    "duration": _safe_int(duration),
                    "view_count": _safe_int(view_count),
                }
            )
        return results

    def list_channel_videos(self, channel_url: str, limit: int = 20, tab: str = "shorts", cookies_file: Path | None = None) -> list[dict[str, Any]]:
        """List videos from a channel URL. Tab can be 'shorts', 'videos', or 'streams'."""
        url = channel_url.rstrip("/")
        if not url.endswith(f"/{tab}"):
            url = f"{url}/{tab}"
        completed = self._run_ytdlp(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--no-download",
                "--no-warnings",
                "--playlist-end",
                str(limit),
                url,
            ],
            timeout=60,
            cookies_file=cookies_file,
        )
        results: list[dict[str, Any]] = []
        for line in completed.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            video_id = data.get("id", "")
            results.append({
                "video_id": video_id,
                "title": data.get("title", ""),
                "url": data.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                "duration": _safe_int(str(data.get("duration", ""))) if data.get("duration") else None,
                "view_count": _safe_int(str(data.get("view_count", ""))) if data.get("view_count") else None,
            })
        return results

    def get_metadata(self, url: str, cookies_file: Path | None = None) -> dict[str, Any]:
        completed = self._run_ytdlp(
            ["yt-dlp", "--dump-json", "--no-warnings", "--no-download", "--no-playlist", url],
            cookies_file=cookies_file,
        )
        raw = json.loads(completed.stdout)
        return {key: raw[key] for key in self._METADATA_KEYS if key in raw}

    def get_comments(self, url: str, limit: int = 20, sort: str = "top", cookies_file: Path | None = None) -> list[dict[str, Any]]:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-warnings",
            "--skip-download",
            "--no-playlist",
            "--write-comments",
        ]
        if is_youtube_url(url):
            cmd.extend(["--extractor-args", f"youtube:comment_sort={sort};max_comments={limit},all,all"])
        cmd.append(url)
        completed = self._run_ytdlp(cmd, timeout=120, cookies_file=cookies_file)
        data = json.loads(completed.stdout)
        comments: list[dict[str, Any]] = []
        for item in data.get("comments") or []:
            comments.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "author": item.get("author"),
                    "author_id": item.get("author_id"),
                    "like_count": item.get("like_count", 0),
                    "is_pinned": item.get("is_pinned", False),
                    "is_favorited": item.get("is_favorited", False),
                    "parent": item.get("parent", "root"),
                    "timestamp": item.get("timestamp"),
                }
            )
        return comments


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
