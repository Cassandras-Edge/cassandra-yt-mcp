from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cassandra_yt_mcp.types import DownloadResult


class Downloader:
    def __init__(self, work_root: Path, *, cookies_file: Path | None = None) -> None:
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file

    def download(self, *, url: str, job_id: str) -> DownloadResult:
        job_dir = self.work_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(job_dir / "%(id)s.%(ext)s")

        # Worst audio is fine — Parakeet resamples to 16kHz mono anyway.
        # Saves bandwidth and download time (48kbps vs 256kbps).
        # Falls back to combined formats if audio-only isn't available.
        format_attempts = [
            ["-f", "worstaudio/worstaudio*,worst", "-x"],
            ["-f", "worst", "-x"],
            ["-x"],  # no format selector — let yt-dlp pick whatever's available
        ]

        last_error = ""
        for fmt_args in format_attempts:
            cmd = [
                "yt-dlp",
                "--print-json",
                "--no-playlist",
                "--no-warnings",
                "--concurrent-fragments",
                "4",
                *fmt_args,
                "-o",
                output_template,
            ]
            if self.cookies_file:
                cmd.extend(["--cookies", str(self.cookies_file)])
            cmd.append(url)

            try:
                completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("yt-dlp timed out after 600 seconds") from exc

            if completed.returncode == 0:
                break
            last_error = completed.stderr.strip() or "yt-dlp failed"
        else:
            raise RuntimeError(last_error)

        metadata = self._parse_last_json_line(completed.stdout)
        video_id = str(metadata.get("id", "")).strip()
        if not video_id:
            raise RuntimeError("yt-dlp did not return video ID")

        candidates = sorted(job_dir.glob(f"{video_id}.*"))
        if not candidates:
            candidates = sorted(job_dir.iterdir())
        if not candidates:
            raise RuntimeError("Audio file was not produced by yt-dlp")
        return DownloadResult(metadata=metadata, audio_path=str(candidates[0]))

    def expand_playlist(self, url: str) -> list[dict[str, object]]:
        """Expand a playlist URL into individual video entries (metadata only, no download)."""
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--no-warnings",
        ]
        if self.cookies_file:
            cmd.extend(["--cookies", str(self.cookies_file)])
        cmd.append(url)

        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Playlist expansion timed out after 60 seconds") from exc
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "yt-dlp playlist expansion failed")

        entries: list[dict[str, object]] = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    video_id = data.get("id", "")
                    entries.append({
                        "id": video_id,
                        "title": data.get("title", ""),
                        "url": data.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                    })
            except json.JSONDecodeError:
                continue

        if not entries:
            raise RuntimeError("No videos found in playlist")
        return entries

    @staticmethod
    def _parse_last_json_line(stdout: str) -> dict[str, object]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
        raise RuntimeError("Could not parse yt-dlp metadata JSON")
