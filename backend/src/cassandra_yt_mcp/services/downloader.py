from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from cassandra_yt_mcp.types import DownloadResult

logger = logging.getLogger(__name__)

# Matches yt-dlp progress lines like:
# [download]  45.2% of  10.50MiB at  2.50MiB/s ETA 00:04
# [download]  45.2% of ~  10.50MiB at  2.50MiB/s ETA 00:04 (frag 3/7)
_PROGRESS_RE = re.compile(
    r"\[download\]\s+(?P<percent>[\d.]+)%\s+of\s+~?\s*(?P<total>\S+)"
    r"\s+at\s+(?P<speed>\S+)"
    r"\s+ETA\s+(?P<eta>\S+)"
    r"(?:\s+\(frag\s+(?P<frag_current>\d+)/(?P<frag_total>\d+)\))?"
)

ProgressCallback = Callable[[dict[str, object]], None]


class Downloader:
    def __init__(self, work_root: Path, *, cookies_file: Path | None = None) -> None:
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file

    def download(
        self,
        *,
        url: str,
        job_id: str,
        cookies_file: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> DownloadResult:
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
            [],  # last resort — download any video+audio, ffmpeg will extract audio later
        ]

        last_error = ""
        for fmt_args in format_attempts:
            cmd = [
                "yt-dlp",
                "--print-json",
                "--no-playlist",
                "--no-warnings",
                "--newline",  # print progress on new lines instead of \r
                "--concurrent-fragments",
                "16",
                "--live-from-start",
                *fmt_args,
                "-o",
                output_template,
            ]
            effective_cookies = cookies_file or self.cookies_file
            if effective_cookies:
                cmd.extend(["--cookies", str(effective_cookies)])
            cmd.append(url)

            try:
                stdout, stderr, returncode = self._run_with_progress(cmd, on_progress, timeout=600)
            except subprocess.TimeoutExpired:
                # For live streams: timeout means we grabbed what was available
                logger.info("yt-dlp timed out (likely live stream), merging fragments")
                merged = self._merge_fragments(job_dir)
                if merged:
                    return DownloadResult(metadata={}, audio_path=str(merged))
                raise RuntimeError("yt-dlp timed out with no usable output")

            if returncode == 0:
                break
            last_error = stderr.strip() or "yt-dlp failed"
        else:
            raise RuntimeError(last_error)

        metadata = self._parse_last_json_line(stdout)
        video_id = str(metadata.get("id", "")).strip()
        if not video_id:
            raise RuntimeError("yt-dlp did not return video ID")

        candidates = sorted(job_dir.glob(f"{video_id}.*"))
        if not candidates:
            candidates = sorted(job_dir.iterdir())
        if not candidates:
            raise RuntimeError("Audio file was not produced by yt-dlp")

        audio_path = candidates[0]
        # Pre-convert to 16kHz mono WAV so downstream skips ffmpeg
        if audio_path.suffix != ".wav":
            wav_path = audio_path.with_suffix(".wav")
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                 str(wav_path)],
                capture_output=True, timeout=300,
            )
            if result.returncode == 0:
                audio_path.unlink()
                audio_path = wav_path
                logger.info("Pre-converted to WAV: %s", wav_path.name)

        return DownloadResult(metadata=metadata, audio_path=str(audio_path))

    def expand_playlist(self, url: str, cookies_file: Path | None = None) -> list[dict[str, object]]:
        """Expand a playlist URL into individual video entries (metadata only, no download)."""
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--no-warnings",
        ]
        effective_cookies = cookies_file or self.cookies_file
        if effective_cookies:
            cmd.extend(["--cookies", str(effective_cookies)])
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
                    entry_url = data.get("url") or data.get("webpage_url", "")
                    entries.append({
                        "id": video_id,
                        "title": data.get("title", ""),
                        "url": entry_url,
                    })
            except json.JSONDecodeError:
                continue

        if not entries:
            raise RuntimeError("No videos found in playlist")
        return entries

    @staticmethod
    def _run_with_progress(
        cmd: list[str],
        on_progress: ProgressCallback | None,
        timeout: int,
    ) -> tuple[str, str, int]:
        """Run yt-dlp, streaming stderr for progress updates.

        Returns (stdout, stderr, returncode).
        Monitors both stdout and stderr to prevent pipe buffer deadlocks
        (e.g. when PO token plugins write verbose output to stdout).
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout_chunks: list[str] = []
        stderr_lines: list[str] = []
        import selectors  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
        deadline = _time.monotonic() + timeout
        open_streams = 2

        while open_streams > 0:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)

            events = sel.select(timeout=min(remaining, 2.0))
            if events:
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        sel.unregister(key.fileobj)
                        open_streams -= 1
                        continue
                    if key.data == "stderr":
                        stderr_lines.append(line)
                        if on_progress:
                            m = _PROGRESS_RE.search(line)
                            if m:
                                progress: dict[str, object] = {
                                    "percent": float(m.group("percent")),
                                    "total_size": m.group("total"),
                                    "speed": m.group("speed"),
                                    "eta": m.group("eta"),
                                }
                                if m.group("frag_current"):
                                    progress["fragment"] = f"{m.group('frag_current')}/{m.group('frag_total')}"
                                on_progress(progress)
                    else:
                        stdout_chunks.append(line)
            elif proc.poll() is not None:
                # Process exited, drain remaining output
                for line in proc.stdout:
                    stdout_chunks.append(line)
                for line in proc.stderr:
                    stderr_lines.append(line)
                break

        sel.close()
        proc.wait()
        return "".join(stdout_chunks), "".join(stderr_lines), proc.returncode

    @staticmethod
    def _merge_fragments(job_dir: Path) -> Path | None:
        """Merge yt-dlp .part fragment files into a single audio file via ffmpeg."""
        part_files = sorted(job_dir.glob("*.part"))
        if not part_files:
            return None

        # Find the base .part file and its fragments
        base = part_files[0]
        frags = sorted(job_dir.glob(f"{base.name}-Frag*"), key=lambda p: int(p.stem.split("Frag")[-1]))
        if not frags:
            return None

        # Concatenate fragments into the base .part file
        merged_path = base.with_suffix(".m4a")
        with merged_path.open("wb") as out:
            for frag in frags:
                out.write(frag.read_bytes())

        logger.info("Merged %d fragments into %s (%.1fMB)", len(frags), merged_path.name,
                     merged_path.stat().st_size / 1024 / 1024)

        # Clean up fragments
        for frag in frags:
            frag.unlink()
        base.unlink()

        return merged_path

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
