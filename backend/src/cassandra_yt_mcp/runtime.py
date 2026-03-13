from __future__ import annotations

import base64
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

from cassandra_yt_mcp.config import Settings
from cassandra_yt_mcp.db.database import Database
from cassandra_yt_mcp.db.jobs import JobsRepository
from cassandra_yt_mcp.db.transcripts import TranscriptsRepository
from cassandra_yt_mcp.metrics import (
    audio_duration_seconds,
    download_duration_seconds,
    jobs_in_progress,
    jobs_queued,
    jobs_total,
    speaker_count,
    speed_ratio,
    transcription_duration_seconds,
    transcripts_stored,
    word_count,
)
from cassandra_yt_mcp.services.downloader import Downloader
from cassandra_yt_mcp.services.fallback_transcriber import FallbackTranscriber
from cassandra_yt_mcp.services.local_transcriber import LocalTranscriber
from cassandra_yt_mcp.services.remote_transcriber import NoHealthyWorkerError, RemoteTranscriber
from cassandra_yt_mcp.services.storage import StorageService
from cassandra_yt_mcp.services.transcriber import AssemblyAITranscriber
from cassandra_yt_mcp.services.youtube_info import YouTubeInfoService
from cassandra_yt_mcp.utils.url import extract_youtube_video_id, is_playlist_url, normalize_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DownloaderWorker — claims queued jobs, downloads audio, marks downloaded
# ---------------------------------------------------------------------------


class DownloaderWorker:
    """Dedicated download loop. Claims queued jobs and downloads audio to the shared PVC."""

    def __init__(
        self,
        *,
        jobs: JobsRepository,
        downloader: Downloader,
        poll_interval_seconds: int,
        max_workers: int,
    ) -> None:
        self.jobs = jobs
        self.downloader = downloader
        self.poll_interval_seconds = poll_interval_seconds
        self.max_workers = max_workers
        self._stop_event = Event()
        self._thread = Thread(target=self._run_loop, name="downloader-worker", daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dl")
        self._active_count = 0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._active_count >= self.max_workers:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            job = self.jobs.claim_next()
            if job is None:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            jobs_queued.dec()
            self._active_count += 1
            self._executor.submit(self._handle_job, job)

    def _handle_job(self, job: dict[str, object]) -> None:
        job_id = str(job["id"])
        attempt = int(job.get("attempt") or 0)
        try:
            self._download_job(job_id=job_id, url=str(job["url"]))
            jobs_total.labels(status="downloaded", transcriber="n/a").inc()
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient_error(exc)
            logger.exception("Download %s for job %s", "transiently failed" if transient else "failed", job_id)
            self.jobs.mark_failed(job_id, str(exc).strip() or "Download error", attempt, transient=transient)
            jobs_total.labels(status="download_failed", transcriber="n/a").inc()
        finally:
            self._active_count -= 1

    def _download_job(self, *, job_id: str, url: str) -> None:
        jobs_in_progress.labels(phase="downloading").inc()
        try:
            t0 = time.monotonic()
            download = self.downloader.download(url=url, job_id=job_id)
            download_duration_seconds.observe(time.monotonic() - t0)
        finally:
            jobs_in_progress.labels(phase="downloading").dec()

        # Store download metadata alongside the audio for the transcribe stage
        meta_path = Path(download.audio_path).parent / "download_meta.json"
        meta_path.write_text(json.dumps(download.metadata, default=str), encoding="utf-8")

        self.jobs.mark_downloaded(job_id, download.audio_path)
        logger.info("Job %s downloaded to %s", job_id, download.audio_path)


# ---------------------------------------------------------------------------
# TranscribeWorker — claims downloaded jobs, transcribes, stores result
# ---------------------------------------------------------------------------


class TranscribeWorker:
    """Claims downloaded jobs and runs transcription via local or remote GPU."""

    def __init__(
        self,
        *,
        jobs: JobsRepository,
        transcripts: TranscriptsRepository,
        transcriber: object,
        storage: StorageService,
        downloader: Downloader,
        poll_interval_seconds: int,
        max_workers: int,
    ) -> None:
        self.jobs = jobs
        self.transcripts = transcripts
        self.transcriber = transcriber
        self.storage = storage
        self.downloader = downloader
        self.poll_interval_seconds = poll_interval_seconds
        self.max_workers = max_workers
        self._stop_event = Event()
        self._thread = Thread(target=self._run_loop, name="transcribe-worker", daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tx")
        self._active_count = 0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._active_count >= self.max_workers:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            job = self.jobs.claim_next_downloaded()
            if job is None:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            self._active_count += 1
            self._executor.submit(self._handle_job, job)

    def _handle_job(self, job: dict[str, object]) -> None:
        job_id = str(job["id"])
        attempt = int(job.get("attempt") or 0)
        try:
            self._transcribe_and_store(
                job_id=job_id,
                url=str(job["url"]),
                normalized_url=str(job["normalized_url"]),
                audio_path=Path(str(job["audio_path"])),
            )
            transcriber_used = getattr(self.transcriber, "last_transcriber_used", "unknown")
            jobs_total.labels(status="completed", transcriber=transcriber_used).inc()
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient_error(exc)
            logger.exception(
                "Transcription %s for job %s",
                "transiently failed" if transient else "failed",
                job_id,
            )
            self.jobs.mark_failed(job_id, str(exc).strip() or "Transcription error", attempt, transient=transient)
            jobs_total.labels(status="failed", transcriber="unknown").inc()
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._active_count -= 1

    def _transcribe_and_store(
        self, *, job_id: str, url: str, normalized_url: str, audio_path: Path
    ) -> None:
        # Load download metadata saved by the downloader
        meta_path = audio_path.parent / "download_meta.json"
        metadata: dict[str, object] = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        # Transcription phase
        jobs_in_progress.labels(phase="transcribing").inc()
        try:
            t1 = time.monotonic()
            transcript_result = self.transcriber.transcribe(audio_path)
            transcribe_elapsed = time.monotonic() - t1
            transcription_duration_seconds.observe(transcribe_elapsed)
        finally:
            jobs_in_progress.labels(phase="transcribing").dec()

        # Record content metrics
        duration_val = _as_float(metadata.get("duration"))
        if duration_val and duration_val > 0:
            audio_duration_seconds.observe(duration_val)
            if transcribe_elapsed > 0:
                speed_ratio.observe(duration_val / transcribe_elapsed)

        persisted = self.storage.persist(
            metadata=metadata,
            normalized_url=normalized_url,
            source_url=url,
            transcript=transcript_result,
            temp_audio_path=audio_path,
        )
        wc = len(transcript_result.text.split())
        word_count.observe(wc)
        speakers = {segment.speaker for segment in transcript_result.segments if segment.speaker}
        if speakers:
            speaker_count.observe(len(speakers))
        self.transcripts.upsert(
            video_id=str(persisted["video_id"]),
            normalized_url=normalized_url,
            url=url,
            path=str(persisted["path"]),
            transcript_text=transcript_result.text,
            title=_as_str(metadata.get("title")),
            channel=_as_str(metadata.get("channel")),
            platform=_as_str(metadata.get("extractor_key")),
            duration=_as_float(metadata.get("duration")),
            upload_date=_as_str(metadata.get("upload_date")),
            description=_as_str(metadata.get("description")),
            thumbnail=_as_str(metadata.get("thumbnail")),
            view_count=_as_int(metadata.get("view_count")),
            speaker_count=len(speakers) if speakers else None,
            word_count=wc,
            confidence=None,
        )
        self.jobs.mark_completed(job_id, str(persisted["video_id"]), str(persisted["path"]))
        transcripts_stored.inc()

        # Clean up work directory
        work_dir = audio_path.parent
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# BackgroundWorker — standalone mode (download + transcribe in one process)
# ---------------------------------------------------------------------------


class BackgroundWorker:
    """Legacy all-in-one worker for standalone mode (download + transcribe in one process)."""

    def __init__(
        self,
        *,
        jobs: JobsRepository,
        transcripts: TranscriptsRepository,
        downloader: Downloader,
        transcriber: object,
        storage: StorageService,
        poll_interval_seconds: int,
        max_workers: int,
    ) -> None:
        self.jobs = jobs
        self.transcripts = transcripts
        self.downloader = downloader
        self.transcriber = transcriber
        self.storage = storage
        self.poll_interval_seconds = poll_interval_seconds
        self.max_workers = max_workers
        self._stop_event = Event()
        self._thread = Thread(target=self._run_loop, name="cassandra-yt-mcp-worker", daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yt-job")
        self._active_count = 0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._active_count >= self.max_workers:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            job = self.jobs.claim_next()
            if job is None:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            jobs_queued.dec()
            self._active_count += 1
            self._executor.submit(self._handle_job, job)

    def _handle_job(self, job: dict[str, object]) -> None:
        job_id = str(job["id"])
        attempt = int(job.get("attempt") or 0)
        try:
            self._process_job(
                job_id=job_id,
                url=str(job["url"]),
                normalized_url=str(job["normalized_url"]),
            )
            transcriber_used = getattr(self.transcriber, "last_transcriber_used", "unknown")
            jobs_total.labels(status="completed", transcriber=transcriber_used).inc()
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient_error(exc)
            logger.exception("Job %s %s", job_id, "transiently failed" if transient else "failed")
            self.jobs.mark_failed(job_id, str(exc).strip() or "Unknown worker error", attempt, transient=transient)
            jobs_total.labels(status="failed", transcriber="unknown").inc()
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._active_count -= 1

    def _process_job(self, *, job_id: str, url: str, normalized_url: str) -> None:
        # Download phase
        jobs_in_progress.labels(phase="downloading").inc()
        try:
            t0 = time.monotonic()
            download = self.downloader.download(url=url, job_id=job_id)
            download_duration_seconds.observe(time.monotonic() - t0)
        finally:
            jobs_in_progress.labels(phase="downloading").dec()

        self.jobs.set_status(job_id, "transcribing")

        # Transcription phase
        jobs_in_progress.labels(phase="transcribing").inc()
        try:
            audio_path = Path(download.audio_path)
            t1 = time.monotonic()
            transcript_result = self.transcriber.transcribe(audio_path)
            transcribe_elapsed = time.monotonic() - t1
            transcription_duration_seconds.observe(transcribe_elapsed)
        finally:
            jobs_in_progress.labels(phase="transcribing").dec()

        # Record content metrics
        duration_val = _as_float(download.metadata.get("duration"))
        if duration_val and duration_val > 0:
            audio_duration_seconds.observe(duration_val)
            if transcribe_elapsed > 0:
                speed_ratio.observe(duration_val / transcribe_elapsed)

        persisted = self.storage.persist(
            metadata=download.metadata,
            normalized_url=normalized_url,
            source_url=url,
            transcript=transcript_result,
            temp_audio_path=audio_path,
        )
        wc = len(transcript_result.text.split())
        word_count.observe(wc)
        speakers = {segment.speaker for segment in transcript_result.segments if segment.speaker}
        if speakers:
            speaker_count.observe(len(speakers))
        self.transcripts.upsert(
            video_id=str(persisted["video_id"]),
            normalized_url=normalized_url,
            url=url,
            path=str(persisted["path"]),
            transcript_text=transcript_result.text,
            title=_as_str(download.metadata.get("title")),
            channel=_as_str(download.metadata.get("channel")),
            platform=_as_str(download.metadata.get("extractor_key")),
            duration=_as_float(download.metadata.get("duration")),
            upload_date=_as_str(download.metadata.get("upload_date")),
            description=_as_str(download.metadata.get("description")),
            thumbnail=_as_str(download.metadata.get("thumbnail")),
            view_count=_as_int(download.metadata.get("view_count")),
            speaker_count=len(speakers) if speakers else None,
            word_count=wc,
            confidence=None,
        )
        self.jobs.mark_completed(job_id, str(persisted["video_id"]), str(persisted["path"]))
        transcripts_stored.inc()
        work_dir = self.downloader.work_root / job_id
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_transient_error(exc: Exception) -> bool:
    """Transient errors don't count toward the retry limit."""
    if isinstance(exc, NoHealthyWorkerError):
        return True
    msg = str(exc).lower()
    # Auth/cookie errors are permanent — won't resolve without human intervention
    permanent_patterns = ("sign in", "confirm you're not a bot", "cookies", "authentication")
    if any(p in msg for p in permanent_patterns):
        return False
    transient_patterns = (
        "server disconnected",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )
    return any(p in msg for p in transient_patterns)


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: object) -> float | None:
    try:
        return float(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# AppRuntime — wired up per role
# ---------------------------------------------------------------------------


class AppRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.database_path)
        self.jobs = JobsRepository(self.database)
        self.transcripts = TranscriptsRepository(self.database)
        self.storage = StorageService(settings.data_dir)

        # Set up yt-dlp cookie file from base64-encoded env var
        cookies_file = self._setup_cookies(settings)
        self.downloader = Downloader(settings.data_dir / "_work", cookies_file=cookies_file)
        self.youtube_info = YouTubeInfoService(cookies_file=cookies_file)

        if settings.role == "coordinator":
            self.transcriber = self._build_transcriber(settings)
            self.worker = TranscribeWorker(
                jobs=self.jobs,
                transcripts=self.transcripts,
                transcriber=self.transcriber,
                storage=self.storage,
                downloader=self.downloader,
                poll_interval_seconds=settings.poll_interval_seconds,
                max_workers=settings.max_workers,
            )
        else:
            # standalone mode: all-in-one
            self.transcriber = self._build_transcriber(settings)
            self.worker = BackgroundWorker(
                jobs=self.jobs,
                transcripts=self.transcripts,
                downloader=self.downloader,
                transcriber=self.transcriber,
                storage=self.storage,
                poll_interval_seconds=settings.poll_interval_seconds,
                max_workers=settings.max_workers,
            )

    def start(self) -> None:
        self.jobs.recover_stale()
        transcripts_stored.set(self.transcripts.count())
        jobs_queued.set(self.jobs.count_queued())
        self.worker.start()

    def close(self) -> None:
        self.worker.stop()
        self.database.close()

    @staticmethod
    def _setup_cookies(settings: Settings) -> Path | None:
        if not settings.ytdlp_cookies:
            return None
        cookies_path = settings.data_dir / "_cookies.txt"
        try:
            cookies_path.write_bytes(base64.b64decode(settings.ytdlp_cookies))
            logger.info("Wrote yt-dlp cookies to %s", cookies_path)
            return cookies_path
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode YTDLP_COOKIES")
            return None

    @staticmethod
    def _build_transcriber(settings: Settings) -> object:
        if settings.role == "coordinator":
            logger.info("Coordinator mode — dispatching to GPU workers: %s", settings.gpu_workers)
            return RemoteTranscriber(settings.gpu_workers)

        # standalone mode: local GPU with optional AssemblyAI fallback
        local = (
            LocalTranscriber(huggingface_token=settings.huggingface_token)
            if settings.enable_local_transcription
            else None
        )
        fallback = (
            AssemblyAITranscriber(api_key=settings.assemblyai_api_key)
            if settings.assemblyai_api_key
            else None
        )
        return FallbackTranscriber(
            local=local,
            fallback=fallback,
            enable_local=settings.enable_local_transcription,
        )

    def enqueue_transcription(self, url: str) -> dict[str, object]:
        # Check for playlist URL — expand and enqueue each video
        if is_playlist_url(url):
            return self._enqueue_playlist(url)

        return self._enqueue_single(url)

    def _enqueue_single(self, url: str) -> dict[str, object]:
        normalized_url = normalize_url(url)
        existing = self.transcripts.get_by_normalized_url(normalized_url)
        if existing is None:
            video_id = extract_youtube_video_id(normalized_url)
            if video_id:
                existing = self.transcripts.get_by_video_id(video_id)
        if existing is not None:
            return {
                "status": "completed",
                "deduplicated": True,
                "video_id": existing["video_id"],
                "transcript_path": existing["path"],
                "metadata": existing,
            }
        active = self.jobs.find_active_by_normalized_url(normalized_url)
        if active is not None:
            return {
                "job_id": active["id"],
                "status": active["status"],
                "deduplicated": True,
            }
        job = self.jobs.enqueue(url=url, normalized_url=normalized_url)
        jobs_queued.inc()
        result: dict[str, object] = {"job_id": job["id"], "status": job["status"], "deduplicated": False}
        try:
            result["metadata"] = self.youtube_info.get_metadata(url)
        except Exception:  # noqa: BLE001
            logger.debug("Could not fetch metadata for %s at enqueue time", url)
        return result

    def _enqueue_playlist(self, url: str) -> dict[str, object]:
        try:
            entries = self.downloader.expand_playlist(url)
        except RuntimeError as exc:
            return {"error": "playlist_expansion_failed", "message": str(exc)}

        jobs: list[dict[str, object]] = []
        skipped = 0
        for entry in entries:
            video_url = entry.get("url", "")
            if not video_url:
                skipped += 1
                continue
            result = self._enqueue_single(video_url)
            jobs.append(result)
            if result.get("deduplicated"):
                skipped += 1

        return {
            "playlist": True,
            "total": len(entries),
            "enqueued": len(jobs) - skipped,
            "skipped": skipped,
            "jobs": jobs,
        }

    def get_job_status(self, job_id: str) -> dict[str, object]:
        job = self.jobs.get(job_id)
        if job is None:
            return {"error": "job_not_found", "job_id": job_id}
        if job["status"] in ("queued", "downloading", "downloaded", "transcribing"):
            self.jobs.increment_poll_count(job_id)
            refreshed = self.jobs.get(job_id) or job
            poll_count = int(refreshed.get("poll_count") or 0)
            payload: dict[str, object] = {
                **refreshed,
                "retry_after": min(5 * (2 ** (poll_count // 3)), 60),
            }
            if refreshed.get("retry_after"):
                payload["waiting_until"] = refreshed["retry_after"]
                payload["attempt"] = int(refreshed.get("attempt") or 0)
            return payload
        return job


class DownloaderRuntime:
    """Minimal runtime for the downloader role — no transcriber, no API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.database_path)
        self.jobs = JobsRepository(self.database)
        cookies_file = AppRuntime._setup_cookies(settings)
        self.downloader = Downloader(settings.data_dir / "_work", cookies_file=cookies_file)
        self.worker = DownloaderWorker(
            jobs=self.jobs,
            downloader=self.downloader,
            poll_interval_seconds=settings.poll_interval_seconds,
            max_workers=settings.download_concurrency,
        )

    def start(self) -> None:
        self.jobs.recover_stale()
        jobs_queued.set(self.jobs.count_queued())
        self.worker.start()

    def close(self) -> None:
        self.worker.stop()
        self.database.close()
