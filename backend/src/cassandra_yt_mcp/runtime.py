from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

from cassandra_yt_mcp.config import Settings
from cassandra_yt_mcp.db.database import Database
from cassandra_yt_mcp.db.jobs import JobsRepository
from cassandra_yt_mcp.db.transcripts import TranscriptsRepository
from cassandra_yt_mcp.services.downloader import Downloader
from cassandra_yt_mcp.services.fallback_transcriber import FallbackTranscriber
from cassandra_yt_mcp.services.local_transcriber import LocalTranscriber
from cassandra_yt_mcp.services.storage import StorageService
from cassandra_yt_mcp.services.transcriber import AssemblyAITranscriber
from cassandra_yt_mcp.services.youtube_info import YouTubeInfoService
from cassandra_yt_mcp.utils.url import extract_youtube_video_id, normalize_url

logger = logging.getLogger(__name__)


class BackgroundWorker:
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
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            self.jobs.mark_failed(job_id, str(exc).strip() or "Unknown worker error", attempt)
            # Clear CUDA cache after failures to prevent corrupted allocator state
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._active_count -= 1

    def _process_job(self, *, job_id: str, url: str, normalized_url: str) -> None:
        download = self.downloader.download(url=url, job_id=job_id)
        self.jobs.set_status(job_id, "transcribing")
        audio_path = Path(download.audio_path)
        transcript_result = self.transcriber.transcribe(audio_path)
        persisted = self.storage.persist(
            metadata=download.metadata,
            normalized_url=normalized_url,
            source_url=url,
            transcript=transcript_result,
            temp_audio_path=audio_path,
        )
        word_count = len(transcript_result.text.split())
        speakers = {segment.speaker for segment in transcript_result.segments if segment.speaker}
        self.transcripts.upsert(
            video_id=str(persisted["video_id"]),
            normalized_url=normalized_url,
            url=url,
            path=str(persisted["path"]),
            transcript_text=transcript_result.text,
            title=self._as_str(download.metadata.get("title")),
            channel=self._as_str(download.metadata.get("channel")),
            platform=self._as_str(download.metadata.get("extractor_key")),
            duration=self._as_float(download.metadata.get("duration")),
            upload_date=self._as_str(download.metadata.get("upload_date")),
            description=self._as_str(download.metadata.get("description")),
            thumbnail=self._as_str(download.metadata.get("thumbnail")),
            view_count=self._as_int(download.metadata.get("view_count")),
            speaker_count=len(speakers) if speakers else None,
            word_count=word_count,
            confidence=None,
        )
        self.jobs.mark_completed(job_id, str(persisted["video_id"]), str(persisted["path"]))
        work_dir = self.downloader.work_root / job_id
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _as_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            return float(str(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: object) -> int | None:
        try:
            return int(str(value)) if value is not None else None
        except (TypeError, ValueError):
            return None


class AppRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.database_path)
        self.jobs = JobsRepository(self.database)
        self.transcripts = TranscriptsRepository(self.database)
        self.downloader = Downloader(settings.data_dir / "_work")
        self.storage = StorageService(settings.data_dir)
        self.youtube_info = YouTubeInfoService()
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
        self.transcriber = FallbackTranscriber(
            local=local,
            fallback=fallback,
            enable_local=settings.enable_local_transcription,
        )
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
        self.worker.start()

    def close(self) -> None:
        self.worker.stop()
        self.database.close()

    def enqueue_transcription(self, url: str) -> dict[str, object]:
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
            }
        active = self.jobs.find_active_by_normalized_url(normalized_url)
        if active is not None:
            return {
                "job_id": active["id"],
                "status": active["status"],
                "deduplicated": True,
            }
        job = self.jobs.enqueue(url=url, normalized_url=normalized_url)
        return {"job_id": job["id"], "status": job["status"], "deduplicated": False}

    def get_job_status(self, job_id: str) -> dict[str, object]:
        job = self.jobs.get(job_id)
        if job is None:
            return {"error": "job_not_found", "job_id": job_id}
        if job["status"] in ("queued", "downloading", "transcribing"):
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
